# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""LRU/size-based file eviction for the HiCache file storage backend.

``HiCacheFile`` is a thin raw-bytes store: it suffixes keys, reads/writes
``.bin`` pages, and answers existence queries. Everything that bounds how much
disk those pages consume -- the LRU recency index, per-file size accounting,
free-space probing, scanning pre-existing files on startup, and unlinking
victims -- lives here so the backend stays a plain key/value store.

A backend constructs one evictor and drives it through a small lifecycle::

    touch(key, path)                 # read hit / already-on-disk: bump recency
    reserve(key, n_bytes) -> bool    # admit a new write, evicting if needed
        commit(key)                  #   write landed on disk
        abort(key)                   #   write failed; release the reservation
    clear()                          # backend wiped all files

When eviction is not configured the evictor is inert: ``reserve`` always admits
and the other calls are no-ops, so the backend behaves as unbounded storage.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Set, Tuple

from sglang.srt.environ import envs
from sglang.srt.utils.common import human_readable_int

logger = logging.getLogger(__name__)


def _parse_size_to_bytes(value: Any) -> int:
    """Parse a size to bytes via human_readable_int (e.g. '200G', '1Gi', '1048576').
    None / empty / '0' disables; an invalid value also disables (with a warning)."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    s = str(value).strip()
    if not s or s == "0":
        return 0
    try:
        return max(0, human_readable_int(s))
    except (argparse.ArgumentTypeError, ValueError):
        logger.warning(f"Invalid size {value!r} for HiCacheFile; disabling.")
        return 0


class LRUFileEvictor:
    """Bounds the on-disk size of a HiCacheFile directory via LRU eviction.

    Tracks one ``.bin`` file per suffixed key (oldest at the front of the LRU),
    enforces an optional byte cap and an optional free-space watermark, and
    unlinks the least-recently-used files to stay within those bounds. Eviction
    config comes from ``extra_config`` (per-backend, takes precedence) falling
    back to the ``SGLANG_HICACHE_FILE_BACKEND_*`` env vars.
    """

    def __init__(
        self,
        file_path: str,
        *,
        owned_suffixes: Tuple[str, ...],
        tp_rank: int,
        extra_config: Optional[dict] = None,
        on_evict: Optional[Callable[[str], None]] = None,
        sibling_keys: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.file_path = file_path
        # Files ending in one of these suffixes are created and evicted by this
        # rank; other files in the shared directory are only read.
        self._owned_suffixes = owned_suffixes
        self._tp_rank = tp_rank
        self._on_evict = on_evict
        # Maps a tracked stem to the stems of the other files storing parts of
        # the same page group (KV payload + sidecar pools). Eviction unlinks
        # the whole group so a restore never finds an incomplete set.
        self.sibling_keys = sibling_keys
        # Optional backend hook that deletes untracked leftover files (tmp
        # writes, stale-format pages) and returns the bytes reclaimed. Called
        # before a watermark-driven eviction; must not take self._lock.
        self.sweep_callback: Optional[Callable[[], int]] = None

        # suffixed_key -> file size in bytes; oldest at front.
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._pending_writes: Set[str] = set()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

        self._load_config(extra_config or {})

        self._eviction_enabled = self.max_size_bytes > 0 or self.min_free_bytes > 0
        if not self._eviction_enabled:
            return

        # Clamp max_size to the filesystem capacity so a too-large cap can't OOM tmpfs.
        fs = self._fs_stats()
        if fs is not None and self.max_size_bytes > 0:
            safe_max = max(0, fs[0] - self.min_free_bytes)
            if self.max_size_bytes > safe_max:
                logger.warning(
                    f"HiCacheFile max_size exceeds filesystem capacity; "
                    f"clamping to {safe_max} B."
                )
                self.max_size_bytes = safe_max

        self._scan_existing_files()
        with self._lock:
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:
                self._evict_locked(0)
            if self.min_free_bytes > 0:
                self._enforce_free_space_locked(0)
        logger.info(
            f"HiCacheFile eviction enabled: cap={self.max_size_bytes} B, "
            f"watermark={self.eviction_ratio:.2f}, min_free={self.min_free_bytes} B, "
            f"existing={self._total_bytes} B ({len(self._lru)} entries)"
        )

    def _load_config(self, extra: dict) -> None:
        # extra_config (per-backend) takes precedence over env vars.
        def _cfg(key, env):
            val = extra.get(key)
            return env.get() if val is None else val

        self.max_size_bytes = _parse_size_to_bytes(
            _cfg("max_size", envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE)
        )
        self.min_free_bytes = _parse_size_to_bytes(
            _cfg("min_free_space", envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE)
        )

        ratio_raw = _cfg(
            "eviction_ratio", envs.SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO
        )
        try:
            self.eviction_ratio = float(ratio_raw)
        except (TypeError, ValueError):
            self.eviction_ratio = 0.9
        if not (0.0 < self.eviction_ratio <= 1.0):
            self.eviction_ratio = 0.9

    @property
    def enabled(self) -> bool:
        """True when a cap or free-space watermark is set."""
        return self._eviction_enabled

    def _owns(self, suffixed_key: str) -> bool:
        return suffixed_key.endswith(self._owned_suffixes)

    def reserve(self, suffixed_key: str, value_bytes: int, key: str = "") -> bool:
        """Admit a new write of ``value_bytes``, evicting LRU victims as needed.

        On success the key is pre-reserved at MRU and flagged in-flight so a
        concurrent ``reserve`` won't evict it before the file is committed; the
        caller must then call ``commit`` (write landed) or ``abort`` (write
        failed). Returns ``False`` -- reserving nothing -- when the write is
        refused: this rank is not the storage owner, the value is larger than
        the cap, there is no evictable space, or the free-space watermark cannot
        be met. When eviction is not configured the write is always admitted.
        """
        if not self._eviction_enabled:
            return True  # unbounded storage: nothing to enforce
        if not self._owns(suffixed_key):
            logger.warning(
                f"HiCacheFile rank {self._tp_rank} does not own key {key}; "
                f"not caching it because file eviction is enabled."
            )
            return False
        if self.max_size_bytes > 0 and value_bytes > self.max_size_bytes:
            logger.warning(
                f"HiCacheFile: value {value_bytes} B exceeds cap "
                f"{self.max_size_bytes} B; not caching {key}"
            )
            return False

        with self._lock:
            # Cap-based eviction: evict, then bail if still over cap.
            if (
                self.max_size_bytes > 0
                and (self._total_bytes + value_bytes) > self.max_size_bytes
            ):
                self._evict_locked(value_bytes)
                if (self._total_bytes + value_bytes) > self.max_size_bytes:
                    logger.warning(
                        f"HiCacheFile: no evictable space for {value_bytes} B "
                        f"under cap {self.max_size_bytes} B; not caching {key}"
                    )
                    return False
            # Free-space watermark.
            if self.min_free_bytes > 0 and not self._enforce_free_space_locked(
                value_bytes
            ):
                logger.warning(
                    f"HiCacheFile: filesystem hosting {self.file_path!r} "
                    f"would fall below min_free={self.min_free_bytes} B "
                    f"after writing {value_bytes} B; refusing {key} "
                    f"to avoid OOM/ENOSPC."
                )
                return False
            # Pre-reserve at MRU so a concurrent evict won't grab this slot.
            prev = self._lru.pop(suffixed_key, None)
            if prev is not None:
                self._total_bytes -= prev
            self._lru[suffixed_key] = value_bytes
            self._pending_writes.add(suffixed_key)
            self._total_bytes += value_bytes
        return True

    def commit(self, suffixed_key: str) -> None:
        """Mark a reserved write as durably on disk (clears its in-flight flag)."""
        if not self._eviction_enabled:
            return
        with self._lock:
            self._pending_writes.discard(suffixed_key)

    def abort(self, suffixed_key: str) -> None:
        """Release a reservation whose write failed: drop it and refund the bytes."""
        if not self._eviction_enabled:
            return
        with self._lock:
            cur = self._lru.pop(suffixed_key, None)
            self._pending_writes.discard(suffixed_key)
            if cur is not None:
                self._total_bytes -= cur

    def touch(self, suffixed_key: str, tensor_path: str) -> None:
        """Mark key as MRU, adopting an untracked on-disk file if needed.

        Also refreshes the file mtime: the startup scan rebuilds the LRU
        from mtimes, so without a read-time bump a hot page written long
        ago looks stale after a restart and becomes the first victim.
        """
        if not self._eviction_enabled or not self._owns(suffixed_key):
            return
        tracked = False
        with self._lock:
            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
                tracked = True
        if not tracked:
            # Untracked file: stat without holding the lock.
            try:
                size = os.path.getsize(tensor_path)
            except OSError:
                return
            with self._lock:
                if suffixed_key in self._lru:
                    self._lru.move_to_end(suffixed_key, last=True)
                else:
                    self._lru[suffixed_key] = size
                    self._total_bytes += size
        try:
            # ~1us on a hot inode (measured); a vanished file is benign.
            os.utime(tensor_path, None)
        except OSError:
            pass

    def touch_many(self, stem_path_pairs) -> None:
        """Batched touch: same semantics as ``touch`` per file, with two lock
        passes and one utime sweep instead of per-file locking.

        The lock is only held for OrderedDict updates; stats and utimes run
        outside so a large batch does not block concurrent reserves.
        """
        if not self._eviction_enabled:
            return
        untracked = []
        with self._lock:
            for stem, path in stem_path_pairs:
                if not self._owns(stem):
                    continue
                if stem in self._lru:
                    self._lru.move_to_end(stem, last=True)
                else:
                    untracked.append((stem, path))
        adopted = []
        for stem, path in untracked:
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            adopted.append((stem, size))
        if adopted:
            with self._lock:
                for stem, size in adopted:
                    if stem in self._lru:
                        self._lru.move_to_end(stem, last=True)
                    else:
                        self._lru[stem] = size
                        self._total_bytes += size
        for _, path in stem_path_pairs:
            try:
                os.utime(path, None)
            except OSError:
                pass

    def clear(self) -> None:
        """Reset all bookkeeping after the backend has removed the files."""
        with self._lock:
            self._lru.clear()
            self._pending_writes.clear()
            self._total_bytes = 0

    def _fs_stats(self) -> Optional[tuple]:
        """(total, available) bytes for the filesystem; None if unavailable."""
        try:
            st = os.statvfs(self.file_path)
        except (OSError, AttributeError):
            return None
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        return total, free

    def _enforce_free_space_locked(self, value_bytes: int) -> bool:
        """Evict until writing value_bytes still leaves min_free_bytes free.
        Caller holds _lock. Returns False if the write can't be satisfied.

        Evicting every live entry frees at most the reclaimable bytes, so
        when even that cannot bring the filesystem above the watermark the
        write is refused up front instead of trashing the cache to no
        effect.
        """
        if self.min_free_bytes <= 0:
            return True
        fs = self._fs_stats()
        if fs is None:
            return True  # cannot probe -> permissive, fall back to OS errors
        free = fs[1]
        if free - value_bytes >= self.min_free_bytes:
            # Ample headroom: the eviction loop below would be a no-op, so
            # skip the (lock-held) orphan sweep and reclaimable sum.
            return True
        if self.sweep_callback is not None:
            # Untracked leftovers are reclaimable without evicting live
            # pages; the sweep does its own directory IO and never takes
            # self._lock.
            free += self.sweep_callback()
        own_reclaimable = sum(
            size
            for key, size in self._lru.items()
            if key not in self._pending_writes
        )
        if free + own_reclaimable - value_bytes < self.min_free_bytes:
            return False
        # tmpfs frees space on unlink, so credit reclaimed bytes back to the
        # estimate rather than re-probing statvfs on every eviction.
        self._evict_while(
            lambda reclaimed: (free + reclaimed) - value_bytes < self.min_free_bytes
        )
        # Re-probe: external writers may have changed free space meanwhile.
        fs = self._fs_stats()
        if fs is None:
            return True
        return fs[1] - value_bytes >= self.min_free_bytes

    def _scan_existing_files(self) -> None:
        """Seed LRU index from disk on startup (oldest mtime first)."""
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        entries = []
        for fn in names:
            if not fn.endswith(".bin"):
                continue
            stem = fn[:-4]
            # Only files belonging to this rank/model.
            if not self._owns(stem):
                continue
            fp = os.path.join(self.file_path, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            entries.append((st.st_mtime, stem, st.st_size))
        entries.sort(key=lambda e: e[0])  # oldest first
        for _, stem, size in entries:
            self._lru[stem] = size
            self._total_bytes += size

    def _group_siblings_locked(self, evict_stem: str) -> List[str]:
        """Other stems storing parts of the same page group. Caller holds
        _lock."""
        if self.sibling_keys is None:
            return []
        try:
            siblings = self.sibling_keys(evict_stem)
        except Exception as e:
            # Derivation is pure string math; failure must not stop eviction.
            logger.warning(
                f"HiCacheFile sibling derivation failed for {evict_stem}: {e}"
            )
            return []
        return [s for s in siblings if s != evict_stem]

    def _evict_one_lru_locked(self) -> Tuple[str, int]:
        """Evict the single oldest evictable LRU entry, with its page group.

        The shared pop / skip-pending / unlink / ``_total_bytes`` step driven by
        `_evict_while`. Returns ``(outcome, freed_bytes)``:

        - ``("evicted", n)``: oldest entry (and its sidecar siblings) dropped
          from the index; ``n`` disk bytes reclaimed (0 for already-gone
          files).
        - ``("skipped", 0)``: the entry or one of its siblings is an in-flight
          write; re-pinned at MRU so the group is not torn apart and the
          writer is not evicted out from under itself.
        - ``("stop", 0)``: nothing evictable (empty index) or the unlink failed
          (entry re-pinned at LRU); the caller should stop its eviction loop.
        """
        if not self._lru:
            return "stop", 0
        evict_stem, evict_size = self._lru.popitem(last=False)  # oldest
        if evict_stem in self._pending_writes:
            # Keep in-flight reservations; their file isn't committed yet.
            self._lru[evict_stem] = evict_size
            return "skipped", 0
        siblings = self._group_siblings_locked(evict_stem)
        if any(sib in self._pending_writes for sib in siblings):
            # A sidecar of this page is being written; evicting the payload
            # now would leave an incomplete group on disk. Defer the group.
            self._lru[evict_stem] = evict_size
            return "skipped", 0
        tensor_path = os.path.join(self.file_path, f"{evict_stem}.bin")
        try:
            os.remove(tensor_path)
            freed = evict_size
            if self._on_evict is not None:
                self._on_evict(evict_stem)
        except FileNotFoundError:
            freed = 0  # file already gone; still drop the stale index entry
            if self._on_evict is not None:
                self._on_evict(evict_stem)
        except OSError as e:
            logger.warning(f"HiCacheFile eviction failed for {evict_stem}: {e}")
            self._lru[evict_stem] = evict_size
            self._lru.move_to_end(evict_stem, last=False)
            return "stop", 0
        self._total_bytes -= evict_size
        for sib in siblings:
            sib_path = os.path.join(self.file_path, f"{sib}.bin")
            sib_size = self._lru.pop(sib, None)
            try:
                os.remove(sib_path)
            except FileNotFoundError:
                pass  # already gone; still drop the stale index entry
            except OSError as e:
                logger.warning(f"HiCacheFile sibling eviction failed for {sib}: {e}")
                if sib_size is not None:
                    self._lru[sib] = sib_size
                continue
            if sib_size is not None:
                freed += sib_size
                self._total_bytes -= sib_size
            if self._on_evict is not None:
                self._on_evict(sib)
        return "evicted", freed

    def _evict_while(self, should_continue) -> int:
        """Evict oldest non-pending entries while ``should_continue(reclaimed)``.

        ``should_continue`` is passed the disk bytes reclaimed so far and returns
        whether to keep evicting. In-flight writes are skipped; the loop is bounded
        so it can't spin once every remaining entry is pending. Caller holds _lock.
        Returns the total disk bytes reclaimed.
        """
        reclaimed = 0
        attempts_left = len(self._lru)
        while self._lru and attempts_left > 0 and should_continue(reclaimed):
            outcome, freed = self._evict_one_lru_locked()
            if outcome == "stop":
                break
            if outcome == "skipped":
                attempts_left -= 1
                continue
            # An entry left the index; reset the skip budget and bank the bytes.
            reclaimed += freed
            attempts_left = len(self._lru)
        return reclaimed

    def _evict_locked(self, needed_bytes: int) -> None:
        """Evict LRU entries until total + needed <= cap*ratio. Caller holds _lock."""
        if self.max_size_bytes <= 0:
            return
        target = max(0, int(self.max_size_bytes * self.eviction_ratio) - needed_bytes)
        reclaimed = self._evict_while(lambda _: self._total_bytes > target)
        if reclaimed:
            logger.debug(
                f"HiCacheFile reclaimed {reclaimed} bytes; "
                f"now {self._total_bytes} bytes used"
            )
