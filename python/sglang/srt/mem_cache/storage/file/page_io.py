"""Positional page IO engines for the HiCache file backend.

An engine owns the raw byte movement of one page file (header + padded
payload) and nothing else: naming, fingerprints, the evictor, and crash
safety (tmp + fsync + rename) stay in ``HiCacheFile``. Engines are used
from the storage threads only (prefetch IO aux and backup), never from
the scheduler thread.

``posix_buffered`` is the plain preadv/pwritev path through the OS page
cache. ``posix_o_direct`` opens files with O_DIRECT and is strict: it
probes the filesystem at construction and fails loudly rather than
falling back. Payloads whose destination segments are 4 KiB-aligned and
whose length is a 4 KiB multiple (KV payload+scale pages, indexer pages)
are read and written in place; everything else (e.g. whole mamba pages
with side states) goes through a per-call aligned bounce buffer.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import List, Optional, Sequence, Tuple

from sglang.srt.mem_cache.hicache_format import (
    HEADER_BYTES,
    IO_ALIGNMENT,
)

logger = logging.getLogger(__name__)

# Bounce transfers copy the payload through scratch of this size; 1 MiB
# keeps per-call allocation cheap while amortizing syscalls.
_BOUNCE_CHUNK_BYTES = 1024 * 1024

# (host buffer address, byte length); sum of lengths is the payload length.
Segment = Tuple[int, int]


class AlignedBuffer:
    """A IO_ALIGNMENT-aligned scratch buffer for O_DIRECT."""

    def __init__(self, size: int):
        raw = (ctypes.c_char * (size + IO_ALIGNMENT))()
        offset = (-ctypes.addressof(raw)) % IO_ALIGNMENT
        self.view = memoryview(raw).cast("B")[offset : offset + size]


def segment_view(ptr: int, size: int) -> memoryview:
    """Writable ctypes-backed view straight onto a host buffer."""
    return memoryview((ctypes.c_char * size).from_address(ptr)).cast("B")


def segments_io_aligned(segments: Sequence[Segment]) -> bool:
    return all(
        ptr % IO_ALIGNMENT == 0 and size % IO_ALIGNMENT == 0
        for ptr, size in segments
    )


class PosixPageIo:
    def __init__(self, file_path: str, *, o_direct: bool):
        self.o_direct = o_direct
        self._extra_flags = os.O_DIRECT if o_direct else 0
        if o_direct:
            self._probe_direct_io(file_path)

    @property
    def name(self) -> str:
        return "posix_o_direct" if self.o_direct else "posix_buffered"

    def _probe_direct_io(self, file_path: str) -> None:
        """Aligned O_DIRECT round trip; raises rather than falling back."""
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError(
                "HiCacheFile io_engine='posix_o_direct' requires os.O_DIRECT, "
                "which this platform does not provide."
            )
        probe_path = os.path.join(
            file_path, f".odirect_probe.{os.getpid()}.{threading.get_ident()}"
        )
        fd = None
        try:
            with open(probe_path, "wb") as probe_file:
                probe_file.truncate(IO_ALIGNMENT)
            buf = AlignedBuffer(IO_ALIGNMENT)
            fd = os.open(probe_path, os.O_RDONLY | os.O_DIRECT)
            if os.preadv(fd, [buf.view], 0) != IO_ALIGNMENT:
                raise IOError("short aligned O_DIRECT probe read")
        except OSError as e:
            raise RuntimeError(
                f"Filesystem backing {file_path!r} rejected an aligned "
                f"O_DIRECT probe ({e}); io_engine='posix_o_direct' is strict "
                "and will not fall back to buffered IO."
            ) from e
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(probe_path)
            except OSError:
                pass

    @staticmethod
    def _advance_iovecs(views: List[memoryview], n: int) -> None:
        while n:
            if views and len(views[0]) <= n:
                n -= len(views[0])
                views.pop(0)
            else:
                views[0] = views[0][n:]
                n = 0

    def _iovec_transfer(
        self, fd: int, offset: int, views: List[memoryview], *, write: bool
    ) -> None:
        """Complete a positional vector IO of sum(len(v)) bytes at *offset*.

        O_DIRECT may return short; the transfer resumes through the iovecs.
        A stall (zero bytes) or an unaligned short result under O_DIRECT is
        rejected.
        """
        total = sum(len(view) for view in views)
        done = 0
        direction = "write" if write else "read"
        while done < total:
            n = (os.pwritev if write else os.preadv)(fd, views, offset + done)
            if n is None or n <= 0:
                raise IOError(
                    f"page file {direction} stalled at offset {offset + done} "
                    f"({done}/{total} bytes complete)"
                )
            done += n
            self._advance_iovecs(views, n)

    def read_header(self, path: str) -> Optional[bytes]:
        """Raw header bytes (up to 4096), or None when unreadable/missing.

        Always buffered: the header is small metadata, and the janitor must
        read foreign/short files that O_DIRECT cannot address.
        """
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            size = os.fstat(fd).st_size
            if size == 0:
                return b""
            n = min(size, HEADER_BYTES)
            buf = bytearray(n)
            done = 0
            while done < n:
                got = os.preadv(fd, [memoryview(buf)[done:]], done)
                if got is None or got <= 0:
                    return None  # truncated mid-read
                done += got
            return bytes(buf)
        except OSError:
            return None
        finally:
            os.close(fd)

    def read_payload(
        self, path: str, payload_len: int, padded_len: int, segments: List[Segment]
    ) -> None:
        """Read the padded payload at HEADER_BYTES into *segments*.

        Segment lengths must sum to ``payload_len``. Aligned segments of an
        aligned payload are filled in place; otherwise the padded region is
        read through an aligned bounce buffer.
        """
        try:
            fd = os.open(path, os.O_RDONLY | self._extra_flags)
        except FileNotFoundError as e:
            raise IOError(f"page file vanished: {path}") from e
        try:
            if not self.o_direct:
                self._iovec_transfer(
                    fd, HEADER_BYTES, self._views(segments), write=False
                )
                return
            if payload_len == padded_len and segments_io_aligned(segments):
                self._iovec_transfer(
                    fd, HEADER_BYTES, self._views(segments), write=False
                )
                return
            self._bounce_transfer(
                fd,
                HEADER_BYTES,
                segments,
                payload_len,
                padded_len,
                write=False,
            )
        finally:
            os.close(fd)

    def write_page_file(
        self, tmp_path: str, header_bytes: bytes, segments: List[Segment]
    ) -> None:
        """Write header + zero-padded payload to *tmp_path*, then fsync."""
        payload_len = sum(size for _, size in segments)
        padded_len = (payload_len + IO_ALIGNMENT - 1) // IO_ALIGNMENT * IO_ALIGNMENT
        fd = os.open(
            tmp_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | self._extra_flags,
            0o644,
        )
        try:
            os.ftruncate(fd, HEADER_BYTES + padded_len)
            if self.o_direct:
                header = AlignedBuffer(HEADER_BYTES)
                header.view[:] = header_bytes
                if payload_len == padded_len and segments_io_aligned(segments):
                    self._iovec_transfer(
                        fd, 0, [header.view, *self._views(segments)], write=True
                    )
                else:
                    self._iovec_transfer(fd, 0, [header.view], write=True)
                    self._bounce_transfer(
                        fd,
                        HEADER_BYTES,
                        segments,
                        payload_len,
                        padded_len,
                        write=True,
                    )
            else:
                self._iovec_transfer(
                    fd, 0, [memoryview(bytes(header_bytes))], write=True
                )
                tail = memoryview(bytes(padded_len - payload_len))
                self._iovec_transfer(
                    fd, HEADER_BYTES, [*self._views(segments), tail], write=True
                )
            os.fsync(fd)
        finally:
            os.close(fd)

    def _bounce_transfer(
        self,
        fd: int,
        file_offset: int,
        segments: List[Segment],
        payload_len: int,
        padded_len: int,
        *,
        write: bool,
    ) -> None:
        """Move the padded payload in aligned chunks through scratch memory.

        Read: chunk bytes land in scratch, the payload prefix is scattered
        into the segments. Write: segments are gathered into scratch and the
        tail zero-padded.
        """
        bounce = AlignedBuffer(min(_BOUNCE_CHUNK_BYTES, padded_len))
        views = self._views(segments)
        done = 0
        while done < padded_len:
            chunk = min(len(bounce.view), padded_len - done)
            image = bounce.view[:chunk]
            keep = min(chunk, payload_len - done)
            if write:
                if keep > 0:
                    self._gather(image[:keep], views)
                if keep < chunk:
                    image[keep:] = bytes(chunk - keep)
                self._iovec_transfer(fd, file_offset + done, [image], write=True)
            else:
                self._iovec_transfer(fd, file_offset + done, [image], write=False)
                if keep > 0:
                    self._scatter(views, image[:keep])
            done += chunk

    @staticmethod
    def _views(segments: List[Segment]) -> List[memoryview]:
        return [segment_view(ptr, size) for ptr, size in segments]

    @staticmethod
    def _scatter(views: List[memoryview], src: memoryview) -> None:
        copied = 0
        while copied < len(src):
            view = views[0]
            n = min(len(view), len(src) - copied)
            view[:] = src[copied : copied + n]
            copied += n
            if len(view) == n:
                views.pop(0)

    @staticmethod
    def _gather(dst: memoryview, views: List[memoryview]) -> None:
        copied = 0
        while copied < len(dst):
            view = views[0]
            n = min(len(view), len(dst) - copied)
            dst[copied : copied + n] = view[:n]
            copied += n
            if len(view) == n:
                views.pop(0)


PAGE_IO_ENGINE_NAMES = ("posix_buffered", "posix_o_direct", "cufile")


def make_page_io(file_path: str, engine: str):
    if engine == "posix_buffered":
        return PosixPageIo(file_path, o_direct=False)
    if engine == "posix_o_direct":
        return PosixPageIo(file_path, o_direct=True)
    if engine == "cufile":
        raise NotImplementedError(
            "HiCacheFile io_engine='cufile' requires the cuFile binding "
            "(python/sglang/srt/utils/cufile.py), which is not available yet."
        )
    raise ValueError(
        f"HiCacheFile io_engine must be one of {PAGE_IO_ENGINE_NAMES}, got {engine!r}"
    )
