"""ctypes binding for NVIDIA cuFile (GPUDirect Storage).

Reads and writes files directly into CUDA device memory through
``libcufile.so``: full GPUDirect Storage when the ``nvidia-fs`` driver is
loaded, PCI P2PDMA to bare NVMe when the platform allows it, and a POSIX
bounce-buffer path ("compat" mode) everywhere else. The API is identical in
all three modes; :meth:`CuFileDriver.mode` reports which one is active.

The library is found via ``SGLANG_CUFILE_LIB``, the pip ``nvidia`` cufile
packages, or ``CUDA_HOME``. :func:`cufile_available` never raises; every
other entry point raises :class:`CuFileError` (or ``OSError``) on failure.

Thread safety
-------------
- :class:`CuFileDriver` is a process-wide singleton; ``instance()`` is
  internally locked and safe from any thread. CUDA must be initialized
  before the first call (cuFile opens a CUDA context internally).
- :class:`CuFileHandle` reads/writes are safe from multiple threads (the
  library serializes on its own locks); ``close()`` must be externally
  synchronized with outstanding IO.
- :func:`register_buffer` / :func:`deregister_buffer` are safe for distinct
  buffers; deregistering a buffer with IO in flight is undefined.
- A :class:`CuFileBatch` is NOT safe for concurrent ``submit``/``wait`` from
  multiple threads, and two batches cannot have IOs outstanding at the same
  time: the driver's batch queue depth is a process-wide budget.
- :func:`read_async` enqueues one stream op; the returned ticket owns the
  argument cells and must be kept alive until the stream is synchronized.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import logging
import os
import threading
import time
from typing import Optional, Sequence, Union

import msgspec
import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# cufile.h enum CUfileFileHandleType.CU_FILE_HANDLE_TYPE_OPAQUE_FD
_HANDLE_TYPE_OPAQUE_FD = 1
# cufile.h enum CUfileOpcode
_OP_READ = 0
_OP_WRITE = 1
# cufile.h enum cufileBatchMode.CUFILE_BATCH: must be the first iocb field.
_BATCH_MODE = 1
# cufile.h enum CUfileStatus_t
_STATUS_WAITING = 0x1
_STATUS_PENDING = 0x2
_STATUS_INVALID = 0x4
_STATUS_CANCELED = 0x8
_STATUS_COMPLETE = 0x10
_STATUS_TIMEOUT = 0x20
_STATUS_FAILED = 0x40
# cufile.h enum CUfileFeatureFlags: the props.fflags bitmask.
_FF_DYN_ROUTING = 1 << 0
_FF_BATCH_IO = 1 << 1
_FF_STREAMS = 1 << 2
_FF_PARALLEL_IO = 1 << 3
_FF_P2P = 1 << 4

# numpy view of the CUfileIOParams_t array: filling a native chunk through
# column assignments instead of per-IO ctypes writes keeps a 131k-IO
# submission's Python cost from dominating its IO time.
_IO_PARAMS_NP = np.dtype(
    [
        ("mode", "<i4"),
        ("pad0", "<i4"),
        ("devPtr_base", "<u8"),
        ("file_offset", "<i8"),
        ("devPtr_offset", "<i8"),
        ("size", "<u8"),
        ("fh", "<u8"),
        ("opcode", "<i4"),
        ("pad1", "<i4"),
        ("cookie", "<u8"),
    ]
)
# Compat-mode only: the POSIX bounce pool (cufile.json posix_pool_slab_count)
# has 64 1-MiB and 128 4-KiB staging slabs, and a batch submit needing more
# blocks forever (the slabs free only when GetStatus runs, which the caller
# cannot do while stuck inside submit). Measured with libcufile 1.19.1:
# 65+ concurrent >4-KiB batch IOs hang; 128 concurrent 4-KiB IOs do not.
_COMPAT_MAX_LARGE_IOS_PER_ROUND = 64
# Partial-transfer resubmissions per wait() before a short count stands.
_MAX_RESUME_ROUNDS = 16

# cufile.h CUFILEOP_STATUS_ENTRIES: code -> (name, description).
_CUFILE_ERRORS = {
    0: ("CU_FILE_SUCCESS", "cufile success"),
    5001: (
        "CU_FILE_DRIVER_NOT_INITIALIZED",
        "nvidia-fs driver is not loaded; set allow_compat_mode in cufile.json",
    ),
    5002: ("CU_FILE_DRIVER_INVALID_PROPS", "invalid property"),
    5003: ("CU_FILE_DRIVER_UNSUPPORTED_LIMIT", "property range error"),
    5004: ("CU_FILE_DRIVER_VERSION_MISMATCH", "nvidia-fs driver version mismatch"),
    5005: ("CU_FILE_DRIVER_VERSION_READ_ERROR", "nvidia-fs version read error"),
    5006: ("CU_FILE_DRIVER_CLOSING", "driver shutdown in progress"),
    5007: (
        "CU_FILE_PLATFORM_NOT_SUPPORTED",
        "GPUDirect Storage unsupported on platform",
    ),
    5008: ("CU_FILE_IO_NOT_SUPPORTED", "GPUDirect Storage unsupported on this file"),
    5009: ("CU_FILE_DEVICE_NOT_SUPPORTED", "GPUDirect Storage unsupported on this GPU"),
    5010: ("CU_FILE_NVFS_DRIVER_ERROR", "nvidia-fs ioctl error"),
    5011: ("CU_FILE_CUDA_DRIVER_ERROR", "CUDA driver API error"),
    5012: ("CU_FILE_CUDA_POINTER_INVALID", "invalid device pointer"),
    5013: ("CU_FILE_CUDA_MEMORY_TYPE_INVALID", "invalid pointer memory type"),
    5014: ("CU_FILE_CUDA_POINTER_RANGE_ERROR", "pointer range exceeds allocation"),
    5015: ("CU_FILE_CUDA_CONTEXT_MISMATCH", "cuda context mismatch"),
    5016: ("CU_FILE_INVALID_MAPPING_SIZE", "access beyond maximum pinned size"),
    5017: ("CU_FILE_INVALID_MAPPING_RANGE", "access beyond mapped size"),
    5018: ("CU_FILE_INVALID_FILE_TYPE", "unsupported file type"),
    5019: ("CU_FILE_INVALID_FILE_OPEN_FLAG", "unsupported file open flags"),
    5020: ("CU_FILE_DIO_NOT_SET", "file descriptor does not have O_DIRECT"),
    5022: ("CU_FILE_INVALID_VALUE", "invalid arguments"),
    5023: ("CU_FILE_MEMORY_ALREADY_REGISTERED", "device pointer already registered"),
    5024: ("CU_FILE_MEMORY_NOT_REGISTERED", "device pointer lookup failure"),
    5025: ("CU_FILE_PERMISSION_DENIED", "driver or file access error"),
    5026: ("CU_FILE_DRIVER_ALREADY_OPEN", "driver is already open"),
    5027: ("CU_FILE_HANDLE_NOT_REGISTERED", "file descriptor is not registered"),
    5028: (
        "CU_FILE_HANDLE_ALREADY_REGISTERED",
        "file descriptor is already registered",
    ),
    5029: ("CU_FILE_DEVICE_NOT_FOUND", "GPU device not found"),
    5030: ("CU_FILE_INTERNAL_ERROR", "internal error"),
    5031: ("CU_FILE_GETNEWFD_FAILED", "failed to obtain a new file descriptor"),
    5033: ("CU_FILE_NVFS_SETUP_ERROR", "NVFS driver initialization error"),
    5034: ("CU_FILE_IO_DISABLED", "GPUDirect Storage disabled by config on this file"),
    5035: ("CU_FILE_BATCH_SUBMIT_FAILED", "failed to submit batch operation"),
    5036: ("CU_FILE_GPU_MEMORY_PINNING_FAILED", "failed to allocate pinned GPU memory"),
    5037: ("CU_FILE_BATCH_FULL", "queue full for batch operation"),
    5038: ("CU_FILE_ASYNC_NOT_SUPPORTED", "cuFile stream operation not supported"),
    5039: ("CU_FILE_INTERNAL_BATCH_SETUP_ERROR", "batch setup internal error; retry"),
    5040: ("CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR", "batch submit internal error; retry"),
    5041: (
        "CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR",
        "batch get-status internal error; retry",
    ),
    5042: ("CU_FILE_INTERNAL_BATCH_CANCEL_ERROR", "batch cancel internal error; retry"),
    5043: ("CU_FILE_NOMEM_ERROR", "cufile no-memory error; retry"),
    5044: ("CU_FILE_IO_ERROR", "cufile io error"),
    5045: ("CU_FILE_INTERNAL_BUF_REGISTER_ERROR", "buffer registration error"),
    5046: ("CU_FILE_HASH_OPR_ERROR", "hash operation error"),
    5047: ("CU_FILE_INVALID_CONTEXT_ERROR", "invalid context error"),
    5048: ("CU_FILE_NVFS_INTERNAL_DRIVER_ERROR", "nvfs internal driver error"),
    5049: ("CU_FILE_BATCH_NOCOMPAT_ERROR", "compat mode off error"),
    5050: ("CU_FILE_IO_MAX_ERROR", "GPUDirect Storage max error"),
}


class CuFileError(RuntimeError):
    """A CU_FILE_* code from libcufile, decoded."""

    def __init__(self, code: int, context: str = "") -> None:
        name, desc = _CUFILE_ERRORS.get(
            code, ("CU_FILE_UNKNOWN", "unknown cufile error")
        )
        msg = f"{name} ({code}): {desc}"
        if context:
            msg = f"{context}: {msg}"
        super().__init__(msg)
        self.code = code


def _decode_status(status: int) -> str:
    names = {
        _STATUS_WAITING: "WAITING",
        _STATUS_PENDING: "PENDING",
        _STATUS_INVALID: "INVALID",
        _STATUS_CANCELED: "CANCELED",
        _STATUS_COMPLETE: "COMPLETE",
        _STATUS_TIMEOUT: "TIMEOUT",
        _STATUS_FAILED: "FAILED",
    }
    return names.get(status, f"0x{status:x}")


class _CUfileHandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]


class _CUfileDescr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _CUfileHandleUnion),
        ("fs_ops", ctypes.c_void_p),
    ]


class _CUfileBatchParams(ctypes.Structure):
    _fields_ = [
        ("devPtr_base", ctypes.c_void_p),
        ("file_offset", ctypes.c_int64),
        ("devPtr_offset", ctypes.c_int64),
        ("size", ctypes.c_size_t),
    ]


class _CUfileIOParamsUnion(ctypes.Union):
    _fields_ = [("batch", _CUfileBatchParams)]


class _CUfileIOParams(ctypes.Structure):
    # Access goes through ``p.u.batch.*``: a flattened anonymous union would
    # expose the inner struct as ``p.batch``, and a write to ``p.devPtr_base``
    # would silently create a Python attribute instead of C memory.
    _fields_ = [
        ("mode", ctypes.c_int),
        ("u", _CUfileIOParamsUnion),
        ("fh", ctypes.c_void_p),
        ("opcode", ctypes.c_int),
        ("cookie", ctypes.c_void_p),
    ]


class _CUfileIOEvents(ctypes.Structure):
    _fields_ = [
        ("cookie", ctypes.c_void_p),
        ("status", ctypes.c_int),
        ("ret", ctypes.c_size_t),
    ]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _CUfileNvfsProps(ctypes.Structure):
    _fields_ = [
        ("major_version", ctypes.c_uint),
        ("minor_version", ctypes.c_uint),
        ("poll_thresh_size", ctypes.c_size_t),
        ("max_direct_io_size", ctypes.c_size_t),
        ("dstatusflags", ctypes.c_uint),
        ("dcontrolflags", ctypes.c_uint),
    ]


class _CUfileDrvProps(ctypes.Structure):
    _fields_ = [
        ("nvfs", _CUfileNvfsProps),
        ("fflags", ctypes.c_uint),
        ("max_device_cache_size", ctypes.c_uint),
        ("per_buffer_cache_size", ctypes.c_uint),
        ("max_device_pinned_mem_size", ctypes.c_uint),
        ("max_batch_io_size", ctypes.c_uint),
        ("max_batch_io_timeout_msecs", ctypes.c_uint),
    ]


assert ctypes.sizeof(_CUfileIOParams) == 64, "CUfileIOParams_t layout mismatch"
assert ctypes.sizeof(_CUfileIOEvents) == 24, "CUfileIOEvents_t layout mismatch"
assert ctypes.sizeof(_CUfileDescr) == 24, "CUfileDescr_t layout mismatch"


def _library_candidates() -> Sequence[str]:
    override = envs.SGLANG_CUFILE_LIB.get()
    if override:
        return [override]
    candidates = []
    import sys

    for entry in sys.path:
        if not entry.endswith("site-packages") or not os.path.isdir(entry):
            continue
        # pip nvidia-cufile-cu12/cu13 package and the bundled wheel layout
        # used by recent torch builds (nvidia/cu13).
        for pkg in ("cufile", "cu12", "cu13"):
            pattern = os.path.join(entry, "nvidia", pkg, "lib", "libcufile.so.*")
            candidates.extend(sorted(glob.glob(pattern), reverse=True))
    cuda_homes = []
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            cuda_homes.append(os.environ[var])
    cuda_homes.extend(sorted(glob.glob("/usr/local/cuda*"), reverse=True))
    for home in cuda_homes:
        candidates.extend(
            [
                os.path.join(home, "lib64", "libcufile.so.0"),
                os.path.join(home, "lib", "libcufile.so.0"),
                os.path.join(home, "targets", "x86_64-linux", "lib", "libcufile.so.0"),
            ]
        )
    found = ctypes.util.find_library("cufile")
    if found:
        candidates.append(found)
    return candidates


_CDLL: Optional[ctypes.CDLL] = None
_CDLL_TRIED = False
_CDLL_LOCK = threading.Lock()


def _cdll() -> Optional[ctypes.CDLL]:
    global _CDLL, _CDLL_TRIED
    if _CDLL_TRIED:
        return _CDLL
    with _CDLL_LOCK:
        if _CDLL_TRIED:
            return _CDLL
        for path in _library_candidates():
            try:
                _CDLL = ctypes.CDLL(path)
                break
            except OSError:
                continue
        if _CDLL is not None:
            _set_symbol_signatures(_CDLL)
        else:
            logger.info("cufile: libcufile.so not found; GPUDirect Storage unavailable")
        _CDLL_TRIED = True
    return _CDLL


def _set_symbol_signatures(lib: ctypes.CDLL) -> None:
    # ssize_t cuFileRead(CUfileHandle_t, void *, size_t, off_t, off_t)
    lib.cuFileRead.restype = ctypes.c_ssize_t
    lib.cuFileRead.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.cuFileWrite.restype = ctypes.c_ssize_t
    lib.cuFileWrite.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    # CUfileError_t is {int err; int cu_err} -> returns as c_int (err alone
    # fits; the union is returned in registers on x86_64 and ctypes reads the
    # first field).
    for fn in (
        "cuFileDriverOpen",
        "cuFileDriverClose",
        "cuFileHandleRegister",
        "cuFileBufRegister",
        "cuFileBufDeregister",
        "cuFileBatchIOSetUp",
        "cuFileBatchIOSubmit",
        "cuFileBatchIOGetStatus",
        "cuFileBatchIOCancel",
        "cuFileReadAsync",
        "cuFileWriteAsync",
    ):
        getattr(lib, fn).restype = ctypes.c_int
    lib.cuFileDriverGetProperties.restype = ctypes.c_int
    lib.cuFileHandleDeregister.restype = ctypes.c_int
    lib.cuFileHandleDeregister.argtypes = [ctypes.c_void_p]
    lib.cuFileBatchIODestroy.restype = None
    lib.cuFileBatchIODestroy.argtypes = [ctypes.c_void_p]
    lib.cuFileDriverGetProperties.argtypes = [ctypes.POINTER(_CUfileDrvProps)]
    lib.cuFileHandleRegister.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_CUfileDescr),
    ]
    lib.cuFileBufRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.cuFileBufDeregister.argtypes = [ctypes.c_void_p]
    lib.cuFileBatchIOSetUp.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cuFileBatchIOSubmit.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(_CUfileIOParams),
        ctypes.c_uint,
    ]
    lib.cuFileBatchIOGetStatus.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(_CUfileIOEvents),
        ctypes.POINTER(_Timespec),
    ]
    lib.cuFileBatchIOCancel.argtypes = [ctypes.c_void_p]
    lib.cuFileReadAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_ssize_t),
        ctypes.c_void_p,
    ]


def cufile_available() -> bool:
    """Whether libcufile.so is loadable on this system."""
    return _cdll() is not None


def _check(rc: int, context: str) -> None:
    if rc != 0:
        raise CuFileError(rc if rc > 0 else -rc, context)


def _ptr(buf: Union[int, object], offset: int = 0) -> int:
    if isinstance(buf, int):
        return buf + offset
    return buf.data_ptr() + offset


class CuFileDriver:
    """Process-wide cuFile driver session.

    ``instance()`` opens the driver once (CUDA must already be initialized)
    and closes it at interpreter exit. Opening multiple times in one process
    returns the same session; libcufile reference-counts driver opens.
    """

    _instance: Optional[CuFileDriver] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        lib = _cdll()
        if lib is None:
            raise RuntimeError("libcufile.so not found; cufile_available() is False")
        self._lib = lib
        _check(lib.cuFileDriverOpen(), "cuFileDriverOpen")
        props = _CUfileDrvProps()
        _check(
            lib.cuFileDriverGetProperties(ctypes.byref(props)),
            "cuFileDriverGetProperties",
        )
        self._props = props
        self._mode = self._detect_mode(props)
        logger.info(
            "cufile: driver open, mode=%s, max_batch_io_size=%d, "
            "max_direct_io_size=%d KiB",
            self._mode,
            props.max_batch_io_size,
            props.nvfs.max_direct_io_size >> 10,
        )

    @staticmethod
    def _detect_mode(props: _CUfileDrvProps) -> str:
        # The nvidia-fs version is nonzero only with the driver loaded, i.e.
        # the full GPUDirect Storage data path. Without nvidia-fs, cuFile
        # either DMAs over PCI P2PDMA to bare NVMe (fflags bit, needs the
        # platform's IOMMU to allow it) or bounces through pinned host memory.
        if props.nvfs.major_version != 0:
            return "gds"
        if props.fflags & _FF_P2P:
            return "p2pdma"
        return "compat"

    @classmethod
    def instance(cls) -> CuFileDriver:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                import atexit

                atexit.register(cls._close_atexit)
            return cls._instance

    @classmethod
    def _close_atexit(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance.close()
                except Exception:
                    pass
                cls._instance = None

    def mode(self) -> str:
        """The active data path: ``"gds"``, ``"p2pdma"`` or ``"compat"``."""
        return self._mode

    @property
    def max_batch_io_size(self) -> int:
        """Per-handle batch IO queue depth (cufile.json ``io_batchsize``)."""
        return int(self._props.max_batch_io_size)

    @property
    def max_direct_io_size(self) -> int:
        """Largest single IO the driver takes on the direct path, in bytes."""
        return int(self._props.nvfs.max_direct_io_size)

    @property
    def supports_batch_io(self) -> bool:
        return bool(self._props.fflags & _FF_BATCH_IO)

    @property
    def supports_stream_io(self) -> bool:
        return bool(self._props.fflags & _FF_STREAMS)

    def close(self) -> None:
        """Shut the driver down. All handles and IO must be finished first."""
        _check(self._lib.cuFileDriverClose(), "cuFileDriverClose")


class CuFileHandle:
    """One file registered with cuFile, opened with ``O_DIRECT``.

    ``read``/``write`` take a device or host buffer (torch tensor or raw
    pointer) plus a byte range. In gds/p2pdma mode the file offset, size and
    device address must be 4 KiB-aligned for the direct path; unaligned IO
    bounces through pinned host memory. In compat mode any alignment works.
    """

    def __init__(self, path: str, writable: bool = False) -> None:
        self._driver = CuFileDriver.instance()
        self._path = str(path)
        flags = os.O_RDWR if writable else os.O_RDONLY
        flags |= getattr(os, "O_DIRECT", 0)
        if writable:
            flags |= os.O_CREAT
        self._fd = os.open(self._path, flags, 0o644)
        descr = _CUfileDescr()
        descr.type = _HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = self._fd
        self._handle = ctypes.c_void_p()
        try:
            _check(
                self._driver._lib.cuFileHandleRegister(
                    ctypes.byref(self._handle), ctypes.byref(descr)
                ),
                f"cuFileHandleRegister({self._path})",
            )
        except BaseException:
            os.close(self._fd)
            self._fd = -1
            raise

    @property
    def path(self) -> str:
        return self._path

    @property
    def native_handle(self) -> int:
        return self._handle.value or 0

    def read(
        self,
        buf: Union[int, object],
        size: int,
        file_offset: int,
        buf_offset: int = 0,
    ) -> int:
        """Read ``size`` bytes at ``file_offset`` into ``buf + buf_offset``.

        Returns the byte count read; raises on error.
        """
        ret = self._driver._lib.cuFileRead(
            self._handle, ctypes.c_void_p(_ptr(buf)), size, file_offset, buf_offset
        )
        return self._io_result(ret, "cuFileRead", file_offset, size)

    def write(
        self,
        buf: Union[int, object],
        size: int,
        file_offset: int,
        buf_offset: int = 0,
    ) -> int:
        """Write ``size`` bytes from ``buf + buf_offset`` to ``file_offset``."""
        ret = self._driver._lib.cuFileWrite(
            self._handle, ctypes.c_void_p(_ptr(buf)), size, file_offset, buf_offset
        )
        return self._io_result(ret, "cuFileWrite", file_offset, size)

    @staticmethod
    def _io_result(ret: int, op: str, file_offset: int, size: int) -> int:
        if ret >= 0:
            return ret
        code = -ret
        if code in _CUFILE_ERRORS:
            raise CuFileError(code, f"{op}(file_offset={file_offset}, size={size})")
        raise OSError(
            ctypes.get_errno() or code,
            f"{op}(file_offset={file_offset}, size={size}) failed",
        )

    def close(self) -> None:
        if self._handle:
            self._driver._lib.cuFileHandleDeregister(self._handle)
            self._handle = ctypes.c_void_p()
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> CuFileHandle:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def register_buffer(buf: Union[int, object], size: Optional[int] = None) -> None:
    """Pin ``buf`` (torch tensor or raw pointer) for the cuFile direct path.

    Registration is required for 4 KiB-aligned direct IO in gds/p2pdma mode
    and extends the bounce-buffer fast path in compat mode. Optional: cuFile
    falls back to its internal pinned pool for unregistered buffers.
    """
    ptr = _ptr(buf)
    if size is None:
        size = buf.numel() * buf.element_size() if not isinstance(buf, int) else 0
    if size <= 0:
        raise ValueError("register_buffer needs a positive size")
    lib = CuFileDriver.instance()._lib
    _check(lib.cuFileBufRegister(ctypes.c_void_p(ptr), size, 0), "cuFileBufRegister")


def deregister_buffer(buf: Union[int, object]) -> None:
    """Release a buffer registered with :func:`register_buffer`."""
    lib = CuFileDriver.instance()._lib
    _check(lib.cuFileBufDeregister(ctypes.c_void_p(_ptr(buf))), "cuFileBufDeregister")


class IoDesc(msgspec.Struct, frozen=True):
    """One IO request within a :class:`CuFileBatch` submission."""

    handle: CuFileHandle
    buf_ptr: int
    size: int
    file_offset: int
    buf_offset: int = 0
    is_write: bool = False


class _BatchHandle:
    """A single native batch handle with its outstanding submission."""

    def __init__(self, lib: ctypes.CDLL, capacity: int) -> None:
        self._lib = lib
        self._capacity = capacity
        self._native = ctypes.c_void_p()
        _check(
            lib.cuFileBatchIOSetUp(ctypes.byref(self._native), capacity),
            "cuFileBatchIOSetUp",
        )
        self._params = (_CUfileIOParams * capacity)()
        self._params_view = np.frombuffer(self._params, dtype=_IO_PARAMS_NP)
        self._events = (_CUfileIOEvents * capacity)()
        self.outstanding = 0
        self.results: list[int] = []

    def submit(self, ios: Sequence[IoDesc]) -> None:
        assert len(ios) <= self._capacity
        for i, io in enumerate(ios):
            p = self._params[i]
            p.mode = _BATCH_MODE
            p.u.batch.devPtr_base = ctypes.c_void_p(io.buf_ptr)
            p.u.batch.file_offset = io.file_offset
            p.u.batch.devPtr_offset = io.buf_offset
            p.u.batch.size = io.size
            p.fh = io.handle._handle
            p.opcode = _OP_WRITE if io.is_write else _OP_READ
            p.cookie = ctypes.c_void_p(i + 1)
        try:
            _check(
                self._lib.cuFileBatchIOSubmit(self._native, len(ios), self._params, 0),
                "cuFileBatchIOSubmit",
            )
        except CuFileError:
            # A failed submit can leave the queue in an error state that
            # rejects every later submit; cancel and drain to reuse the handle.
            self.cancel()
            raise
        self.outstanding = len(ios)
        self.results = []

    def submit_vectors(
        self,
        handle: CuFileHandle,
        buf_ptr: int,
        file_offsets: np.ndarray,
        sizes: np.ndarray,
        buf_offsets: np.ndarray,
        is_write: bool = False,
    ) -> None:
        """Submit one file's IOs from numpy arrays; same contract as submit."""
        n = len(file_offsets)
        assert n <= self._capacity
        if n == 0:
            self.outstanding = 0
            self.results = []
            return
        view = self._params_view[:n]
        view["mode"] = _BATCH_MODE
        view["devPtr_base"] = buf_ptr
        view["file_offset"] = file_offsets
        view["devPtr_offset"] = buf_offsets
        view["size"] = sizes
        view["fh"] = handle.native_handle
        view["opcode"] = _OP_WRITE if is_write else _OP_READ
        view["cookie"] = np.arange(1, n + 1, dtype=np.uint64)
        try:
            _check(
                self._lib.cuFileBatchIOSubmit(self._native, n, self._params, 0),
                "cuFileBatchIOSubmit",
            )
        except CuFileError:
            self.cancel()
            raise
        self.outstanding = n
        self.results = []

    def wait(self, timeout_s: float) -> None:
        """Collect per-IO byte counts into ``results``; raise on IO failure."""
        deadline = time.monotonic() + timeout_s
        # cookie is io index + 1 at submit; entries may complete out of order.
        self.results = [0] * self.outstanding
        collected = 0
        while collected < self.outstanding:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.results = []
                raise TimeoutError(
                    f"cufile batch: {self.outstanding - collected} IO(s) still "
                    f"pending after {timeout_s}s"
                )
            nr = ctypes.c_uint(self._capacity)
            rc = self._lib.cuFileBatchIOGetStatus(
                self._native,
                self.outstanding - collected,
                ctypes.byref(nr),
                self._events,
                ctypes.byref(_Timespec(int(remaining), 0)),
            )
            if rc != 0:
                self.results = []
                raise CuFileError(rc, "cuFileBatchIOGetStatus")
            for i in range(nr.value):
                ev = self._events[i]
                if ev.status != _STATUS_COMPLETE:
                    self.cancel()
                    self.results = []
                    # A failed IO reports a negative CUfileOpError in ret.
                    ret = int(ev.ret)
                    if ret >= 1 << 63:
                        ret -= 1 << 64
                    raise CuFileError(
                        -ret if ret < 0 else 5044,
                        f"batch IO #{ev.cookie} status {_decode_status(ev.status)}",
                    )
                self.results[int(ev.cookie) - 1] = int(ev.ret)
            collected += nr.value
        self.outstanding = 0

    def cancel(self) -> None:
        try:
            self._lib.cuFileBatchIOCancel(self._native)
        except Exception:
            pass
        # Drain whatever is left so the handle is reusable.
        try:
            while True:
                nr = ctypes.c_uint(self._capacity)
                rc = self._lib.cuFileBatchIOGetStatus(
                    self._native,
                    0,
                    ctypes.byref(nr),
                    self._events,
                    None,
                )
                if rc != 0 or nr.value == 0:
                    break
        except Exception:
            pass
        self.outstanding = 0

    def close(self) -> None:
        if self._native:
            if self.outstanding:
                self.cancel()
            self._lib.cuFileBatchIODestroy(self._native)
            self._native = ctypes.c_void_p()


class _DescChunk:
    """Pending native-chunk-sized batch of IoDescs."""

    __slots__ = ("ios",)

    def __init__(self, ios: list) -> None:
        self.ios = ios


class _VecChunk:
    """Pending native-chunk-sized batch of vectorized IOs."""

    __slots__ = (
        "handle",
        "buf_ptr",
        "file_offsets",
        "sizes",
        "buf_offsets",
        "is_write",
    )

    def __init__(
        self,
        handle: CuFileHandle,
        buf_ptr: int,
        file_offsets: np.ndarray,
        sizes: np.ndarray,
        buf_offsets: np.ndarray,
        is_write: bool,
    ) -> None:
        self.handle = handle
        self.buf_ptr = buf_ptr
        self.file_offsets = file_offsets
        self.sizes = sizes
        self.buf_offsets = buf_offsets
        self.is_write = is_write


class CuFileBatch:
    """Batched cuFile IO with up to ``max_ios`` requests queued per round.

    The driver's batch queue is only ``max_batch_io_size`` IOs deep for the
    whole process -- a second native handle with IOs outstanding blocks the
    submitting thread until the first drains -- so the batch owns ONE native
    handle and executes submissions larger than the queue depth in sequential
    rounds inside ``wait``. In compat mode, rounds additionally stay within
    the POSIX bounce pool (see ``_COMPAT_MAX_LARGE_IOS_PER_ROUND``), whose
    exhaustion blocks ``submit`` irrecoverably. ``submit`` starts the first
    round immediately, so queued IO is already in flight before ``wait`` is
    called. Not thread-safe, and not concurrently usable with another batch:
    the queue depth is a process-wide budget.
    """

    def __init__(self, max_ios: int) -> None:
        if max_ios <= 0:
            raise ValueError("max_ios must be positive")
        self._driver = CuFileDriver.instance()
        if not self._driver.supports_batch_io:
            raise RuntimeError("cuFile batch IO unsupported in this mode")
        self._chunk = min(max_ios, self._driver.max_batch_io_size)
        self._handle = _BatchHandle(self._driver._lib, self._chunk)
        self._pending: list = []  # _DescChunk / _VecChunk records, in order
        self._results: list[int] = []
        # Every queued IO in order, so wait() can resume partial transfers.
        self._log: list[tuple] = []
        self._total_capacity = max_ios
        self._queued = 0

    @property
    def max_ios(self) -> int:
        return self._total_capacity

    def submit(self, ios: Sequence[IoDesc]) -> None:
        """Queue IOs; must fit the remaining capacity. Raises on failure.

        The first round goes to the native queue at once; the rest wait
        inside ``wait``.
        """
        if len(ios) + self._queued > self._total_capacity:
            raise ValueError(
                f"cufile batch full: {len(ios)} submitted with {self._queued} "
                f"queued of {self._total_capacity} capacity; wait() first"
            )
        for base in range(0, len(ios), self._chunk):
            self._pending.append(_DescChunk(list(ios[base : base + self._chunk])))
        for io in ios:
            self._log.append(
                (
                    io.handle,
                    io.buf_ptr,
                    io.is_write,
                    np.array([io.file_offset], dtype=np.int64),
                    np.array([io.size], dtype=np.int64),
                    np.array([io.buf_offset], dtype=np.int64),
                )
            )
        self._queued += len(ios)
        if self._handle.outstanding == 0:
            self._pump_round()

    def submit_spans(
        self,
        handle: CuFileHandle,
        *,
        buf_ptr: int,
        file_offsets: np.ndarray,
        sizes: np.ndarray,
        buf_offsets: np.ndarray,
        is_write: bool = False,
    ) -> None:
        """Queue one file's IOs from numpy arrays -- the vectorized hot path
        for span-shaped workloads (tens of thousands of IOs per submission,
        where per-IO ``IoDesc`` construction dominates). Same capacity
        contract and ordering as ``submit``.
        """
        n = len(file_offsets)
        if n + self._queued > self._total_capacity:
            raise ValueError(
                f"cufile batch full: {n} submitted with {self._queued} "
                f"queued of {self._total_capacity} capacity; wait() first"
            )
        for base in range(0, n, self._chunk):
            take = min(self._chunk, n - base)
            self._pending.append(
                _VecChunk(
                    handle,
                    buf_ptr,
                    file_offsets[base : base + take],
                    sizes[base : base + take],
                    buf_offsets[base : base + take],
                    is_write,
                )
            )
        self._log.append(
            (
                handle,
                buf_ptr,
                is_write,
                np.asarray(file_offsets, dtype=np.int64),
                np.asarray(sizes, dtype=np.int64),
                np.asarray(buf_offsets, dtype=np.int64),
            )
        )
        self._queued += n
        if self._handle.outstanding == 0:
            self._pump_round()

    def wait(self, timeout_s: float) -> list[int]:
        """Drive all rounds to completion; returns per-IO byte counts in order.

        Like ``pread``, cuFile may complete an IO partially (seen with
        nvidia-fs loaded and a file on the compat fallback); the remainder is
        resubmitted until the IO completes or stops progressing, so a count
        below the requested size means end of file.

        Raises :class:`CuFileError` naming the failed IO, or ``TimeoutError``.
        After either, the batch is reusable (the error state is cleared).
        """
        deadline = time.monotonic() + timeout_s
        log, self._log = self._log, []
        results = np.asarray(self._drain(deadline, timeout_s), dtype=np.int64)
        # positions[i] is the result index of row i of the current log, and
        # round_done[i] the bytes that row moved in the latest round.
        positions = np.arange(results.size)
        round_done = results.copy()
        for _ in range(_MAX_RESUME_ROUNDS):
            resumed = self._queue_remainders(log, round_done)
            if resumed is None:
                break
            log, mask = resumed
            positions = positions[mask]
            round_done = np.asarray(self._drain(deadline, timeout_s), dtype=np.int64)
            results[positions] += round_done
        return results.tolist()

    def _queue_remainders(self, log, done):
        """Queue the unfinished tail of every IO in ``log`` that progressed.

        ``done`` holds the bytes each log row moved in the latest round. Returns the log of
        the queued remainders and the mask of resumed rows, or ``None``.
        """
        if not log:
            return None
        sizes = np.concatenate([entry[4] for entry in log])
        resumable = (done > 0) & (done < sizes)
        if not resumable.any():
            return None
        resumed_log = []
        base = 0
        for handle, buf_ptr, is_write, offs, szs, bufs in log:
            n = szs.size
            got = done[base : base + n]
            mask = resumable[base : base + n]
            base += n
            if not mask.any():
                continue
            entry = (
                handle,
                buf_ptr,
                is_write,
                offs[mask] + got[mask],
                szs[mask] - got[mask],
                bufs[mask] + got[mask],
            )
            resumed_log.append(entry)
            for c in range(0, entry[4].size, self._chunk):
                self._pending.append(
                    _VecChunk(
                        handle,
                        buf_ptr,
                        entry[3][c : c + self._chunk],
                        entry[4][c : c + self._chunk],
                        entry[5][c : c + self._chunk],
                        is_write,
                    )
                )
        self._queued = int(resumable.sum())
        self._pump_round()
        return resumed_log, resumable

    def _drain(self, deadline: float, timeout_s: float) -> list[int]:
        try:
            while self._pending or self._handle.outstanding:
                if self._handle.outstanding:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"cufile batch: IOs still pending after {timeout_s}s"
                        )
                    self._handle.wait(remaining)
                    self._results.extend(self._handle.results)
                    self._handle.results = []
                else:
                    self._pump_round()
        except (CuFileError, TimeoutError):
            self._handle.cancel()
            self._pending.clear()
            self._queued = 0
            self._results = []
            raise
        results = self._results
        self._results = []
        self._queued = 0
        return results

    def _pump_round(self) -> None:
        if not self._pending:
            return
        chunk = self._pending.pop(0)
        # Unaligned IO and files on the compat fallback bounce through the
        # POSIX pool in every driver mode, not only when compat is global.
        compat = True
        if isinstance(chunk, _DescChunk):
            ios = chunk.ios
            if compat and any(io.size > 4096 for io in ios):
                head = ios[:_COMPAT_MAX_LARGE_IOS_PER_ROUND]
                if len(ios) > len(head):
                    self._pending.insert(0, _DescChunk(ios[len(head) :]))
                ios = head
            self._handle.submit(ios)
            return
        take = len(chunk.file_offsets)
        if compat and bool((chunk.sizes > 4096).any()):
            take = min(take, _COMPAT_MAX_LARGE_IOS_PER_ROUND)
        if take < len(chunk.file_offsets):
            self._pending.insert(
                0,
                _VecChunk(
                    chunk.handle,
                    chunk.buf_ptr,
                    chunk.file_offsets[take:],
                    chunk.sizes[take:],
                    chunk.buf_offsets[take:],
                    chunk.is_write,
                ),
            )
        self._handle.submit_vectors(
            chunk.handle,
            chunk.buf_ptr,
            chunk.file_offsets[:take],
            chunk.sizes[:take],
            chunk.buf_offsets[:take],
            chunk.is_write,
        )

    def close(self) -> None:
        self._handle.close()
        self._pending.clear()
        self._log = []
        self._queued = 0
        self._results = []

    def __enter__(self) -> CuFileBatch:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CuFileAsyncIo:
    """Ticket for one :func:`read_async` / :func:`write_async` stream op.

    cuFile dereferences the size/offset cells when the stream reaches the
    operation, so the ticket (and the buffer it names) must be kept alive
    until the stream is synchronized. ``bytes_read`` is valid afterwards.
    """

    def __init__(
        self,
        lib: ctypes.CDLL,
        handle: CuFileHandle,
        buf: Union[int, object],
        size: int,
        file_offset: int,
        buf_offset: int,
        is_write: bool,
        stream: int,
    ) -> None:
        self._lib = lib
        self._handle = handle
        # The cells below are read at execution time, not submission time.
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_int64(file_offset)
        self._buf_offset = ctypes.c_int64(buf_offset)
        self._bytes = ctypes.c_ssize_t(-1)
        fn = lib.cuFileWriteAsync if is_write else lib.cuFileReadAsync
        rc = fn(
            handle._handle,
            ctypes.c_void_p(_ptr(buf)),
            ctypes.byref(self._size),
            ctypes.byref(self._file_offset),
            ctypes.byref(self._buf_offset),
            ctypes.byref(self._bytes),
            ctypes.c_void_p(stream),
        )
        _check(rc, "cuFileReadAsync" if not is_write else "cuFileWriteAsync")

    @property
    def bytes_read(self) -> int:
        """Bytes transferred; read only after synchronizing the stream."""
        return int(self._bytes.value)


def read_async(
    handle: CuFileHandle,
    buf: Union[int, object],
    size: int,
    file_offset: int,
    buf_offset: int = 0,
    stream: Optional[object] = None,
) -> CuFileAsyncIo:
    """Enqueue one read on ``stream`` (a torch stream or raw CUstream).

    Works in compat mode too (the library stages through its pinned pool).
    Keep the returned ticket alive until the stream is synchronized.
    """
    if stream is None:
        import torch

        stream = torch.cuda.current_stream()
    if isinstance(stream, int):
        stream_ptr = stream
    else:
        stream_ptr = stream.cuda_stream
    return CuFileAsyncIo(
        CuFileDriver.instance()._lib,
        handle,
        buf,
        size,
        file_offset,
        buf_offset,
        False,
        stream_ptr,
    )
