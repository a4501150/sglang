"""On-disk page format shared by the HiCache file backend and its IO engines.

One file per page-group per pool: ``<key>[.<pool>]<config_suffix>.bin``. A
file is ``HEADER_BYTES`` (4 KiB) followed by the page payload padded up to
``IO_ALIGNMENT`` (4 KiB), so every file offset and size is a multiple of
4 KiB and positional (O_DIRECT / cuFile) segment IO can address the payload
as one unit.

Header layout (little endian, 4096 B):

    magic                8s   b"SGLHIC01"
    version              u32  1
    header_size          u32  4096
    format_fingerprint  16s   hex digest of the pool's storage format spec
    payload_len          u64  exact page bytes
    payload_padded_len  u64   payload_len rounded up to IO_ALIGNMENT
    crc32                u32  CRC-32 (zlib) of the payload bytes
    reserved                  zeros

The checksum is zlib's CRC-32 rather than CRC-32C: it is the only C-speed
in-process CRC available here, and the polynomial is internal to this
format (nothing interoperates with Castagnoli hardware paths).
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Optional

import msgspec
import torch

HEADER_BYTES = 4096
IO_ALIGNMENT = 4096
MAGIC = b"SGLHIC01"
VERSION = 1

# magic, version, header_size, fingerprint, payload_len, padded_len, crc32
_HEADER_STRUCT = struct.Struct("<8sII16sQQI")


class PageHeader(msgspec.Struct, frozen=True):
    fingerprint: str
    payload_len: int
    payload_padded_len: int
    crc32: int


def pad_to_io_alignment(n: int) -> int:
    return (n + IO_ALIGNMENT - 1) // IO_ALIGNMENT * IO_ALIGNMENT


def page_file_size(payload_len: int) -> int:
    """Total on-disk size of a page file: header plus padded payload."""
    return HEADER_BYTES + pad_to_io_alignment(payload_len)


def format_fingerprint(format_spec: Optional[dict]) -> str:
    """Stable 16-hex digest of a pool's page format specification.

    The spec describes everything that determines a page's byte layout
    (pool, layout, page size, layers, dtypes, component shapes, side
    states), so a geometry change produces a different fingerprint and
    pages written by the old build miss instead of loading corrupt.
    """
    payload = json.dumps(
        {"schema": VERSION, "format": format_spec or {}},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def encode_page_header(fingerprint: str, payload_len: int, crc32: int) -> bytes:
    padded_len = pad_to_io_alignment(payload_len)
    body = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        fingerprint.encode("ascii"),
        payload_len,
        padded_len,
        crc32,
    )
    return body + bytes(HEADER_BYTES - _HEADER_STRUCT.size)


def decode_page_header(buf) -> PageHeader:
    """Parse and validate a 4096-byte header buffer.

    Raises ``ValueError`` on a magic, version, or size mismatch — callers
    treat that as a miss and drop the file.
    """
    if len(buf) < _HEADER_STRUCT.size:
        raise ValueError("page file header truncated")
    magic, version, header_size, fingerprint, payload_len, padded_len, crc32 = (
        _HEADER_STRUCT.unpack_from(buf)
    )
    if magic != MAGIC:
        raise ValueError("page file header magic mismatch")
    if version != VERSION:
        raise ValueError(f"page file header version {version} unsupported")
    if header_size != HEADER_BYTES:
        raise ValueError(f"page file header size {header_size} unsupported")
    return PageHeader(
        fingerprint=fingerprint.decode("ascii"),
        payload_len=payload_len,
        payload_padded_len=padded_len,
        crc32=crc32,
    )


def hicache_tensor_digest(tensor: torch.Tensor) -> str:
    """sha256 over a tensor's raw bytes, for HiCache debug digests."""
    flat_bytes = tensor.detach().contiguous().view(torch.uint8).reshape(-1).cpu()
    return hashlib.sha256(flat_bytes.numpy()).hexdigest()
