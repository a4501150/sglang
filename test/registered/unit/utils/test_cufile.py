"""GPU tests for the cuFile (GPUDirect Storage) ctypes binding.

Writes a pattern file under ``$HOME/.cache/sglang-test``, reads it back
through a registered handle and through batch IO at aligned and unaligned
offsets, and exercises the async stream path. All of it runs identically in
gds, p2pdma and compat modes (the active one is asserted and logged); compat
bounces through pinned host memory, the others DMA directly.

The library itself is optional: the whole suite skips when libcufile.so is
not loadable (``cufile_available()`` is False).
"""

from __future__ import annotations

import os
import unittest

import torch

from sglang.srt.utils import cufile
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _cache_dir() -> str:
    # Per the repo's test conventions: never /tmp (tmpfs), always the cache.
    root = os.path.expanduser(
        os.environ.get("SGLANG_CACHE_DIR", os.path.join("~", ".cache", "sglang"))
    )
    path = os.path.join(root, "sglang-test", "cufile")
    os.makedirs(path, exist_ok=True)
    return path


class TestCuFileBinding(CustomTestCase):
    """Handle reads/writes, batch IO, buffer registration and mode query."""

    @classmethod
    def setUpClass(cls):
        if not cufile.cufile_available():
            raise unittest.SkipTest("libcufile.so not available")
        # cuFile needs a CUDA context before the driver opens.
        cls._warm = torch.zeros(8, device="cuda")
        cls.driver = cufile.CuFileDriver.instance()
        cls.mode = cls.driver.mode()
        assert cls.mode in ("gds", "p2pdma", "compat"), cls.mode
        cls.path = os.path.join(_cache_dir(), "pattern_8m.bin")
        if not os.path.exists(cls.path) or os.path.getsize(cls.path) != 8 << 20:
            with open(cls.path, "wb") as f:
                f.write(bytes(range(256)) * ((8 << 20) // 256))
        cls.data = torch.from_file(
            cls.path, shared=False, size=8 << 20, dtype=torch.uint8
        ).clone()

    def _new_device_buffer(self, nbytes: int) -> torch.Tensor:
        buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        cufile.register_buffer(buf, nbytes)
        self.addCleanup(cufile.deregister_buffer, buf)
        return buf

    def test_mode_reported(self):
        self.assertIn(self.driver.mode(), ("gds", "p2pdma", "compat"))

    def test_handle_read_aligned(self):
        with cufile.CuFileHandle(self.path) as handle:
            buf = self._new_device_buffer(64 << 10)
            n = handle.read(buf, 64 << 10, 4096)
            self.assertEqual(n, 64 << 10)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(buf.cpu(), self.data[4096 : 4096 + (64 << 10)]))

    def test_handle_read_unaligned(self):
        # Rows can straddle pages; compat mode bounces unaligned ranges.
        with cufile.CuFileHandle(self.path) as handle:
            buf = self._new_device_buffer(333)
            n = handle.read(buf, 333, 100_000 + 7)
            self.assertEqual(n, 333)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(buf.cpu(), self.data[100_007 : 100_007 + 333]))

    def test_handle_read_with_buf_offset(self):
        with cufile.CuFileHandle(self.path) as handle:
            buf = self._new_device_buffer(8192)
            n = handle.read(buf, 4096, 8192, buf_offset=4096)
            self.assertEqual(n, 4096)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(buf[4096:].cpu(), self.data[8192 : 8192 + 4096])
            )

    def test_handle_write_readback(self):
        path = os.path.join(_cache_dir(), "write_test.bin")
        if os.path.exists(path):
            os.unlink(path)
        expect = (torch.arange(16 << 10) % 251).to(torch.uint8)
        with cufile.CuFileHandle(path, writable=True) as handle:
            src = expect.to(device="cuda")
            cufile.register_buffer(src, src.numel())
            self.addCleanup(cufile.deregister_buffer, src)
            n = handle.write(src, 16 << 10, 0)
            self.assertEqual(n, 16 << 10)
        # The write path synchronizes inside cuFileWrite; a plain read sees it.
        back = torch.from_file(path, shared=False, size=16 << 10, dtype=torch.uint8)
        self.assertTrue(torch.equal(back, expect))

    def test_batch_read_aligned_and_unaligned(self):
        if not self.driver.supports_batch_io:
            raise unittest.SkipTest("batch IO unsupported in this mode")
        with (
            cufile.CuFileHandle(self.path) as handle,
            cufile.CuFileBatch(max_ios=64) as batch,
        ):
            buf = self._new_device_buffer(64 * 8192)
            ios = [
                cufile.IoDesc(
                    handle=handle,
                    buf_ptr=buf.data_ptr(),
                    size=8192,
                    file_offset=4096 * i,
                    buf_offset=8192 * i,
                )
                for i in range(64)
            ]
            batch.submit(ios)
            rets = batch.wait(timeout_s=30)
            self.assertEqual(rets, [8192] * 64)
            torch.cuda.synchronize()
            for i in range(64):
                chunk = buf[8192 * i : 8192 * i + 8192].cpu()
                self.assertTrue(
                    torch.equal(chunk, self.data[4096 * i : 4096 * i + 8192])
                )

    def test_batch_split_across_driver_chunks(self):
        # More IOs per submit than one native handle takes: the batch splits.
        if not self.driver.supports_batch_io:
            raise unittest.SkipTest("batch IO unsupported in this mode")
        with (
            cufile.CuFileHandle(self.path) as handle,
            cufile.CuFileBatch(max_ios=self.driver.max_batch_io_size * 3) as batch,
        ):
            count = self.driver.max_batch_io_size * 3
            buf = self._new_device_buffer(4096 * count)
            ios = [
                cufile.IoDesc(
                    handle=handle,
                    buf_ptr=buf.data_ptr(),
                    size=4096,
                    file_offset=4096 * (i % 2000),
                    buf_offset=4096 * i,
                )
                for i in range(count)
            ]
            batch.submit(ios)
            rets = batch.wait(timeout_s=60)
            self.assertEqual(rets, [4096] * count)

    def test_batch_capacity_enforced(self):
        if not self.driver.supports_batch_io:
            raise unittest.SkipTest("batch IO unsupported in this mode")
        with (
            cufile.CuFileHandle(self.path) as handle,
            cufile.CuFileBatch(max_ios=4) as batch,
        ):
            buf = self._new_device_buffer(4 * 4096)
            ios = [
                cufile.IoDesc(
                    handle=handle, buf_ptr=buf.data_ptr(), size=4096, file_offset=0
                )
                for _ in range(5)
            ]
            with self.assertRaises(ValueError):
                batch.submit(ios)

    def test_batch_reusable_after_wait(self):
        if not self.driver.supports_batch_io:
            raise unittest.SkipTest("batch IO unsupported in this mode")
        with (
            cufile.CuFileHandle(self.path) as handle,
            cufile.CuFileBatch(max_ios=8) as batch,
        ):
            buf = self._new_device_buffer(8 * 4096)
            for _ in range(3):
                ios = [
                    cufile.IoDesc(
                        handle=handle,
                        buf_ptr=buf.data_ptr(),
                        size=4096,
                        file_offset=4096 * i,
                        buf_offset=4096 * i,
                    )
                    for i in range(8)
                ]
                batch.submit(ios)
                self.assertEqual(batch.wait(timeout_s=30), [4096] * 8)

    def test_read_async_on_stream(self):
        if not self.driver.supports_stream_io:
            raise unittest.SkipTest("stream IO unsupported in this mode")
        stream = torch.cuda.Stream()
        with cufile.CuFileHandle(self.path) as handle:
            buf = self._new_device_buffer(16384)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                ticket = cufile.read_async(
                    handle, buf, 16384, file_offset=8192, stream=stream
                )
            stream.synchronize()
            self.assertEqual(ticket.bytes_read, 16384)
            self.assertTrue(torch.equal(buf.cpu(), self.data[8192 : 8192 + 16384]))

    def test_read_past_eof_is_short(self):
        # pread semantics: no error, the transfer just comes up empty.
        with cufile.CuFileHandle(self.path) as handle:
            buf = self._new_device_buffer(4096)
            self.assertEqual(handle.read(buf, 4096, (8 << 20) + 4096), 0)
            # The last page still reads in full.
            self.assertEqual(handle.read(buf, 4096, (8 << 20) - 4096), 4096)


if __name__ == "__main__":
    unittest.main()
