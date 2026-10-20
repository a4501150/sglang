"""Microbenchmark for the cuFile (GPUDirect Storage) binding.

Sequential and random batched reads through ``sglang.srt.utils.cufile``
into a registered device buffer, at 4 KiB (page) and 64 KiB (large IO)
granularity, against a scratch file under ``$HOME/.cache/sglang-test``.
Reports the active cuFile mode (gds / p2pdma / compat) next to every
number, because the data path dominates the result: compat bounces
through pinned host memory, gds and p2pdma DMA straight into VRAM.

Run:
    python benchmark/io/bench_cufile.py [--file-gib 8] [--ios 2048] [--rounds 5]
"""

import argparse
import os
import time

import torch

from sglang.srt.utils import cufile


def _cache_path(gib: float) -> str:
    root = os.path.expanduser(
        os.environ.get("SGLANG_CACHE_DIR", os.path.join("~", ".cache", "sglang"))
    )
    path = os.path.join(root, "sglang-test", "cufile", f"bench_{gib:g}gib.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbytes = int(gib * (1 << 30))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        print(f"writing {gib:g} GiB pattern file (one-time)...")
        block = os.urandom(1 << 20)
        with open(path, "wb") as f:
            for _ in range(nbytes >> 20):
                f.write(block)
    return path, nbytes


def _bench(fn, rounds: int) -> float:
    times = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-gib", type=float, default=8.0)
    parser.add_argument("--ios", type=int, default=2048, help="IOs per round")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not cufile.cufile_available():
        raise RuntimeError("libcufile.so not found")
    torch.zeros(8, device="cuda")
    driver = cufile.CuFileDriver.instance()
    mode = driver.mode()
    path, nbytes = _cache_path(args.file_gib)
    print(f"cuFile mode: {mode}  max_batch_io_size: {driver.max_batch_io_size}")
    print(f"file: {path} ({args.file_gib:g} GiB), {args.ios} IOs/round\n")

    buf = torch.empty(args.ios * (64 << 10), dtype=torch.uint8, device="cuda")
    cufile.register_buffer(buf, buf.numel())
    with cufile.CuFileHandle(path) as handle, cufile.CuFileBatch(args.ios) as batch:
        for io_size in (4 << 10, 64 << 10):
            per_io = args.ios

            def seq():
                ios = [
                    cufile.IoDesc(
                        handle=handle,
                        buf_ptr=buf.data_ptr(),
                        size=io_size,
                        file_offset=io_size * i,
                        buf_offset=io_size * i,
                    )
                    for i in range(per_io)
                ]
                batch.submit(ios)
                batch.wait(timeout_s=120)

            # Always starts the sequential window at file offset 0.
            dt = _bench(seq, args.rounds)
            total = io_size * per_io
            print(
                f"seq    {io_size >> 10:>3} KiB x {per_io}: {dt * 1e3:8.1f} ms  "
                f"{total / dt / 2**30:6.2f} GiB/s"
            )

            # Fresh offsets every round: reused ones are served by the SSD's
            # internal cache, the storage analog of a hot L2.
            def rnd():
                offs = (
                    torch.randint(0, nbytes - io_size, (per_io,)) & ~(io_size - 1)
                ).tolist()
                batch.submit(
                    [
                        cufile.IoDesc(
                            handle=handle,
                            buf_ptr=buf.data_ptr(),
                            size=io_size,
                            file_offset=off,
                            buf_offset=io_size * i,
                        )
                        for i, off in enumerate(offs)
                    ]
                )
                batch.wait(timeout_s=120)

            dt = _bench(rnd, args.rounds)
            print(
                f"random {io_size >> 10:>3} KiB x {per_io}: {dt * 1e3:8.1f} ms  "
                f"{per_io / dt / 1e3:6.0f} K IOPS"
            )

        # Single large streaming read, the HiCache-style bulk path.
        read_bytes = min(1 << 30, nbytes)

        def bulk():
            n = 0
            while n < read_bytes:
                n += handle.read(buf, min(64 << 20, read_bytes - n), n)
            return n

        dt = _bench(bulk, args.rounds)
        print(
            f"seq  handle.read 64 MiB chunks, {read_bytes >> 20} MiB: {dt * 1e3:8.1f} ms  "
            f"{read_bytes / dt / 2**30:6.2f} GiB/s"
        )

    print(f"\n(all numbers in cuFile mode: {mode})")


if __name__ == "__main__":
    main()
