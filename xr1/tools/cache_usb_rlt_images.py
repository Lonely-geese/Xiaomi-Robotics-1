"""Decode USB HEVC once into lossless, resized uint8 arrays for random sampling."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def convert(job):
    import av
    import numpy as np
    from PIL import Image

    av.logging.set_level(None)
    src, dst, count = job
    src, dst = Path(src), Path(dst)
    fingerprint = {
        "size": src.stat().st_size,
        "mtime_ns": src.stat().st_mtime_ns,
        "frames": count,
    }
    meta = dst.with_suffix(".json")
    if (
        dst.exists()
        and meta.exists()
        and json.loads(meta.read_text()).get("source") == fingerprint
    ):
        return {"file": dst.name, "cached": True}
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_suffix(".npy.tmp")
    array = None
    n = 0
    with av.open(str(src)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "SLICE"
        rate = float(stream.average_rate or stream.guessed_rate)
        start = int(stream.start_time or 0)
        for frame in container.decode(stream):
            expected = start + round(n / rate / float(stream.time_base))
            if frame.pts != expected:
                raise ValueError(f"Nonuniform video PTS {src}:{n}")
            image = frame.to_image()
            w, h = image.size
            nw = max(32, round(w / 32) * 32)
            nh = max(32, round(h / 32) * 32)
            if nw * nh > 160000:
                scale = (h * w / 160000) ** 0.5
                nh = int(h / scale // 32) * 32
                nw = int(w / scale // 32) * 32
            elif nw * nh < 4 * 28 * 28:
                import math

                scale = (4 * 28 * 28 / (h * w)) ** 0.5
                nh = math.ceil(h * scale / 32) * 32
                nw = math.ceil(w * scale / 32) * 32
            image = image.resize((nw, nh), resample=Image.Resampling.BICUBIC)
            if array is None:
                array = np.lib.format.open_memmap(
                    temp, mode="w+", dtype=np.uint8, shape=(count, nh, nw, 3)
                )
            if n >= count:
                raise ValueError(f"Too many frames: {src}")
            array[n] = np.asarray(image)
            n += 1
        if n != count:
            raise ValueError(f"Frame count mismatch: {src} {n}!={count}")
    array.flush()
    del array
    temp.replace(dst)
    meta.write_text(
        json.dumps({"source": fingerprint, "format": "resized-rgb-uint8", "frames": n})
        + "\n"
    )
    return {"file": dst.name, "cached": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/data/wz/usb/usb")
    p.add_argument("--cache", default="/data/wz/usb/rlt_image_cache")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    root = Path(a.root)
    cache = Path(a.cache)
    cache.mkdir(exist_ok=True)
    records = [
        json.loads(x) for x in (root / "meta/manifest.jsonl").read_text().splitlines()
    ]
    jobs = [
        (str(root / v), str(cache / (Path(v).stem + ".npy")), r["length"])
        for r in records
        for v in r["video_paths"].values()
    ]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(convert, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            f.result()
            if i % 20 == 0 or i == len(jobs):
                status = {
                    "done": i,
                    "total": len(jobs),
                    "seconds": time.monotonic() - start,
                }
                (cache / "status.json").write_text(json.dumps(status))
                print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
