#!/usr/bin/env python3
"""Recover complete, requested JSON members, including from a partial tar prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wanted = {p.stem for p in args.video_root.rglob("*.MP4")}
    args.output.mkdir(parents=True, exist_ok=True)
    found = []
    archive_complete = True
    try:
        with tarfile.open(args.archive, mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or name.suffix != ".json" or name.stem not in wanted:
                    continue
                stream = archive.extractfile(member)
                payload = stream.read()
                if len(payload) != member.size:
                    raise EOFError("incomplete member")
                data = json.loads(payload)
                if not data.get("images") or not data.get("camera"):
                    raise ValueError(f"invalid EPIC Fields schema for {name.name}")
                output = args.output / name.name
                if not output.exists():
                    output.write_bytes(payload)
                elif hashlib.sha256(output.read_bytes()).digest() != hashlib.sha256(payload).digest():
                    raise ValueError(f"existing JSON differs: {output}")
                row = {"video_id": name.stem, "registered_frames": len(data["images"]),
                       "file": str(output), "sha256": hashlib.sha256(payload).hexdigest()}
                found.append(row)
                print(json.dumps(row), flush=True)
    except (EOFError, tarfile.ReadError) as error:
        archive_complete = False
        print(f"Partial archive ended: {type(error).__name__}; only complete JSON members retained", flush=True)
    report = {"archive": str(args.archive), "archive_complete": archive_complete,
              "wanted_video_count": len(wanted), "extracted": found,
              "available_video_ids": sorted(p.stem for p in args.output.glob("*.json") if p.stem in wanted)}
    (args.output / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
