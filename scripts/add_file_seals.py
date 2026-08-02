"""One-time Day-27 seal migration: add byte seals beside the frame seals.

The Day-1 manifest sealed each split as SHA-256 over the *re-serialised
frame* (``df.to_parquet(index=False)``), which only verifies in the sealing
environment's exact pandas/pyarrow (the reason the Docker image pins
pandas ==2.1.4). This script upgrades every sealed dataset — the Day-1
benchmark manifest and any per-upload manifests — to ALSO carry
``train_file_sha256`` / ``holdout_file_sha256`` over the files' raw bytes,
which any environment can verify.

Chain of custody: a byte seal is only recorded after the existing FRAME seal
re-verifies right here, in the sealing environment — so the new seal provably
fingerprints the same untouched data the Day-1 seal did, not a later state.
The frame seals are kept untouched (they remain the original commitment).

Idempotent: datasets already carrying byte seals are re-verified against
them and skipped. Any mismatch aborts loudly — never reseal over a failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.datasets import (  # noqa: E402
    MANIFEST_PATH,
    holdout_path,
    load_holdout,
    load_train,
    sha256_of_file,
    sha256_of_frame,
    train_path,
)
from crewml.uploads import discover_upload_manifests  # noqa: E402


def _upgrade(entry: dict, key: str) -> str:
    """Verify frame seals, then add/verify byte seals in-place. Returns action."""
    frames = {"train": load_train(key), "holdout": load_holdout(key)}
    paths = {"train": train_path(key), "holdout": holdout_path(key)}
    for split, df in frames.items():
        recorded = entry[f"{split}_sha256"]
        current = sha256_of_frame(df)
        if recorded != current:
            raise SystemExit(
                f"ABORT: {key} {split} FRAME seal mismatch "
                f"({recorded[:12]}… != {current[:12]}…) — refusing to add a "
                "byte seal on top of an unverified split. Investigate first."
            )
    action = "added"
    for split in ("train", "holdout"):
        field = f"{split}_file_sha256"
        digest = sha256_of_file(paths[split])
        if entry.get(field):
            if entry[field] != digest:
                raise SystemExit(
                    f"ABORT: {key} {split} BYTE seal mismatch — file changed "
                    "since byte-sealing. Investigate first."
                )
            action = "verified (already sealed)"
        else:
            entry[field] = digest
    return action


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for key, entry in manifest["datasets"].items():
        print(f"{key}: {_upgrade(entry, key)}")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")

    for mpath in discover_upload_manifests():
        entry = json.loads(mpath.read_text(encoding="utf-8"))
        print(f"{entry['key']} (upload): {_upgrade(entry, entry['key'])}")
        mpath.write_text(json.dumps(entry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
