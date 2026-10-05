"""Deterministic ZIP built only from authoritative src/extension files."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package_bytes(source=ROOT / "src/extension"):
    source = Path(source)
    buffer = io.BytesIO()
    files = sorted(p for p in source.rglob("*") if p.is_file())
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            name = path.relative_to(source).as_posix()
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes().replace(b"\r\n", b"\n"))
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "nullius-extension.zip")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = package_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != data:
            parser.exit(1, "Package differs from authoritative source\n")
    else:
        # Preserve the previous untrusted package, never silently destroy it.
        if args.output.exists() and args.output.read_bytes() != data:
            old = args.output.read_bytes()
            backup = args.output.with_name(
                args.output.stem + ".previous-" + hashlib.sha256(old).hexdigest()[:12] + ".zip"
            )
            if not backup.exists():
                backup.write_bytes(old)
        args.output.write_bytes(data)
        args.output.with_suffix(".sha256").write_text(digest + "  " + args.output.name + "\n")
        manifest = {
            "sha256": digest,
            "source": "src/extension",
            "eol": "canonical LF",
            "compression": "stored",
            "timestamp": "1980-01-01T00:00:00",
            "files": {
                p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for p in sorted((ROOT / "src/extension").iterdir())
                if p.is_file()
            },
        }
        (ROOT / "docs/extension-package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(digest)


if __name__ == "__main__":
    main()
