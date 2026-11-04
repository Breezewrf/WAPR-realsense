"""Export an allowlisted source tree for a new WAPR-realsense GitHub repository."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ".gitignore", ".python-version", "pyproject.toml", "MANIFEST.in", "README.md", "README.zh-CN.md",
    "requirements.txt", "LICENSE", "AUTHORS.md", "WEIGHTS_LICENSE.txt",
    "THIRD_PARTY_NOTICES.txt", "UPSTREAM.md", ".github/workflows/ci.yml",
    "examples/realsense_pose.py", "examples/read_pose.py", "examples/README_realsense.md",
    "assets/meshes/README.md",
)
DIRECTORIES = ("wapr_realsense", "wapr", "configs", "docs", "scripts", "tests")
GENERATED_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".pyd", ".engine", ".pth", ".pt", ".onnx"}


def source_files(root=ROOT):
    paths = {root / name for name in FILES}
    if (root / "uv.lock").is_file():
        paths.add(root / "uv.lock")
    for name in DIRECTORIES:
        for path in (root / name).rglob("*"):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink() and "__pycache__" not in relative.parts
                    and not any(part.startswith(".") for part in relative.parts)
                    and path.suffix not in GENERATED_SUFFIXES):
                paths.add(path)
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing or symlinked source file: %s" % path)
        yield path


def export_source(destination, root=ROOT):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Destination exists; choose a new empty path: %s" % destination)
    paths = list(source_files(root))
    destination.mkdir(parents=True)
    manifest = {}
    for source in paths:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "SOURCE_MANIFEST.json").write_text(json.dumps({
        "project": "WAPR-realsense", "upstream_commit": "f165cec8ebdd48ae6f46d06a46c40d796ab80b71",
        "files_sha256": manifest,
        "excluded": ["Git history", "virtualenv", "runtime caches", "weights", "engines", "recordings",
                     "operator meshes", "upstream dataset/reconstruction/robot simulation examples", "upstream PyPI release workflow"],
    }, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--zip", action="store_true", help="Also create a sibling ZIP archive")
    args = parser.parse_args()
    archive = args.destination.resolve().with_suffix(".zip")
    if args.zip and archive.exists():
        parser.error("ZIP already exists; choose a different destination")
    destination = export_source(args.destination)
    if args.zip:
        archive_name = shutil.make_archive(str(archive.with_suffix("")), "zip", destination.parent, destination.name)
        print("ARCHIVE", archive_name)
    print("SOURCE_READY", destination)
    print("Source export does not include a Git repository or remote; see docs/publish.md.")


if __name__ == "__main__":
    main()
