"""Exercise universal uv resolution using small local wheels, without GPU downloads."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:
    from pip._vendor import tomli as tomllib


class PlatformResolutionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("uv"), "uv is required")
    def test_all_platform_groups_lock_without_cross_python_wheel_errors(self):
        project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        settings = project["tool"]["uv"]
        groups = project["dependency-groups"]
        versions = {"common": ("2.7.1", "0.22.1", "cp310-cp310-linux_x86_64"),
                    "orin-nx-jetpack511": ("2.1.0a0+41361538.nv23.06", "0.16.2+c6f3977",
                                          "cp38-cp38-linux_aarch64"),
                    "thor": ("2.9.0", "0.24.0", "cp312-cp312-linux_aarch64")}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = ['[project]', 'name = "platform-resolution-check"', 'version = "0.1.0"',
                      'requires-python = ">=3.8,<3.13"', '[dependency-groups]']
            sources = {"torch": [], "torchvision": []}
            for group, requirements in groups.items():
                # Preserve actual group markers and version constraints, replacing only downloads.
                selected = [r for r in requirements if Requirement(r).name in sources]
                config.append(group + " = " + json.dumps(selected))
                for name, version in zip(sources, versions[group][:2]):
                    tag = versions[group][2]
                    wheel = root / (name + "-" + version + "-" + tag + ".whl")
                    dist = name + "-" + version + ".dist-info/"
                    with zipfile.ZipFile(wheel, "w") as archive:
                        archive.writestr(dist + "METADATA", "Metadata-Version: 2.1\nName: " + name
                                         + "\nVersion: " + version + "\n")
                        archive.writestr(dist + "WHEEL", "Wheel-Version: 1.0\nTag: " + tag + "\n")
                        archive.writestr(dist + "RECORD", "")
                    sources[name].append('{path = ' + json.dumps(str(wheel))
                                         + ', group = ' + json.dumps(group) + '}')
            config.extend(['[tool.uv]', 'environments = ' + json.dumps(settings["environments"]),
                           'conflicts = [[' + ', '.join('{group = ' + json.dumps(g) + '}'
                                                        for g in groups) + ']]',
                           '[tool.uv.dependency-groups]'])
            for group, value in settings["dependency-groups"].items():
                config.append(group + ' = {requires-python = ' + json.dumps(value["requires-python"]) + '}')
            config.append('[tool.uv.sources]')
            for name, values in sources.items():
                config.append(name + ' = [' + ', '.join(values) + ']')
            (root / "pyproject.toml").write_text('\n'.join(config))
            result = subprocess.run(["uv", "lock", "--offline", "--no-python-downloads",
                                     "--python", sys.executable, "--directory", str(root),
                                     "--cache-dir", str(root / "cache")],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lock = tomllib.loads((root / "uv.lock").read_text())
            self.assertEqual(len([p for p in lock["package"] if p["name"] == "torch"]), 3)


if __name__ == "__main__":
    unittest.main()
