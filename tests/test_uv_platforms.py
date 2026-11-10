"""Exercise universal uv resolution using small local wheels, without GPU downloads."""

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
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
                                         + "\nVersion: " + version + "\nRequires-Python: "
                                         + ('>=3.8' if group == 'orin-nx-jetpack511' else '>=3.10') + '\n'
                                         + ('Requires-Dist: torch\n' if name == 'torchvision' else ''))
                        archive.writestr(dist + "WHEEL", "Wheel-Version: 1.0\nTag: " + tag + "\n")
                        archive.writestr(dist + "RECORD", "")
                    required_marker = str(Requirement(selected[0]).marker)
                    source = next(s for s in settings['sources'][name]
                                  if str(Requirement('torch; ' + s['marker']).marker) == required_marker)
                    sources[name].append((wheel.name, group, source.get('marker')))
                    if group != 'orin-nx-jetpack511':
                        index = root / group / name
                        index.mkdir(parents=True, exist_ok=True)
                        (index / 'index.html').write_text('<a href="../../' + wheel.name + '">wheel</a>')
            config.extend(['[tool.uv]', 'environments = ' + json.dumps(settings["environments"])])
            if 'dependency-groups' in settings:
                config.append('[tool.uv.dependency-groups]')
                for group, value in settings['dependency-groups'].items():
                    config.append(group + ' = {requires-python = ' + json.dumps(value['requires-python']) + '}')
            class QuietHandler(SimpleHTTPRequestHandler):
                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=folder))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                def resolve(broad_environment=False):
                    base = 'http://127.0.0.1:%s' % server.server_port
                    content = config.copy()
                    if broad_environment:
                        content = [line if not line.startswith('environments =') else
                                   'environments = ["sys_platform == \'linux\' and '
                                   '(platform_machine == \'x86_64\' or platform_machine == \'aarch64\')"]'
                                   for line in content]
                    for group in ('common', 'thor'):
                        content.extend(['[[tool.uv.index]]', 'name = ' + json.dumps(group),
                                        'url = ' + json.dumps(base + '/' + group), 'explicit = true'])
                    content.append('[tool.uv.sources]')
                    for name, values in sources.items():
                        entries = []
                        for filename, group, marker in values:
                            if group == 'orin-nx-jetpack511':
                                entry = '{url = ' + json.dumps(base + '/' + filename)
                            else:
                                entry = '{index = ' + json.dumps(group)
                            if marker:
                                entry += ', marker = ' + json.dumps(marker)
                            entries.append(entry + '}')
                        content.append(name + ' = [' + ', '.join(entries) + ']')
                    (root / 'pyproject.toml').write_text('\n'.join(content))
                    lock = root / 'uv.lock'
                    if lock.exists():
                        lock.unlink()
                    return subprocess.run([os.environ.get('UV_TEST_BINARY', 'uv'), 'lock', '--no-python-downloads',
                                           '--python', sys.executable, '--directory', folder,
                                           '--cache-dir', str(root / 'cache')],
                                          capture_output=True, text=True, timeout=30)

                # Match the real mixture of indexes and URL wheels, including
                # torchvision's transitive Torch dependency. No conflict forks.
                before = resolve(broad_environment=True)
                self.assertNotEqual(before.returncode, 0)
                self.assertIn('matching Python version tag', before.stderr)
                result = resolve()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                lock = tomllib.loads((root / 'uv.lock').read_text())
                self.assertEqual(len([p for p in lock['package'] if p['name'] == 'torch']), 3)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
