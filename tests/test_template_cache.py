"""Cache migration and reuse need neither CUDA nor downloaded models."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from wapr.template_cache import select_cad_cache


class TemplateCacheTests(unittest.TestCase):
    def identity(self, source="old", mesh="mesh-a"):
        return {"source_sha256": source, "weight_sha256": "encoder-a",
                "meshes_glb_sha256": {"1": mesh}, "recipe": "test"}

    def write_cache(self, path, identity):
        path = Path(path)
        path.write_bytes(b"template-bank")
        Path(str(path) + ".json").write_text(json.dumps({
            "bank_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "weight_sha256": identity["weight_sha256"], "source_sha256": identity["source_sha256"]}))
        Path(str(path) + ".meshes.json").write_text(json.dumps(identity))

    def select(self, path, identity):
        with redirect_stdout(io.StringIO()):
            return select_cad_cache(path, identity, policy="version")

    def test_new_and_compatible_cache_keep_requested_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.pt"
            self.assertEqual(self.select(path, self.identity()), (str(path), False))
            self.write_cache(path, self.identity())
            self.assertEqual(self.select(path, self.identity()), (str(path), True))

    def test_source_change_builds_new_version_then_reuses_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.pt"
            self.write_cache(path, self.identity())
            old = {p: p.read_bytes() for p in Path(folder).iterdir()}
            changed = self.identity(source="new")
            version, hit = self.select(path, changed)
            self.assertFalse(hit)
            self.assertNotEqual(version, str(path))
            self.assertEqual(self.select(path, changed), (version, False))
            self.write_cache(version, changed)
            self.assertEqual(self.select(path, changed), (version, True))
            for p, data in old.items():
                self.assertEqual(p.read_bytes(), data)

    def test_mesh_change_uses_another_version_and_old_identity_still_works(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.pt"
            self.write_cache(path, self.identity())
            one, _ = self.select(path, self.identity(mesh="mesh-b"))
            two, _ = self.select(path, self.identity(mesh="mesh-c"))
            self.assertNotEqual(one, two)
            self.assertEqual(self.select(path, self.identity()), (str(path), True))

    def test_corrupt_version_is_preserved_and_repaired_copy_is_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.pt"
            self.write_cache(path, self.identity())
            changed = self.identity(source="new")
            version, _ = self.select(path, changed)
            self.write_cache(version, changed)
            Path(version).write_bytes(b"corrupt-bank")
            repaired, hit = self.select(path, changed)
            self.assertFalse(hit)
            self.assertNotEqual(repaired, version)
            self.write_cache(repaired, changed)
            self.assertEqual(self.select(path, changed), (repaired, True))
            self.assertEqual(Path(version).read_bytes(), b"corrupt-bank")

    def test_partial_or_malformed_cache_selects_new_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.pt"
            Path(str(path) + ".json").write_text("{")
            version, hit = self.select(path, self.identity())
            self.assertFalse(hit)
            self.assertNotEqual(version, str(path))
            self.write_cache(path, self.identity())
            Path(str(path) + ".meshes.json").write_text("{")
            self.assertFalse(self.select(path, self.identity())[1])

    def test_strict_api_still_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.pt"
            self.write_cache(path, self.identity())
            with self.assertRaisesRegex(ValueError, "different inputs/code"):
                select_cad_cache(path, self.identity(source="new"))


if __name__ == "__main__":
    unittest.main()
