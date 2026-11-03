# Create your own GitHub repository / 新建自己的仓库

The current checkout retains the original WAPR Git history/remote and reference examples.
Use the allowlisted export to start a focused **WAPR-realsense** repository:

```bash
python scripts/export_source.py .release/WAPR-realsense --zip
```

The destination must not exist; choose a new name for another export. The tool creates a
`SOURCE_MANIFEST.json` with SHA256 hashes and preserves source/weight/third-party license notices.

Included: `wapr_realsense/`, vendored `wapr/` backend, configs, docs, tests, two application examples,
packaging metadata, source-export/setup helpers and CPU CI.

Excluded: `.git`, `.venv`, runtime caches, checkpoints, ONNX, TensorRT engines, native binaries,
recordings, operator meshes/textures, the original benchmark/reconstruction/simulation examples,
and the upstream PyPI release workflow. Local files are not deleted.

Read the exported README. Add your own mesh locally or point the config to an external path.
Meshes are ignored by default; include an asset in a public repository only under the terms
you have selected for that asset. Preserve upstream `LICENSE`, `AUTHORS.md`, `UPSTREAM.md`,
`THIRD_PARTY_NOTICES.txt`, `WEIGHTS_LICENSE.txt` and bundled notices.

Create an empty GitHub repository named `WAPR-realsense` in your account. Then run these commands
inside the **exported directory**, substituting your actual repository URL:

```bash
cd .release/WAPR-realsense
git init -b main
git add .
git commit -m "Initialize WAPR-realsense robotics application"
git remote add origin https://github.com/YOUR_USERNAME/WAPR-realsense.git
git push -u origin main
```

The export tool does not initialize Git, change the original remote, create a GitHub repository,
publish a package or push anything. The Git commands above are the operator's publication step.
CI performs CPU contract tests and source-export checks; it does not claim GPU/camera validation.
