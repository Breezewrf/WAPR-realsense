# Object meshes / 目标模型

`box/` is a bundled example, including:

- `textured_mesh_modified.obj`: mesh, with vertices in meters.
- `textured_mesh_modified.obj.mtl`: material definitions.
- `material_0.png`: texture.

The OBJ, MTL and texture must stay together with their relative paths preserved.
Use `configs/box.yaml` for manual ROI or `configs/box.example.yaml` for YOLOE initialization
with pre-exported reference embeddings from `assets/references/box/`.

`box/` 作为示例随仓库提供；其他目标 mesh 默认由 `.gitignore` 排除。
其他目标可放在独立子目录，或通过 `--mesh` 指定外部路径。
`mesh_unit` 支持 `m`、`mm`、`cm`，`mesh_scale` 是额外统一缩放；模型尺寸应与实物一致。
框选示例和 YOLOE 参考特征创建步骤见主 README。

Source licensing does not assign a license to an operator-supplied mesh or texture.
