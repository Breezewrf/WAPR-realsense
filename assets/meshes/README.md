# Object meshes / 目标模型

Place your own metric object mesh here, or pass an external path with `--mesh`.
OBJ material (`.mtl`) and texture images must retain their relative paths.

模型顶点应与实物尺寸一致。`mesh_unit` 支持 `m`、`mm`、`cm`；`mesh_scale`
是额外的统一缩放。包围盒直径由加载后顶点计算，不手填类别训练参数。

`configs/box.json` references `box/textured_mesh_modified.obj` in the local workspace.
The local box is **not bundled** in source exports or automatically added to Git.
After creating the new repository, copy your mesh into this directory or edit the config.
Source licensing does not assign a license to an operator-supplied mesh or texture.
