"""Rotate a mesh, center its bounding box, then resize along XYZ.

Example:
    python scripts/resize_mesh.py model.obj --rotation 0 90 0 --size 0.17 0.36 0.33
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path, help='Input mesh path')
    parser.add_argument('--rotation', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=('X', 'Y', 'Z'),
                        help='Right-hand rotation angles in degrees about fixed X, then Y, then Z axes (default: 0 0 0)')
    parser.add_argument('--size', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
                        help='Final bounding-box dimensions along X/Y/Z after rotation')
    parser.add_argument('--unit', choices=('m', 'cm', 'mm'), default='m',
                        help='Unit of --size (default: m); output coordinates are always meters')
    parser.add_argument('--output', type=Path,
                        help='Output OBJ (default: <input folder>/<mesh stem>_scaled/model.obj)')
    args = parser.parse_args()
    angles = np.asarray(args.rotation)
    if not np.isfinite(angles).all():
        parser.error('Rotation angles must be finite')
    target = np.asarray(args.size) * {'m': 1.0, 'cm': 0.01, 'mm': 0.001}[args.unit]
    if not np.isfinite(target).all() or np.any(target <= 0):
        parser.error('Target dimensions must be finite and positive')
    source = args.mesh.expanduser().resolve()
    if not source.is_file():
        parser.error('Input mesh does not exist: %s' % source)
    output = (args.output.expanduser().resolve() if args.output else
              source.parent / (source.stem + '_scaled') / 'model.obj')
    if output.suffix.lower() != '.obj':
        parser.error('--output must end in .obj')
    mesh = trimesh.load(source, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        parser.error('Expected a nonempty triangle mesh')
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
        parser.error('Mesh coordinates must be finite with nonzero extent on every axis')
    original_size = mesh.extents.copy()
    rotation = trimesh.transformations.euler_matrix(*np.deg2rad(angles), axes='sxyz')
    mesh.apply_transform(rotation)
    rotated_size = mesh.extents.copy()
    # Center after rotation, so the final axis-aligned bounding box is centered.
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    scale = target / rotated_size
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scale)
    mesh.apply_transform(transform)
    # Export UVs and the color material together with their texture images.
    obj, textures = export_obj(mesh, include_texture=True, return_texture=True)
    files = {output: obj}
    for name, data in (textures or {}).items():
        path = (output.parent / name).resolve()
        if not path.is_relative_to(output.parent):
            parser.error('Exporter produced an invalid material path: %s' % name)
        files[path] = data
    for path in files:
        if path.exists():
            parser.error('Refusing to overwrite an existing file: %s' % path)
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data)
        else:
            path.write_bytes(data)
    print('Input XYZ:', original_size, '(input coordinates)')
    print('Rotation XYZ [degrees], fixed axes:', angles)
    print('Rotated XYZ:', rotated_size, '(input coordinates)')
    print('Axis scale:', scale)
    print('Output XYZ [m]:', mesh.extents)
    print('Output center [m]:', mesh.bounds.mean(axis=0))
    print('Saved:', output)
    print('Project config: mesh_unit: m, mesh_scale: 1.0')


if __name__ == '__main__':
    main()
