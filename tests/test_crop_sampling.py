"""CPU numerical comparison against independent Kornia perspective warps.

Rendering is supplied as fixed RGB-D tensors; this does not validate CUDA/GL.
"""

import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from wapr import ogl, recipe


@unittest.skipUnless(importlib.util.find_spec("kornia"), "Kornia is not installed")
class CropSamplingTests(unittest.TestCase):
    def test_shared_grid_matches_kornia_with_bilinear_and_nearest_sampling(self):
        from kornia.geometry.transform import warp_perspective
        for dtype in (torch.float32, torch.float64):
            transforms = torch.tensor([
                [[0.8, 0, -2], [0, 1.2, 3], [0, 0, 1]],
                [[1.1, 0.1, 4], [-0.1, 0.9, -3], [0.001, -0.002, 1]],
            ], dtype=dtype)
            image = torch.arange(3 * 19 * 23, dtype=dtype).reshape(1, 3, 19, 23).expand(2, -1, -1, -1)
            grid = ogl._crop_sampling_grid(transforms, (19, 23), (11, 13))
            for mode in ("bilinear", "nearest"):
                with self.subTest(dtype=dtype, mode=mode):
                    expected = warp_perspective(image, transforms, (11, 13), mode=mode, align_corners=False)
                    actual = F.grid_sample(image, grid, mode=mode, padding_mode="zeros", align_corners=False)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_crop_pairs_preserve_mask_xyz_and_rgb_with_one_grid_and_no_scalar_read(self):
        from kornia.geometry.transform import warp_perspective
        generator = torch.Generator().manual_seed(17)
        height, width, crop = 31, 43, 16
        rgb = torch.rand(height, width, 3, generator=generator) * 255
        depth = torch.rand(height, width, generator=generator) + 0.3
        depth[::3, ::4] = 0
        mask = (torch.rand(height, width, generator=generator) > 0.5).float()
        camera = torch.tensor([[60., 0, 21.], [0, 58., 15.], [0, 0, 1.]])
        mesh = object()
        for count in (1, 2, 3, 6, 12):
            poses = torch.eye(4).repeat(count, 1, 1)
            poses[:, 0, 3] = torch.linspace(-0.4, 0.3, count)
            poses[:, 2, 3] = torch.linspace(0.6, 1.5, count)
            diameters = torch.linspace(0.2, 0.5, count)
            cameras = camera[None].expand(count, -1, -1)
            transform, boxes = ogl.compute_crop_window(poses, cameras, 1.2, (crop, crop), diameters)
            observed_xyz = ogl.depth2xyzmap_batch(depth[None], camera[None]).permute(0, 3, 1, 2)
            rendered_rgb = torch.rand(count, crop, crop, 3, generator=generator)
            rendered_depth = torch.rand(count, crop, crop, generator=generator) + 0.3
            rendered_depth[:, ::4, ::3] = 0
            observed_rgb = warp_perspective(rgb[None].permute(0, 3, 1, 2).expand(count, -1, -1, -1),
                transform, (crop, crop), align_corners=False) / 255
            warped_xyz = warp_perspective(observed_xyz.expand(count, -1, -1, -1), transform,
                (crop, crop), mode="nearest", align_corners=False)
            expected_xyz_b = ogl.normalize_xyz_maps(warped_xyz, poses, diameters)
            adjusted_camera = ogl._crop_adjusted_K(cameras, boxes, crop, crop)
            xyz_a = ogl.depth2xyzmap_batch(rendered_depth, adjusted_camera).permute(0, 3, 1, 2)
            expected_xyz_a = ogl.normalize_xyz_maps(xyz_a, poses, diameters)
            # Match the previous rendered RGB multiplication/division sequence.
            expected_rgb_a = (rendered_rgb * 255).permute(0, 3, 1, 2).contiguous() / 255
            for use_mask in (False, True):
                for cached_xyz in (None, observed_xyz):
                    with self.subTest(count=count, mask=use_mask, cached_xyz=cached_xyz is not None):
                        channels_a, channels_b = [expected_rgb_a], [observed_rgb]
                        if use_mask:
                            warped_mask = warp_perspective(mask[None, None].expand(count, -1, -1, -1),
                                transform, (crop, crop), mode="nearest", align_corners=False)
                            channels_a.append((rendered_depth[:, None] > 0).float())
                            channels_b.append((warped_mask > 0).float())
                        expected_a = torch.cat(channels_a + [expected_xyz_a], dim=1)
                        expected_b = torch.cat(channels_b + [expected_xyz_b], dim=1)
                        runtime = Mock()
                        runtime.render_tiles.return_value = rendered_rgb, rendered_depth
                        with patch.object(recipe, "crop_px", crop), patch.object(ogl, "runtime_for", return_value=runtime), \
                             patch.object(ogl, "_crop_sampling_grid", wraps=ogl._crop_sampling_grid) as sampling, \
                             patch("torch.Tensor.__float__", side_effect=AssertionError("GPU scalar read")), \
                             patch("torch.nn.functional.grid_sample", wraps=F.grid_sample) as sample:
                            A, B, pose_a, diameter = ogl.make_crop_pair(rgb, depth, mask, poses, camera,
                                mesh, diameters, 1.2, use_mask, "cpu", observed_xyz=cached_xyz,
                                mesh_ids=[1] * count)
                        sampling.assert_called_once()
                        self.assertEqual(sample.call_count, 3 if use_mask else 2)
                        torch.testing.assert_close(A, expected_a, rtol=1e-6, atol=1e-7)
                        torch.testing.assert_close(B, expected_b, rtol=0, atol=0)
                        torch.testing.assert_close(pose_a, poses)
                        torch.testing.assert_close(diameter, diameters)
                        self.assertTrue(A.is_contiguous() and B.is_contiguous())


if __name__ == "__main__":
    unittest.main()
