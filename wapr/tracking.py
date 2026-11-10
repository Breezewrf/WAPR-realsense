# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Update known track identities together on one RGB-D frame.

在同一 RGB-D 帧中批量更新身份已知的跟踪实例。
Initialization, region association, and lost-track recovery belong to the caller.
初始化、区域关联与丢失恢复由调用方负责。
"""

from pathlib import Path

import numpy as np
import torch

from wapr import recipe
from wapr.estimator import _rgb_u8, center_from_mesh, poses_original_to_centered
from wapr.tracking_config import FULL_HYPOTHESES as tracking_hypotheses, tracking_settings
from wapr.nets import _Engine
from wapr.ogl import depth2xyzmap_batch, make_crop_pair
from wapr.pose_groups import wbps_group_sizes
from wapr.timing import pose_component_timer


def refine_batch(estimator, net_name, poses, rgb, depth_m, K, mesh, center, diameter_m, iters, max_rot):
    """Refine one mesh's hypothesis rows as a batch, with the mask channel off.

    关闭掩码通道，批量计算修正同一网格的候选姿态。
    """
    pose_t = torch.as_tensor(poses, device=estimator.device, dtype=torch.float32)
    mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    out = estimator._refine(
        estimator.nets[net_name], pose_t, rgb, depth_m, mask, K, mesh, center, diameter_m,
        iters=iters, max_rot=max_rot, crop_ratio=recipe.refine_crop_ratio, use_mask=False,
    )
    return out.detach().float().cpu().numpy()


def build_hypotheses(estimator, previous, center_pose, rgb, depth_m, K, mesh, center, diameter_m):
    """Build six candidates in tracking_hypotheses order, batching shared prefixes.

    按 tracking_hypotheses 顺序构造六个候选，共享前缀批量执行。
    """
    wapr_in = np.stack([center_pose, previous], axis=0)
    wapr_out = refine_batch(
        estimator, "wapr_wo_mask", wapr_in, rgb, depth_m, K, mesh, center, diameter_m,
        recipe.wapr_iters, recipe.wapr_max_rot_rad,
    )
    sapr_in = np.stack([center_pose, wapr_out[0], wapr_out[1]], axis=0)
    sapr_out = refine_batch(
        estimator, "sapr", sapr_in, rgb, depth_m, K, mesh, center, diameter_m,
        recipe.sapr_iters, recipe.sapr_max_rot_rad,
    )
    return np.stack([previous, center_pose, wapr_out[0], sapr_out[0], sapr_out[1], sapr_out[2]], axis=0)


def choose_pose(estimator, poses, rgb, depth_m, K, mesh, center, diameter_m):
    """Select the largest within_group score and return all between_group errors.

    选择 within_group 分数最大者，返回全部候选的 between_group 误差。
    """
    pose_t = torch.as_tensor(poses, device=estimator.device, dtype=torch.float32)
    mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    within, between = estimator._wbps(
        pose_t, rgb, depth_m, mask, K, mesh, center, diameter_m, len(poses),
    )
    within_np = within.reshape(-1).detach().float().cpu().numpy()
    between_np = between.reshape(-1).detach().float().cpu().numpy()
    return int(np.argmax(within_np)), between_np


def between_group_of_pose(estimator, pose, rgb, depth_m, mask, K, mesh, center, diameter_m):
    """Score a complete original-object pose group (N, 4, 4), N >= 2.

    评估原始物体系下的完整位姿组 (N, 4, 4)，N >= 2。
    """
    pose_t = torch.as_tensor(pose, device=estimator.device, dtype=torch.float32).reshape(-1, 4, 4)
    count = int(pose_t.shape[0])
    if count < 2:
        raise ValueError("Pass a pose group, or reuse the selected candidate's existing WBPS score")
    _within, between = estimator._wbps(
        pose_t, rgb, depth_m, mask, K, mesh, center, float(diameter_m), count,
    )
    return between.reshape(-1).detach().float().cpu().numpy()


def _tracking_score_engine(estimator, instance_count, group=6, wapr_rows=2, sapr_rows=3):
    """Use ordinary WBPS for one object; separate grouped engines for several."""
    wapr_engine = estimator.nets["wapr_wo_mask"].engine
    sapr_engine = estimator.nets["sapr"].engine
    if wapr_engine is None:
        return None, max(wbps_group_sizes) // group
    capacities = [int(sapr_engine.max_batch) // sapr_rows]
    if wapr_rows:
        capacities.append(int(wapr_engine.max_batch) // wapr_rows)
    base_capacity = min(capacities)
    if instance_count == 1:
        # wbps.engine accepts one complete dynamic group, including L=6.
        # A separate object-axis engine is needed only for multiple objects.
        return None, min(base_capacity, int(estimator.nets["wbps"].engine.max_batch) // group)
    engine_dir = Path(recipe.engine_file("wbps")).parent
    if (engine_dir / "wbps_batch.engine").is_file():
        return None, min(base_capacity, int(estimator.nets["wbps"].engine.max_batch) // group)
    if group != len(tracking_hypotheses):
        raise FileNotFoundError("Reduced multi-object tracking requires wbps_batch.engine: "
                                "python -m wapr.export_engines --object-mode multi --engine wbps_batch")
    score_engine = getattr(estimator, "tracking_score_engine", None)
    if score_engine is None:
        path = engine_dir / "wbps_tracking.engine"
        if not path.is_file():
            raise FileNotFoundError("Run python -m wapr.export_tracking_engine on the inference GPU: " + str(path))
        score_engine = _Engine(path, ("within_group", "between_group"),
                               execution_memory=getattr(wapr_engine, "execution_memory", None))
        if score_engine.min_batch != group or score_engine.max_batch != max(wbps_group_sizes):
            raise RuntimeError("Tracking engine profile differs; rebuild wbps_tracking.engine")
        estimator.tracking_score_engine = score_engine
    return score_engine, min(base_capacity, int(score_engine.max_batch) // group)


def _build_tracking_candidates(previous, centered, same_start, settings, reuse, refine):
    """Reuse only branches known identical from their CPU input poses.

    No approximate pose comparisons, GPU uniqueness scans or cross-object reuse.
    Return every scoring candidate, including duplicates in the full preset.
    """
    count = len(previous)
    if len(settings.hypotheses) != 6:
        owners = list(range(count))
        if settings.wapr_iters:
            wapr = refine("wapr_wo_mask", centered, owners, settings.wapr_iters)
            sapr = refine("sapr", wapr, owners, settings.sapr_iters)
            candidates = torch.stack((previous, wapr, sapr), dim=1)
        else:
            sapr = refine("sapr", centered, owners, settings.sapr_iters)
            candidates = torch.stack((previous, sapr), dim=1)
        rows = list(range(count * len(settings.hypotheses)))
        return candidates, rows, rows

    wapr_inputs, wapr_owners, wapr_indices = [], [], []
    for index in range(count):
        first = len(wapr_inputs)
        wapr_inputs.append(centered[index])
        wapr_owners.append(index)
        if not (reuse and same_start[index]):
            wapr_inputs.append(previous[index])
            wapr_owners.append(index)
        wapr_indices.extend((first, first if reuse and same_start[index] else first + 1))
    wapr_unique = refine("wapr_wo_mask", torch.stack(wapr_inputs), wapr_owners, settings.wapr_iters)
    wapr = wapr_unique[torch.tensor(wapr_indices, device=previous.device)].reshape(count, 2, 4, 4)

    sapr_inputs, sapr_owners, sapr_indices = [], [], []
    for index in range(count):
        first = len(sapr_inputs)
        sapr_inputs.extend((centered[index], wapr[index, 0]))
        sapr_owners.extend((index, index))
        if not (reuse and same_start[index]):
            sapr_inputs.append(wapr[index, 1])
            sapr_owners.append(index)
        sapr_indices.extend((first, first + 1, first + 1 if reuse and same_start[index] else first + 2))
    sapr_unique = refine("sapr", torch.stack(sapr_inputs), sapr_owners, settings.sapr_iters)
    sapr = sapr_unique[torch.tensor(sapr_indices, device=previous.device)].reshape(count, 3, 4, 4)
    candidates = torch.stack((previous, centered, wapr[:, 0], sapr[:, 0], sapr[:, 1], sapr[:, 2]), dim=1)

    crop_rows, restore_rows = [], []
    for index in range(count):
        offset = len(crop_rows)
        if reuse and same_start[index]:
            crop_rows.extend(index * 6 + value for value in (0, 2, 3, 4))
            restore_rows.extend(offset + value for value in (0, 0, 1, 2, 3, 3))
        else:
            crop_rows.extend(range(index * 6, (index + 1) * 6))
            restore_rows.extend(range(offset, offset + 6))
    return candidates, crop_rows, restore_rows


def track_many_categories_many_instances(estimator, rgb, depth_m, K, instances,
                                        tracking_preset="full", reuse_tracking=True):
    """Batch a configurable pose update for one or several tracked instances.

    批量执行一个或多个跟踪实例的六候选位姿更新。

    Each item supplies a unique track_id, mesh, diameter_m, and previous
    pose_4x4 (object to camera, meters). Optional center_pose_4x4 supplies the
    same rotation with translation from a propagated region and sensor depth;
    without it, the previous pose is used. obj_id is optional metadata.
    每项提供唯一 track_id、mesh、diameter_m，以及上一帧 pose_4x4（物体到
    相机、米）。可选 center_pose_4x4 保留原旋转，平移来自传播区域与传感器
    深度；未提供时使用上一帧位姿。obj_id 是可选类别信息。

    tracking_preset selects tracking-only iterations and scoring hypotheses.
    full keeps the original six positions/multiplicities; balanced/fast use
    previous, WAPR(center), SAPR(WAPR(center)); small-motion uses previous and
    SAPR(center). reuse_tracking only reuses full-preset branches with exactly
    equal input poses, and restores all six rows before WBPS attention.
    These presets never modify initialization settings in recipe.py.

    WAPR/SAPR and WBPS run across instances; each object's hypotheses remain an independent
    score group. A single object uses wbps.engine directly. Multiple
    objects use wbps_batch.engine, or the explicitly
    exported six-hypothesis tracking engine. Torch also scores groups together. Larger lists
    are chunked at the engine row limit. Empty input returns []. No GT or mask
    is read here, and no ID assignment, redetection, or automatic rescue occurs.
    within_group selects the pose; between_group is the complete group's maximum.
    WAPR/SAPR 与 WBPS 跨实例批量计算，每个实例的候选姿态仍独立评分。
    full 保留原六候选及重复次数；balanced/fast 使用上一帧、WAPR(center)、
    SAPR(WAPR(center))；small-motion 使用上一帧和 SAPR(center)。
    reuse_tracking 只复用输入位姿完全相同的 full 分支，并在评分前恢复六行。
    跟踪预设不会修改 recipe.py 的初始化配置。
    TensorRT 单物体直接使用 wbps.engine，多物体使用 wbps_batch.engine 或显式导出的
    六候选跟踪引擎；torch 路径同样批量计算评分。超过引擎行数
    上限时分块。空列表返回 []。此处不读取真值或 mask，不分配身份、不重新
    检测，也不自动恢复。输出顺序与输入一致，保留 track_id 和可选 obj_id。
    within_group 选择位姿；between_group 返回该实例完整候选组的最大值。
    """
    if not instances:
        return []
    settings = tracking_settings(tracking_preset)
    rgb_u8 = _rgb_u8(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    if depth.ndim != 2 or rgb_u8.shape[:2] != depth.shape:
        raise ValueError("RGB and depth sizes differ")
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    track_ids = [item["track_id"] for item in instances]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track_id must be unique, including instances of the same category")
    meshes = []
    centers = []
    diameters = []
    previous = []
    centered = []
    same_start = []
    prepared_meshes = estimator._pose_meshes([item["mesh"] for item in instances])
    prepared_ids = [estimator.renderer.cached_mesh_id(mesh) for mesh in prepared_meshes]
    for index, item in enumerate(instances):
        pose = np.asarray(item["pose_4x4"], dtype=np.float32)
        center_pose = np.asarray(item.get("center_pose_4x4", pose), dtype=np.float32)
        for candidate in (pose, center_pose):
            if candidate.shape != (4, 4) or not np.isfinite(candidate).all() or candidate[2, 3] <= 0:
                raise ValueError("Tracking requires a finite 4x4 pose with positive depth")
            if not np.allclose(candidate[3], [0, 0, 0, 1]):
                raise ValueError("Invalid homogeneous pose row")
        if not np.allclose(center_pose[:3, :3], pose[:3, :3]):
            raise ValueError("center_pose_4x4 must preserve the previous rotation")
        diameter = float(item["diameter_m"])
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("diameter_m must be finite and positive")
        mesh = prepared_meshes[index]
        meshes.append(mesh)
        centers.append(center_from_mesh(mesh))
        diameters.append(diameter)
        previous.append(pose)
        centered.append(center_pose)
        same_start.append(np.array_equal(center_pose, pose))

    group = len(settings.hypotheses)
    score_engine, objects_per_batch = _tracking_score_engine(estimator, len(instances), group,
                                                           settings.wapr_rows, settings.sapr_rows)
    if objects_per_batch < 1:
        raise RuntimeError("Engine profile cannot hold one tracking instance")
    rgb_device = torch.as_tensor(rgb_u8, device=estimator.device, dtype=torch.float32)
    depth_device = torch.as_tensor(depth, device=estimator.device, dtype=torch.float32)
    camera_device = torch.as_tensor(K, device=estimator.device, dtype=torch.float32)[None]
    observed_xyz = depth2xyzmap_batch(depth_device[None], camera_device).permute(0, 3, 1, 2)
    mask_device = torch.zeros_like(depth_device)
    results = []
    for start in range(0, len(instances), objects_per_batch):
        stop = min(start + objects_per_batch, len(instances))
        count = stop - start
        previous_t = torch.as_tensor(np.stack(previous[start:stop]), device=estimator.device)
        center_t = torch.as_tensor(np.stack(centered[start:stop]), device=estimator.device)
        batch_centers = np.stack(centers[start:stop]).astype(np.float32)
        batch_diameters = np.asarray(diameters[start:stop], dtype=np.float32)
        batch_meshes = meshes[start:stop]
        def refine_rows(name, poses, owners, iters):
            stage = "SAPR" if name == "sapr" else "WAPR"
            with pose_component_timer(stage, estimator.device):
                return estimator._refine(
                    estimator.nets[name], poses, rgb_device, depth_device, mask_device,
                    K, batch_meshes[0], batch_centers[owners], batch_diameters[owners],
                    iters=iters,
                    max_rot=recipe.sapr_max_rot_rad if name == "sapr" else recipe.wapr_max_rot_rad,
                    crop_ratio=recipe.refine_crop_ratio, use_mask=False,
                    meshes=[batch_meshes[index] for index in owners], group_size=None,
                    observed_xyz=observed_xyz,
                    mesh_ids=[prepared_ids[start + index] for index in owners],
                )

        candidates, crop_rows, restore_rows = _build_tracking_candidates(
            previous_t, center_t, same_start[start:stop], settings, reuse_tracking, refine_rows)
        with pose_component_timer('WBPS', estimator.device):
            crop_owners = np.asarray(crop_rows) // group
            score_centers = batch_centers[crop_owners]
            scoring_poses = candidates.reshape(count * group, 4, 4)
            if len(crop_rows) != count * group:
                scoring_poses = scoring_poses[torch.tensor(crop_rows, device=estimator.device)]
            score_poses = poses_original_to_centered(scoring_poses, score_centers)
            with pose_component_timer("WBPS crop", estimator.device):
                A, B, _poses, _diameters = make_crop_pair(
                    rgb_device, depth_device, mask_device, score_poses, K, batch_meshes[0],
                    batch_diameters[crop_owners], crop_ratio=recipe.wbps_crop_ratio,
                    use_mask=False, device=estimator.device,
                    meshes=[batch_meshes[index] for index in crop_owners],
                    observed_xyz=observed_xyz,
                    mesh_ids=[prepared_ids[start + index] for index in crop_owners],
                )
                if len(crop_rows) != count * group:
                    # Restore multiplicity/order BEFORE WBPS attention. Six scoring
                    # candidates remain six candidates even when only four crops ran.
                    restore = torch.tensor(restore_rows, device=estimator.device)
                    A, B = A[restore], B[restore]
            with pose_component_timer('WBPS network', estimator.device):
                with torch.no_grad():
                    if score_engine is None:
                        scores = estimator.nets["wbps"](A, B, L=group)
                    else:
                        scores = score_engine.run({"A": A, "B": B})
        if any(tuple(scores[name].shape) != (count, group)
               for name in ("within_group", "between_group")):
            raise RuntimeError("Tracking scorer must preserve the independent object/group axes")
        within = scores["within_group"].reshape(count, group)
        between = scores["between_group"].reshape(count, group)
        best = within.argmax(dim=1)
        selected = candidates[torch.arange(count, device=best.device), best].detach().float().cpu().numpy()
        # Quality uses the maximum along the complete candidate axis, separately
        # for each instance. within_group still selects the pose. Transfer once.
        # 质量分数沿完整候选轴取最大值，每实例独立；位姿仍由 within_group
        # 选择。每块只回传一次结果张量。
        group_between = between.max(dim=1).values.detach().float().cpu().numpy()
        choices = best.detach().cpu().numpy()
        for index in range(count):
            item = instances[start + index]
            pose = selected[index]
            row = {"track_id": item["track_id"], "pose_4x4": pose, "R": pose[:3, :3].copy(),
                   "t_m": pose[:3, 3].copy(), "hypothesis": settings.hypotheses[int(choices[index])],
                   "hypothesis_index": int(choices[index]), "between_group": float(group_between[index]),
                   "score_6d": (100.0 - float(group_between[index])) / 200.0}
            if "obj_id" in item:
                row["obj_id"] = item["obj_id"]
            results.append(row)
    return results
