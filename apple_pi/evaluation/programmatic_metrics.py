#!/usr/bin/env python3
"""
Evaluate metrics for generated videos against Ground Truth frames.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import cv2
import math
import json
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod

import torch
from PIL import Image
from torchvision import transforms as T

from apple_pi.constants import (
    MOGE_MODEL_REPO,
    MOGE_MODEL_REVISION,
    SAM3_MODEL_REPO,
    SAM3_MODEL_REVISION,
)
from apple_pi.evaluation.gt_frames import load_gt_frames

# SAM3 model imports
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from transformers.video_utils import load_video

# MoGe depth estimation model imports (lazy — only needed for VelocityMetric)
try:
    from moge.model.v2 import MoGeModel
except ImportError:
    MoGeModel = None

# --- Metric Definitions ---

class BaseMetric(ABC):
    """Abstract base class for video metrics."""

    @property
    @abstractmethod
    def name(self):
        """Name of the metric."""
        pass

    @abstractmethod
    def compute(self, pred_frame, gt_frame, **kwargs):
        """
        Compute metric for a single frame pair.
        Args:
            pred_frame: Predicted frame (numpy array)
            gt_frame: Ground truth frame (numpy array)
            **kwargs: Additional metric-specific parameters.
        Returns:
            float: Metric value
        """
        pass

    def aggregate(self, frame_scores):
        """Aggregate per-frame scores into a video score."""
        if not frame_scores:
            return 0.0
        return np.mean(frame_scores)

class PSNRMetric(BaseMetric):
    """Peak Signal-to-Noise Ratio (PSNR) Metric."""

    @property
    def name(self):
        return "psnr"

    def compute(self, pred_frame, gt_frame, **kwargs):
        assert pred_frame.shape[:2] == gt_frame.shape[:2], (
            f"Resolution mismatch: pred {pred_frame.shape[1]}x{pred_frame.shape[0]} "
            f"vs gt {gt_frame.shape[1]}x{gt_frame.shape[0]}"
        )

        img1 = pred_frame.astype(np.float64)
        img2 = gt_frame.astype(np.float64)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')

        pixel_max = 255.0
        return 20 * math.log10(pixel_max / math.sqrt(mse))

class MaskedPSNRMetric(BaseMetric):
    """Per-object Masked PSNR.

    For each object that appears in both pred and GT, crop the object region
    from each frame using its own mask bounding box, resize the pred crop to
    match the GT crop size, then compute PSNR on the intersection of the two
    masks.  This evaluates visual quality of each object independently of its
    spatial position.
    """

    @property
    def name(self):
        return "masked_psnr"

    def compute(self, pred_frame, gt_frame, pred_obj_masks=None,
                gt_obj_masks=None, **kwargs):
        """
        Args:
            pred_obj_masks: dict {gt_instance_id: binary_mask (H, W)} from SAM3.
            gt_obj_masks:   dict {gt_instance_id: binary_mask (H, W)} from GT.
        """
        assert pred_obj_masks is not None and gt_obj_masks is not None, "pred_obj_masks and gt_obj_masks must be provided"

        common_ids = set(pred_obj_masks.keys()) & set(gt_obj_masks.keys())
        if not common_ids:
            return 0.0

        psnr_scores = []
        for obj_id in common_ids:
            score = self._object_psnr(
                pred_frame, gt_frame,
                pred_obj_masks[obj_id], gt_obj_masks[obj_id],
            )
            psnr_scores.append(score)

        if not psnr_scores:
            return 0.0
        return float(np.mean(psnr_scores))

    @staticmethod
    def _bbox(mask):
        """Return (y1, y2, x1, x2) bounding box of non-zero region, or None."""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    @staticmethod
    def _object_psnr(pred_frame, gt_frame, pred_mask, gt_mask):
        pred_bbox = MaskedPSNRMetric._bbox(pred_mask)
        gt_bbox = MaskedPSNRMetric._bbox(gt_mask)

        py1, py2, px1, px2 = pred_bbox
        gy1, gy2, gx1, gx2 = gt_bbox

        pred_crop = pred_frame[py1:py2, px1:px2]
        gt_crop = gt_frame[gy1:gy2, gx1:gx2]
        pred_mask_crop = pred_mask[py1:py2, px1:px2]
        gt_mask_crop = gt_mask[gy1:gy2, gx1:gx2]

        th, tw = gt_crop.shape[:2]

        pred_crop_r = cv2.resize(pred_crop, (tw, th))
        pred_mask_r = cv2.resize(
            pred_mask_crop.astype(np.uint8), (tw, th),
            interpolation=cv2.INTER_NEAREST,
        )

        valid = (pred_mask_r > 0) & (gt_mask_crop > 0)

        img1 = pred_crop_r.astype(np.float64)
        img2 = gt_crop.astype(np.float64)

        mse = np.mean((img1[valid] - img2[valid]) ** 2)
        if mse == 0:
            return float('inf')

        return 20 * math.log10(255.0 / math.sqrt(mse))


# --- Video-level Mask Metrics ---

class BaseVideoMaskMetric(ABC):
    """Abstract base class for video-level mask metrics."""

    @property
    @abstractmethod
    def name(self):
        """Name of the metric."""
        pass

    @abstractmethod
    def compute(self, pred_masks, gt_masks):
        """
        Compute metric for mask arrays.
        Args:
            pred_masks: Predicted masks (T, H, W) numpy array, values 0 or 1
            gt_masks: Ground truth masks (T, H, W) numpy array, values 0 or 1
        Returns:
            float: Metric value
        """
        pass

class SpatialIoUMetric(BaseVideoMaskMetric):
    """
    Spatial IoU: Collapse masks over time dimension (max), then compute IoU.
    """

    @property
    def name(self):
        return "spatial_iou"

    def compute(self, pred_masks, gt_masks):
        assert pred_masks is not None and gt_masks is not None, "pred_masks and gt_masks must be provided"
        assert pred_masks.shape == gt_masks.shape, f"Mask shape mismatch: expected {pred_masks.shape}, got {gt_masks.shape}"

        # Collapse time dimension using max
        pred_spatial = np.max(pred_masks, axis=0)
        gt_spatial = np.max(gt_masks, axis=0)

        # Binarize (should already be 0/1, but just in case)
        pred_spatial = (pred_spatial > 0).astype(np.uint8)
        gt_spatial = (gt_spatial > 0).astype(np.uint8)

        intersection = np.logical_and(pred_spatial, gt_spatial).sum()
        union = np.logical_or(pred_spatial, gt_spatial).sum()

        if union == 0:
            return 1.0  # Both masks are empty
        return float(intersection) / float(union)

class SpatiotemporalIoUMetric(BaseVideoMaskMetric):
    """
    Spatiotemporal IoU: Compute per-frame IoU and average.
    """

    @property
    def name(self):
        return "spatiotemporal_iou"

    def compute(self, pred_masks, gt_masks):
        assert pred_masks is not None and gt_masks is not None, "pred_masks and gt_masks must be provided"
        assert pred_masks.shape == gt_masks.shape, f"Mask shape mismatch: expected {pred_masks.shape}, got {gt_masks.shape}"

        iou_values = []
        for i in range(gt_masks.shape[0]):
            pred_frame = (pred_masks[i] > 0).astype(np.uint8)
            gt_frame = (gt_masks[i] > 0).astype(np.uint8)

            intersection = np.logical_and(pred_frame, gt_frame).sum()
            union = np.logical_or(pred_frame, gt_frame).sum()

            if union == 0:
                iou = 1.0  # Both masks are empty
            else:
                iou = float(intersection) / float(union)
            iou_values.append(iou)

        return np.mean(iou_values)

class WeightedSpatialIoUMetric(BaseVideoMaskMetric):
    """
    Weighted Spatial IoU: Create weighted spatial masks (average over time),
    then compute weighted IoU.
    """

    @property
    def name(self):
        return "weighted_spatial_iou"

    def compute(self, pred_masks, gt_masks):
        assert pred_masks is not None and gt_masks is not None, "pred_masks and gt_masks must be provided"
        assert pred_masks.shape == gt_masks.shape, f"Mask shape mismatch: expected {pred_masks.shape}, got {gt_masks.shape}"

        # Compute weighted spatial masks (average over time)
        pred_weighted = np.mean(pred_masks.astype(np.float32), axis=0)
        gt_weighted = np.mean(gt_masks.astype(np.float32), axis=0)

        # Compute intersection and union
        intersection = np.minimum(pred_weighted, gt_weighted)
        union = np.maximum(pred_weighted, gt_weighted)

        # Pixels where motion exists in at least one
        valid_pixels = union > 0

        if np.sum(valid_pixels) == 0:
            return 1.0  # Perfect match (both empty)

        return float(np.sum(intersection[valid_pixels])) / float(np.sum(union[valid_pixels]))

class VelocityMetric(BaseVideoMaskMetric):
    @property
    def name(self):
        return "velocity_error"

    def compute(self, pred_masks, gt_masks):
        """
        This method is not used directly for velocity error.
        The actual computation is done in compute_velocity_error which takes additional parameters.
        """
        return 0.0

    def compute_velocity_error(self, aligned_depths, video_segments,
                               sampled_video_idx, n_gt, valid_gt_ids,
                               gt_velocity_path, cam_params_dir, obj_ids,
                               gt_ins_seg_path=None,
                               save_velocity_dir=None, gt_fps=24):
        """
        Compute per-object velocity error in GT physics-frame space.

        ``aligned_depths[0]`` is the condition frame, ``aligned_depths[k]``
        for ``k=1..n_gt`` are physics frames.  For each consecutive pair
        ``(k, k+1)`` the estimated 3D velocity is compared against GT
        velocity at GT npz index ``k+1``.

        Args:
            aligned_depths: Shape ``(1+n_gt, H, W)``: index 0 = condition
                frame, indices 1..n_gt = physics frames.
            video_segments: Dict ``{pred_frame_idx: {sam3_obj_id: mask_tensor}}``.
            sampled_video_idx: List mapping GT physics frame *i* → pred video
                frame index used for SAM3 mask lookup.
            n_gt: Number of GT physics frames.
            valid_gt_ids: List mapping SAM3 obj index → original GT instance ID.
            gt_velocity_path: Path to GT velocity ``.npz`` (includes condition
                frame at index 0).
            cam_params_dir: Directory with per-frame ``XXXX.json`` camera
                parameters (index 0 = condition frame).
            obj_ids: SAM3 object IDs to evaluate.
            gt_ins_seg_path: Path to GT instance segmentation ``.npz``.
            save_velocity_dir: Directory to save velocity visualisation videos.
            gt_fps: GT frame rate (used for displacement → velocity conversion).

        Returns:
            float: mean per-object velocity error, or ``None`` on failure.
        """
        try:
            assert aligned_depths is not None and video_segments is not None
            assert gt_velocity_path is not None and os.path.exists(gt_velocity_path)
            assert cam_params_dir is not None and os.path.isdir(cam_params_dir)
            assert gt_ins_seg_path is not None and os.path.exists(gt_ins_seg_path)

            gt_velocity = np.load(gt_velocity_path)['maps']
            gt_ins_seg = np.load(gt_ins_seg_path)['maps']

            H, W = aligned_depths.shape[1], aligned_depths.shape[2]
            num_objects = len(obj_ids)
            # aligned_depths: [0]=condition, [1..n_gt]=physics → n_gt pairs
            num_pairs = n_gt

            # Pred video indices: condition at 0, physics from sampled_video_idx
            pred_frame_indices = [0] + list(sampled_video_idx)  # length 1+n_gt

            object_errors = {oi: [] for oi in range(num_objects)}
            estimated_velocity = np.zeros((num_pairs, H, W, 3), dtype=np.float32)
            gt_velocity_vis = np.zeros((num_pairs, H, W, 3), dtype=np.float32)

            def _to_np_mask(tensor):
                if hasattr(tensor, 'cpu'):
                    arr = tensor.cpu().float().numpy().squeeze()
                else:
                    arr = np.asarray(tensor).squeeze()
                if arr.ndim > 2:
                    arr = arr[0]
                return (arr > 0).astype(bool)

            for i in range(num_pairs):
                # aligned_depths[i] and [i+1]; index == GT npz index
                depth_curr = aligned_depths[i]
                depth_next = aligned_depths[i + 1]

                gt_npz_curr = i
                gt_npz_next = i + 1

                cam_file_curr = os.path.join(cam_params_dir,
                                             f"{gt_npz_curr:04d}.json")
                cam_file_next = os.path.join(cam_params_dir,
                                             f"{gt_npz_next:04d}.json")

                with open(cam_file_curr, 'r') as f:
                    cam_curr = json.load(f)
                with open(cam_file_next, 'r') as f:
                    cam_next = json.load(f)

                intrinsics_curr = np.array(cam_curr["intrinsic"])
                fx_c, fy_c = intrinsics_curr[0, 0], intrinsics_curr[1, 1]
                cx_c, cy_c = intrinsics_curr[0, 2], intrinsics_curr[1, 2]

                intrinsics_next = np.array(cam_next["intrinsic"])
                fx_n, fy_n = intrinsics_next[0, 0], intrinsics_next[1, 1]
                cx_n, cy_n = intrinsics_next[0, 2], intrinsics_next[1, 2]

                extrinsics_curr = np.array(cam_curr["extrinsic"])
                R_c, t_c = extrinsics_curr[:3, :3], extrinsics_curr[:3, 3]
                extrinsics_next = np.array(cam_next["extrinsic"])
                R_n, t_n = extrinsics_next[:3, :3], extrinsics_next[:3, 3]

                # Pred masks from SAM3 at the corresponding pred video indices
                pred_idx_curr = pred_frame_indices[i]
                pred_idx_next = pred_frame_indices[i + 1]
                masks_curr = video_segments.get(pred_idx_curr, {})
                masks_next = video_segments.get(pred_idx_next, {})

                # GT instance seg at the "next" GT frame
                gt_seg_next = gt_ins_seg[min(gt_npz_next,
                                             gt_ins_seg.shape[0] - 1)]
                # GT velocity at the "next" GT frame
                gt_vel_frame = gt_velocity[min(gt_npz_next,
                                               gt_velocity.shape[0] - 1)]
                gt_velocity_vis[i] = gt_vel_frame

                for obj_idx, pred_obj_id in enumerate(obj_ids):
                    if pred_obj_id not in masks_curr or pred_obj_id not in masks_next:
                        continue

                    mask_c = _to_np_mask(masks_curr[pred_obj_id])
                    mask_n = _to_np_mask(masks_next[pred_obj_id])
                    assert mask_c.shape == mask_n.shape == (H, W), f"Mask shape mismatch: expected {(H, W)}, got {mask_c.shape}, {mask_n.shape}"

                    if mask_c.sum() < 10 or mask_n.sum() < 10:
                        continue

                    # Back-project current frame
                    y_c, x_c = np.where(mask_c)
                    z_c = depth_curr[mask_c]
                    vz = np.isfinite(z_c)
                    if vz.sum() < 10:
                        continue
                    z_c, x_c, y_c = z_c[vz], x_c[vz], y_c[vz]
                    pts_cam_c = np.stack([
                        (x_c - cx_c) * z_c / fx_c,
                        (y_c - cy_c) * z_c / fy_c,
                        z_c,
                    ], axis=-1)

                    # Back-project next frame
                    y_n, x_n = np.where(mask_n)
                    z_n = depth_next[mask_n]
                    vz2 = np.isfinite(z_n)
                    if vz2.sum() < 10:
                        continue
                    z_n, x_n, y_n = z_n[vz2], x_n[vz2], y_n[vz2]
                    pts_cam_n = np.stack([
                        (x_n - cx_n) * z_n / fx_n,
                        (y_n - cy_n) * z_n / fy_n,
                        z_n,
                    ], axis=-1)

                    # Camera → world
                    pts_w_c = (pts_cam_c - t_c) @ R_c
                    pts_w_n = (pts_cam_n - t_n) @ R_n

                    ok_c = np.isfinite(pts_w_c).all(1) & (z_c != 0)
                    ok_n = np.isfinite(pts_w_n).all(1) & (z_n != 0)
                    if ok_c.sum() < 10 or ok_n.sum() < 10:
                        continue

                    center_c = pts_w_c[ok_c].mean(0)
                    center_n = pts_w_n[ok_n].mean(0)
                    estimated_vel = (center_n - center_c) * gt_fps

                    estimated_velocity[i][mask_n] = estimated_vel

                    # GT velocity for this object
                    gt_obj_id = valid_gt_ids[obj_idx]
                    gt_obj_mask = (gt_seg_next == gt_obj_id)
                    if gt_obj_mask.sum() < 10:
                        continue
                    gt_vel = gt_vel_frame[gt_obj_mask].mean(0)

                    error = float(np.linalg.norm(estimated_vel - gt_vel))
                    object_errors[obj_idx].append(error)

            # Aggregate per-object errors
            LOST_OBJECT_PENALTY = 100.0
            object_mean_errors = []
            for obj_idx in range(num_objects):
                errors = object_errors[obj_idx]
                if errors:
                    mean_err = np.mean(errors)
                    normalized = mean_err * (num_pairs / len(errors))
                    object_mean_errors.append(normalized)
                else:
                    object_mean_errors.append(LOST_OBJECT_PENALTY)

            if not object_mean_errors:
                print("Warning: No valid object velocity errors computed")
                return None

            final_error = float(np.mean(object_mean_errors))

            if save_velocity_dir is not None:
                self._save_velocity_visualization(
                    estimated_velocity, gt_velocity_vis, save_velocity_dir, gt_fps)

            return final_error

        except Exception as e:
            print(f"Error computing velocity metric: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_velocity_visualization(self, estimated_velocity, gt_velocity, save_dir, gt_fps=24):
        """
        Save velocity visualization videos for debugging.

        Creates 3 videos (one for each velocity component: x, y, z), where each frame
        shows estimated velocity (left) and GT velocity (right) side by side using
        the same colormap normalization.

        Args:
            estimated_velocity: Estimated velocity with shape (T, H, W, 3).
            gt_velocity: Ground truth velocity with shape (T, H, W, 3).
            save_dir: Directory to save the visualization videos.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)

            T, H, W, _ = estimated_velocity.shape
            component_names = ['vx', 'vy', 'vz']

            for c_idx, c_name in enumerate(component_names):
                # Extract velocity component
                est_comp = estimated_velocity[..., c_idx]  # Shape (T, H, W)
                gt_comp = gt_velocity[..., c_idx]  # Shape (T, H, W)

                # Compute global min/max for consistent colormap across both
                global_min = min(est_comp.min(), gt_comp.min())
                global_max = max(est_comp.max(), gt_comp.max())

                # Avoid division by zero
                if global_max - global_min < 1e-6:
                    global_max = global_min + 1e-6

                # Create video writer
                # Concatenated width: estimated (W) + GT (W) = 2W
                video_path = os.path.join(save_dir, f"velocity_{c_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, gt_fps, (W * 2, H), isColor=True)

                for t in range(T):
                    # Normalize to 0-1 using global min/max
                    est_norm = (est_comp[t] - global_min) / (global_max - global_min)
                    gt_norm = (gt_comp[t] - global_min) / (global_max - global_min)

                    # Convert to uint8 for colormap
                    est_uint8 = (est_norm * 255).clip(0, 255).astype(np.uint8)
                    gt_uint8 = (gt_norm * 255).clip(0, 255).astype(np.uint8)

                    # Apply colormap (COLORMAP_JET for velocity visualization)
                    est_colored = cv2.applyColorMap(est_uint8, cv2.COLORMAP_JET)
                    gt_colored = cv2.applyColorMap(gt_uint8, cv2.COLORMAP_JET)

                    # Concatenate horizontally: estimated | GT
                    combined = np.concatenate([est_colored, gt_colored], axis=1)

                    out.write(combined)

                out.release()
                print(f"Saved velocity visualization: {video_path}")

        except Exception as e:
            print(f"Error saving velocity visualization: {e}")
            import traceback
            traceback.print_exc()

# --- Evaluator ---

class Evaluator:
    def __init__(self, metrics, video_mask_metrics=None, velocity_metric=None):
        self.metrics = metrics
        self.video_mask_metrics = video_mask_metrics or []
        self._velocity_metric = velocity_metric
        self._sam3_model = None
        self._sam3_processor = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_sam3_model(self):
        """Lazy-load the SAM3 video tracker model."""
        if self._sam3_model is None:
            print("Loading SAM3 video tracker model...")
            self._sam3_model = Sam3TrackerVideoModel.from_pretrained(
                SAM3_MODEL_REPO,
                revision=SAM3_MODEL_REVISION,
            ).to(self._device, dtype=torch.bfloat16)
            self._sam3_processor = Sam3TrackerVideoProcessor.from_pretrained(
                SAM3_MODEL_REPO,
                revision=SAM3_MODEL_REVISION,
            )
        return self._sam3_model, self._sam3_processor

    def _get_moge_model(self):
        """Lazy-load the MoGe depth estimation model."""
        if not hasattr(self, '_moge_model') or self._moge_model is None:
            print("Loading MoGe-2 depth estimation model...")
            self._moge_model = MoGeModel.from_pretrained(
                MOGE_MODEL_REPO,
                revision=MOGE_MODEL_REVISION,
            ).to(self._device)
        return self._moge_model

    @staticmethod
    def _extract_valid_objects(case_dir, is_realworld=False):
        """Extract per-object binary masks and their GT IDs from the initial
        instance segmentation.

        Simulator cases:
            Reads ``instance_segmentation_0000.npy`` + ``…_mapping_0000.json``,
            filters out background / invalid / ground / ramp objects.
        Real-world cases:
            Reads ``instance_segmentation_0000.npy`` only; background is ID 0,
            all non-zero IDs are treated as valid objects.

        Returns:
            (initial_mask_list, valid_gt_ids): parallel lists of binary masks
            ``(H, W)`` and the corresponding GT instance IDs, or ``([], [])``
            if no valid objects are found.
        """
        initial_state_dir = os.path.join(case_dir, "initial_state")
        ins_seg_path = os.path.join(initial_state_dir,
                                    "instance_segmentation_0000.npy")
        if not os.path.exists(ins_seg_path):
            return [], []

        ins_seg = np.load(ins_seg_path)  # (H, W)

        initial_mask_list = []
        valid_gt_ids = []

        if is_realworld:
            for gid in sorted(int(v) for v in np.unique(ins_seg) if int(v) != 0):
                obj_mask = (ins_seg == gid)
                if obj_mask.sum() > 0:
                    initial_mask_list.append(obj_mask)
                    valid_gt_ids.append(gid)
        else:
            mapping_path = os.path.join(
                initial_state_dir, "instance_segmentation_mapping_0000.json")
            if not os.path.exists(mapping_path):
                return [], []
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            for obj_id_str, label in mapping.items():
                if label in ["INVALID", "/World/Ground"]:
                    continue
                ll = label.lower()
                if "invalid" in ll or "ground" in ll:
                    continue
                if "Circular" in label or "InclinedPlane" in label:
                    continue
                gid = int(obj_id_str)
                obj_mask = (ins_seg == gid)
                if obj_mask.sum() > 0:
                    initial_mask_list.append(obj_mask)
                    valid_gt_ids.append(gid)

        return initial_mask_list, valid_gt_ids

    @staticmethod
    def _load_gt_frames(case_dir, skip_initial=True, gt_video_path=None):
        """Load GT RGB frames from the metadata-declared source."""
        return load_gt_frames(
            case_dir,
            gt_video_path=gt_video_path,
            skip_initial=skip_initial,
        )

    def generate_aligned_depth(self, video_path, case_dir=None,
                               sampled_video_idx=None, n_gt=0,
                               save_depth_dir=None, batch_size=32):
        """
        Generate MoGe depth for sampled pred frames and align to GT depth.

        Only the pred frames referenced by *sampled_video_idx* are processed.
        Each MoGe depth is resized to GT resolution and aligned via
        least-squares (scale + shift) against the corresponding GT depth.

        Args:
            video_path: Path to the predicted video file.
            case_dir: Case directory (contains depth/, instance_segmentation/).
            sampled_video_idx: List of length *n_gt* mapping GT physics frame
                index ``i`` → pred video frame index.
            n_gt: Number of GT physics frames.
            save_depth_dir: Directory to save aligned depth visualisations.
            batch_size: Batch size for MoGe inference.

        Returns:
            np.ndarray of shape ``(1+n_gt, gt_H, gt_W)``: index 0 is the
            condition frame, indices 1..n_gt are physics frames.
            ``None`` on failure.
        """
        try:
            assert case_dir is not None and os.path.exists(case_dir)
            assert sampled_video_idx is not None and n_gt > 0

            # Load pred video frames
            cap = cv2.VideoCapture(str(video_path))
            all_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()

            # Load GT depth & instance segmentation (npz[0] = condition frame)
            gt_depth_path = os.path.join(case_dir, "depth", "maps.npz")
            gt_maps = np.load(gt_depth_path)['maps']

            ins_seg_path = os.path.join(case_dir, "instance_segmentation", "maps.npz")
            ins_seg_maps = np.load(ins_seg_path)['maps']

            gt_h, gt_w = gt_maps.shape[1], gt_maps.shape[2]

            # Determine INVALID IDs for masking (sim only)
            invalid_ids = []
            mapping_path = os.path.join(
                case_dir, "initial_state",
                "instance_segmentation_mapping_0000.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                for k, v in mapping.items():
                    if v == "INVALID":
                        invalid_ids.append(int(k))

            # Build frame mapping: output[0] = condition, output[1..n_gt] = physics
            # Each entry is (pred_video_idx, gt_npz_idx)
            frame_pairs = [(0, 0)]  # condition frame
            for i in range(n_gt):
                frame_pairs.append((sampled_video_idx[i],
                                    min(i + 1, gt_maps.shape[0] - 1)))

            # Collect unique pred frame indices, resize to GT resolution first
            unique_pred_indices = sorted(set(p for p, _ in frame_pairs))
            frames_to_process = []
            for idx in unique_pred_indices:
                frame = all_frames[idx]
                if frame.shape[0] != gt_h or frame.shape[1] != gt_w:
                    frame = cv2.resize(frame, (gt_w, gt_h),
                                       interpolation=cv2.INTER_LANCZOS4)
                frames_to_process.append(frame)

            # MoGe depth estimation on GT-resolution frames
            moge_model = self._get_moge_model()
            input_images = []
            for frame in frames_to_process:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = torch.tensor(img / 255, dtype=torch.float32,
                                   device=self._device).permute(2, 0, 1)
                input_images.append(img)
            input_images = torch.stack(input_images)

            depths_dict = {}
            with torch.no_grad():
                for b_start in range(0, len(input_images), batch_size):
                    batch = input_images[b_start:b_start + batch_size]
                    output = moge_model.infer(batch)
                    for j, d in enumerate(output["depth"]):
                        pred_idx = unique_pred_indices[b_start + j]
                        depths_dict[pred_idx] = d.cpu().numpy()

            # Align each frame's depth (condition + physics) with GT
            aligned_depths = []
            for pred_idx, gt_npz_idx in frame_pairs:
                dpt = depths_dict[pred_idx].copy()

                try:
                    gt_depth = gt_maps[gt_npz_idx]
                    ins_seg_map = ins_seg_maps[gt_npz_idx]

                    valid_mask = np.ones_like(gt_depth, dtype=bool)
                    for iid in invalid_ids:
                        valid_mask[ins_seg_map == iid] = False

                    valid_mask &= (gt_depth != 0) & np.isfinite(gt_depth) & np.isfinite(dpt)

                    if valid_mask.sum() > 100:
                        pred_vals = dpt[valid_mask].astype(np.float64)
                        gt_vals = gt_depth[valid_mask].astype(np.float64)
                        A = np.vstack([pred_vals, np.ones_like(pred_vals)]).T
                        res = np.linalg.lstsq(A, gt_vals, rcond=None)
                        scale, shift = res[0]
                        dpt = dpt * scale + shift
                    else:
                        print(f"Depth align (pred={pred_idx}, gt_npz={gt_npz_idx}): "
                              f"Not enough valid pixels ({valid_mask.sum()})")
                except Exception as e:
                    print(f"Depth align (pred={pred_idx}, gt_npz={gt_npz_idx}) "
                          f"failed: {e}")

                aligned_depths.append(dpt)

            aligned_depths = np.array(aligned_depths)  # (1+n_gt, gt_H, gt_W)

            # Save aligned depth visualisations
            if save_depth_dir:
                os.makedirs(save_depth_dir, exist_ok=True)
                for i, depth in enumerate(aligned_depths):
                    depth_vis = depth.copy()
                    valid = np.isfinite(depth_vis) & (depth_vis != 0)
                    if valid.sum() > 0:
                        vmin, vmax = depth_vis[valid].min(), depth_vis[valid].max()
                        if vmax - vmin > 1e-6:
                            depth_vis = (depth_vis - vmin) / (vmax - vmin)
                        else:
                            depth_vis = np.zeros_like(depth_vis)
                    else:
                        depth_vis = np.zeros_like(depth_vis)

                    depth_vis = (depth_vis * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_vis,
                                                     cv2.COLORMAP_VIRIDIS)
                    cv2.imwrite(os.path.join(save_depth_dir,
                                             f"depth_{i:04d}.png"),
                                depth_colored)

            return aligned_depths

        except Exception as e:
            print(f"Error generating aligned depth for {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_keyframes(self, keyframe_dir, case_dir, gt_fps,
                           physics_duration=10.0,
                           is_realworld=False,
                           save_mask_dir=None,
                           gt_video_path=None):
        """
        Evaluate generated keyframes against GT frames.

        Keyframes are individual PNGs named ``frame_XX_tYYs.png`` where
        ``YY`` is the timestamp in seconds.  Only keyframes whose timestamp
        is ≤ ``physics_duration`` are evaluated (the condition image is NOT
        among them).

        Temporal alignment: for each keyframe at time *t* (parsed from its
        filename) the closest GT frame is selected.  No fixed keyframe FPS
        is assumed.

        Velocity is **never** evaluated for keyframes.

        Args:
            keyframe_dir: Directory containing generated keyframes
                (e.g. ``generation_frames_variant_B/``).
            case_dir: Path to the case directory (contains rgb/,
                initial_state/, …).
            gt_fps: Ground-truth frame rate declared in case metadata.
            physics_duration: Real-world physics time the keyframes
                correspond to.
            is_realworld: Whether this case uses real-world data.
            save_mask_dir: Directory to save generated masks (optional).
            gt_video_path: Complete GT video declared in case metadata, or
                ``None`` when GT RGB is a PNG sequence.

        Returns:
            dict {metric_name: score} or None on failure.
        """
        # ================================================================
        # 1. Load keyframes from directory, parsing timestamps from filenames
        #    Naming convention: frame_XX_tYYs.png  (YY = timestamp in seconds)
        # ================================================================
        keyframe_dir = Path(keyframe_dir)
        kf_files = sorted(glob.glob(str(keyframe_dir / "*.png")))

        timestamp_pattern = re.compile(r"_t([\d.]+)s\.png$")

        pred_frames = []
        kf_timestamps = []  # timestamp (seconds) for each loaded keyframe
        for kf_path in kf_files:
            m = timestamp_pattern.search(os.path.basename(kf_path))
            if m is None:
                print(f"Warning: cannot parse timestamp from {kf_path}, skipping")
                continue
            t_sec = float(m.group(1))
            if t_sec > physics_duration:
                continue  # skip keyframes beyond the physics duration
            img = cv2.imread(kf_path)
            if img is None:
                print(f"Warning: could not read keyframe {kf_path}")
                continue
            pred_frames.append(img)
            kf_timestamps.append(t_sec)

        n_keyframes = len(pred_frames)
        if n_keyframes == 0:
            print(f"Error: no valid keyframes found in {keyframe_dir}")
            return None

        # ================================================================
        # 2. Load GT – separate condition frame, use metadata fps, truncate
        #    (identical to evaluate_video §2)
        # ================================================================
        gt_all, _ = self._load_gt_frames(
            case_dir,
            skip_initial=False,
            gt_video_path=gt_video_path,
        )
        gt_fps = float(gt_fps)

        gt_physics = gt_all[1:]  # skip condition frame

        max_gt = int(round(physics_duration * gt_fps))
        n_gt_phys = min(max_gt, len(gt_physics))
        gt_physics = gt_physics[:n_gt_phys]

        # ================================================================
        # 3. Temporal alignment – select GT frames at keyframe timestamps
        #    Each keyframe has an explicit timestamp parsed from its filename.
        #    We pick the GT physics frame closest to that timestamp.
        # ================================================================
        sampled_gt = []
        sampled_gt_indices = []  # 0-based index into gt_physics
        for t in kf_timestamps:
            gt_idx = min(int(round(t * gt_fps)) - 1, n_gt_phys - 1)
            gt_idx = max(0, gt_idx)
            sampled_gt.append(gt_physics[gt_idx])
            sampled_gt_indices.append(gt_idx)

        # Resize pred frames to match GT resolution
        gt_h, gt_w = gt_physics[0].shape[:2]
        for i in range(n_keyframes):
            if pred_frames[i].shape[:2] != (gt_h, gt_w):
                pred_frames[i] = cv2.resize(
                    pred_frames[i], (gt_w, gt_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )

        # ================================================================
        # 4. SAM3 mask generation (keyframes as a pseudo-video)
        # ================================================================
        initial_mask_list, valid_gt_ids = self._extract_valid_objects(
            case_dir, is_realworld=is_realworld)

        # Build pseudo-video: condition frame + all keyframes
        video_segments = None
        pred_masks = None
        sam3_to_gt = {}
        if initial_mask_list:
            try:
                # The selected GT source always includes its condition frame
                # at index 0, for both PNG sequences and complete GT videos.
                cond_frame = gt_all[0] if gt_all else None

                if cond_frame is not None:
                    cond_rgb = cv2.cvtColor(cond_frame, cv2.COLOR_BGR2RGB)
                    video_frames_for_sam = [cond_rgb]
                    for pf in pred_frames:
                        pf_rgb = cv2.cvtColor(pf, cv2.COLOR_BGR2RGB)
                        video_frames_for_sam.append(pf_rgb)
                    video_frames_np = np.array(video_frames_for_sam)

                    model, processor = self._get_sam3_model()
                    inference_session = processor.init_video_session(
                        video=video_frames_np,
                        inference_device=self._device,
                        dtype=torch.bfloat16,
                    )

                    ann_frame_idx = 0
                    obj_ids_sam = []
                    all_objs_points = []
                    all_objs_labels = []
                    for idx, m in enumerate(initial_mask_list):
                        ys, xs = np.where(m > 0)
                        if len(ys) == 0:
                            continue
                        obj_ids_sam.append(idx)
                        all_objs_points.append([[int(np.mean(xs)), int(np.mean(ys))]])
                        all_objs_labels.append([1])

                    if obj_ids_sam:
                        processor.add_inputs_to_inference_session(
                            inference_session=inference_session,
                            frame_idx=ann_frame_idx,
                            obj_ids=obj_ids_sam,
                            input_points=[all_objs_points],
                            input_labels=[all_objs_labels],
                        )
                        model(inference_session=inference_session, frame_idx=ann_frame_idx)

                        video_segments = {}
                        for sam3_out in model.propagate_in_video_iterator(inference_session):
                            video_res_masks = processor.post_process_masks(
                                [sam3_out.pred_masks],
                                original_sizes=[[
                                    inference_session.video_height,
                                    inference_session.video_width]],
                                binarize=False,
                            )[0]
                            video_segments[sam3_out.frame_idx] = {
                                oid: video_res_masks[k]
                                for k, oid in enumerate(
                                    inference_session.obj_ids)
                            }

                        sam3_to_gt = (
                            {i: gid for i, gid in enumerate(valid_gt_ids)}
                            if valid_gt_ids else {}
                        )

                        # Build combined binary masks per frame
                        h, w = inference_session.video_height, inference_session.video_width
                        n_pseudo = len(video_frames_for_sam)
                        masks_list = []
                        for fi in range(n_pseudo):
                            combined = np.zeros((h, w), dtype=np.uint8)
                            if fi in video_segments:
                                for _, mt in video_segments[fi].items():
                                    mn = mt.cpu().float().numpy().squeeze()
                                    if mn.ndim > 2:
                                        mn = mn[0]
                                    combined = np.logical_or(
                                        combined, mn > 0).astype(np.uint8)
                            masks_list.append(combined)
                            if save_mask_dir:
                                self.save_mask_image(
                                    combined,
                                    Path(save_mask_dir) / f"keyframe_{fi:03d}_mask.png",
                                )
                        pred_masks = np.array(masks_list)
            except Exception as e:
                print(f"Error generating keyframe masks with SAM3: {e}")
                import traceback
                traceback.print_exc()

        # ================================================================
        # 5. Load per-frame GT auxiliary data
        # ================================================================
        gt_ins_seg_path = os.path.join(case_dir, "instance_segmentation", "maps.npz")
        gt_ins_seg_frames = np.load(gt_ins_seg_path)['maps']

        gt_mask_npz_path = os.path.join(case_dir, "mask", "maps.npz")
        gt_masks_npz = np.load(gt_mask_npz_path)['maps']

        # ================================================================
        # 5b. Real-world GT instance-seg quality check
        # ================================================================
        skip_masked_psnr = False
        if is_realworld and gt_ins_seg_frames is not None:
            id_sets = []
            for gt_phys_idx in sampled_gt_indices:
                npz_idx = min(gt_phys_idx + 1, gt_ins_seg_frames.shape[0] - 1)
                ids = set(int(v) for v in np.unique(gt_ins_seg_frames[npz_idx])
                          if int(v) != 0)
                id_sets.append(ids)
            if id_sets:
                reference_ids = id_sets[0]
                unstable_count = sum(
                    1 for s in id_sets[1:] if s != reference_ids)
                instability_ratio = unstable_count / max(
                    len(id_sets) - 1, 1)
                if instability_ratio > 0.1:
                    print(
                        f"Warning: Real-world GT instance seg IDs are "
                        f"unstable ({unstable_count}/{len(id_sets)-1} "
                        f"frames differ). Skipping MaskedPSNR.")
                    skip_masked_psnr = True

        # ================================================================
        # 6. Compute per-frame metrics (PSNR, MaskedPSNR)
        #    For each keyframe i, the aligned GT physics index is
        #    sampled_gt_indices[i].  NPZ files include the condition
        #    frame at index 0, so physics frame j maps to npz index
        #    (j + 1).
        # ================================================================
        metric_scores = defaultdict(list)
        for i in range(n_keyframes):
            gt_phys_idx = sampled_gt_indices[i]

            # Pseudo-video index: condition=0, keyframes=1..n_keyframes
            pseudo_vid_idx = i + 1

            # Per-object masks from SAM3
            pred_obj_masks_i = {}
            if (video_segments is not None
                    and pseudo_vid_idx in video_segments and sam3_to_gt):
                for sam3_id, mask_tensor in video_segments[pseudo_vid_idx].items():
                    gt_id = sam3_to_gt.get(sam3_id)
                    if gt_id is None:
                        continue
                    mask_np = mask_tensor.cpu().float().numpy().squeeze()
                    if mask_np.ndim > 2:
                        mask_np = mask_np[0]
                    pred_obj_masks_i[gt_id] = (mask_np > 0).astype(np.uint8)

            # Per-object GT masks from instance segmentation
            gt_obj_masks_i = {}
            if gt_ins_seg_frames is not None and valid_gt_ids:
                npz_idx = min(gt_phys_idx + 1,
                              gt_ins_seg_frames.shape[0] - 1)
                seg_frame = gt_ins_seg_frames[npz_idx]
                for gt_id in valid_gt_ids:
                    obj_mask = (seg_frame == gt_id).astype(np.uint8)
                    if obj_mask.sum() > 0:
                        gt_obj_masks_i[gt_id] = obj_mask

            for metric in self.metrics:
                if skip_masked_psnr and metric.name == "masked_psnr":
                    continue
                score = metric.compute(
                    pred_frames[i], sampled_gt[i],
                    pred_obj_masks=pred_obj_masks_i or None,
                    gt_obj_masks=gt_obj_masks_i or None,
                )
                metric_scores[metric.name].append(score)

        results = {}
        for metric in self.metrics:
            if metric.name not in metric_scores:
                continue
            scores = metric_scores[metric.name]
            results[metric.name] = metric.aggregate(scores) if scores else 0.0

        # ================================================================
        # 7. IoU metrics – sample pred_masks and gt_masks at aligned indices
        # ================================================================
        if (self.video_mask_metrics and pred_masks is not None
                and gt_masks_npz is not None):
            sampled_pred_masks = []
            sampled_gt_masks = []
            for i in range(n_keyframes):
                # pseudo-video: index 0=condition, 1..n_keyframes=keyframes
                sampled_pred_masks.append(pred_masks[i + 1])
                gt_phys_idx = sampled_gt_indices[i]
                npz_idx = min(gt_phys_idx + 1, gt_masks_npz.shape[0] - 1)
                sampled_gt_masks.append((gt_masks_npz[npz_idx] > 0).astype(np.uint8))

            pred_m = np.array(sampled_pred_masks)
            gt_m = np.array(sampled_gt_masks)
            for metric in self.video_mask_metrics:
                score = metric.compute(pred_m, gt_m)
                results[metric.name] = score

        return results

    def evaluate_video(self, video_path, case_dir, gt_fps,
                       physics_duration=10.0,
                       is_realworld=False,
                       save_mask_path=None, save_depth_dir=None,
                       save_velocity_dir=None,
                       gt_video_path=None):
        """
        Compute metrics for a generated video against GT frames.

        Both GT and pred may contain a *condition frame* (the first frame that
        depicts the initial scene).  This frame is stripped before comparison.
        The remaining "physics frames" cover ``physics_duration`` seconds of
        simulation / real-world footage.  Pred is down-sampled to GT timestamps
        for an apples-to-apples comparison.

        Args:
            video_path: Path to the predicted video (.mp4).
            case_dir: Path to the case directory (contains rgb/, initial_state/, …).
            gt_fps: Ground-truth frame rate declared in case metadata.
            physics_duration: Real-world physics time the generated video
                corresponds to.  Also used to truncate GT.
            save_mask_path: Path to save the generated mask video (optional).
            save_depth_dir: Directory to save aligned depth maps (optional).
            save_velocity_dir: Directory to save velocity visualization videos (optional).
            gt_video_path: Complete GT video declared in case metadata, or
                ``None`` when GT RGB is a PNG sequence.

        Returns:
            dict {metric_name: score} or None on failure.
        """
        if physics_duration <= 0:
            print("Error: physics_duration must be > 0")
            return None

        # ================================================================
        # 1. Load pred video – separate condition frame from physics frames
        # ================================================================
        cap_pred = cv2.VideoCapture(str(video_path))
        if not cap_pred.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        pred_src_fps = float(cap_pred.get(cv2.CAP_PROP_FPS))

        pred_all_frames = []
        while True:
            ret, frame = cap_pred.read()
            if not ret:
                break
            pred_all_frames.append(frame)
        cap_pred.release()

        if not pred_all_frames:
            print(f"Error: no frames loaded from {video_path}")
            return None

        # The first frame is the condition image, so we skip it
        pred_physics = pred_all_frames[1:]
        pred_cond_offset = 1

        n_pred_phys = len(pred_physics)

        # ================================================================
        # 2. Load GT – separate condition frame, use metadata fps, truncate
        # ================================================================
        gt_all, _ = self._load_gt_frames(
            case_dir,
            skip_initial=False,
            gt_video_path=gt_video_path,
        )
        gt_fps = float(gt_fps)

        gt_physics = gt_all[1:]

        max_gt = int(round(physics_duration * gt_fps))
        n_gt = min(max_gt, len(gt_physics))
        gt_physics = gt_physics[:n_gt]

        # ================================================================
        # 3. Temporal alignment – sample pred at GT frame timestamps
        # ================================================================
        pred_eff_fps = n_pred_phys / physics_duration

        sampled_pred = []
        sampled_video_idx = []
        for i in range(n_gt):
            t = i / gt_fps
            pidx = min(int(round(t * pred_eff_fps)), n_pred_phys - 1)
            pidx = max(0, pidx)
            sampled_pred.append(pred_physics[pidx])
            sampled_video_idx.append(pidx + pred_cond_offset)

        gt_h, gt_w = gt_physics[0].shape[:2]
        for i in range(n_gt):
            if sampled_pred[i].shape[:2] != (gt_h, gt_w):
                sampled_pred[i] = cv2.resize(
                    sampled_pred[i], (gt_w, gt_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )

        # ================================================================
        # 4. SAM3 mask generation (runs on full pred video)
        # ================================================================
        pred_masks, video_segments, valid_gt_ids = self.generate_mask(
            str(video_path), case_dir=case_dir,
            is_realworld=is_realworld,
        )

        sam3_to_gt = (
            {i: gid for i, gid in enumerate(valid_gt_ids)}
            if valid_gt_ids else {}
        )

        # ================================================================
        # 5. Load per-frame GT auxiliary data
        # ================================================================
        gt_ins_seg_frames = None
        gt_ins_seg_path = os.path.join(case_dir, "instance_segmentation", "maps.npz")
        if os.path.exists(gt_ins_seg_path):
            try:
                gt_ins_seg_frames = np.load(gt_ins_seg_path)['maps']
            except Exception as e:
                print(f"Warning: Failed to load GT instance seg: {e}")

        gt_masks = None
        gt_mask_npz = os.path.join(case_dir, "mask", "maps.npz")
        if os.path.exists(gt_mask_npz):
            try:
                gt_masks = np.load(gt_mask_npz)['maps']
            except Exception as e:
                print(f"Warning: Failed to load GT masks: {e}")

        # ================================================================
        # 5b. Real-world GT instance-seg quality check
        #     Objects never appear/disappear, so the set of non-zero IDs
        #     should be identical across all frames.  If IDs fluctuate
        #     (appear/disappear between frames), the GT seg is unreliable
        #     and MaskedPSNR is skipped.
        # ================================================================
        skip_masked_psnr = False
        if is_realworld and gt_ins_seg_frames is not None:
            id_sets = []
            for i in range(n_gt):
                npz_idx = min(i + 1, gt_ins_seg_frames.shape[0] - 1)
                ids = set(int(v) for v in np.unique(gt_ins_seg_frames[npz_idx]) if int(v) != 0)
                id_sets.append(ids)
            if id_sets:
                reference_ids = id_sets[0]
                unstable_count = sum(1 for s in id_sets[1:] if s != reference_ids)
                instability_ratio = unstable_count / max(len(id_sets) - 1, 1)
                if instability_ratio > 0.1:
                    print(f"Warning: Real-world GT instance seg IDs are unstable "
                          f"({unstable_count}/{len(id_sets)-1} frames differ from "
                          f"reference {reference_ids}). Skipping MaskedPSNR.")
                    skip_masked_psnr = True

        # ================================================================
        # 6. Compute per-frame metrics (PSNR, MaskedPSNR)
        #    Iterate over GT timestamps; i is 0-based physics frame index.
        #    NPZ files include the condition frame at index 0, so physics
        #    frame i maps to npz index (i + 1).
        # ================================================================
        metric_scores = defaultdict(list)
        for i in range(n_gt):
            vid_idx = sampled_video_idx[i]
            npz_idx = min(i + 1, gt_ins_seg_frames.shape[0] - 1)

            pred_obj_masks_i = {}
            if video_segments is not None and vid_idx in video_segments and sam3_to_gt:
                for sam3_id, mask_tensor in video_segments[vid_idx].items():
                    gt_id = sam3_to_gt.get(sam3_id)
                    if gt_id is None:
                        continue
                    mask_np = mask_tensor.cpu().float().numpy().squeeze()
                    if mask_np.ndim > 2:
                        mask_np = mask_np[0]
                    pred_obj_masks_i[gt_id] = (mask_np > 0).astype(np.uint8)

            gt_obj_masks_i = {}
            if gt_ins_seg_frames is not None and valid_gt_ids:
                seg_frame = gt_ins_seg_frames[npz_idx]
                for gt_id in valid_gt_ids:
                    obj_mask = (seg_frame == gt_id).astype(np.uint8)
                    if obj_mask.sum() > 0:
                        gt_obj_masks_i[gt_id] = obj_mask

            for metric in self.metrics:
                if skip_masked_psnr and metric.name == "masked_psnr":
                    continue
                score = metric.compute(
                    sampled_pred[i], gt_physics[i],
                    pred_obj_masks=pred_obj_masks_i or None,
                    gt_obj_masks=gt_obj_masks_i or None,
                )
                metric_scores[metric.name].append(score)

        results = {}
        for metric in self.metrics:
            if metric.name not in metric_scores:
                continue
            scores = metric_scores[metric.name]
            results[metric.name] = metric.aggregate(scores) if scores else 0.0

        # ================================================================
        # 7. Save mask video
        # ================================================================
        if save_mask_path and pred_masks is not None:
            self.save_mask_video(pred_masks, save_mask_path, fps=pred_src_fps)

            if video_segments:
                all_obj_ids = set()
                for fidx in video_segments:
                    all_obj_ids.update(video_segments[fidx].keys())

                save_path_obj = Path(save_mask_path)
                parent_dir = save_path_obj.parent
                stem = save_path_obj.stem
                suffix = save_path_obj.suffix
                h, w = pred_masks.shape[1], pred_masks.shape[2]
                n_vid_frames = len(pred_all_frames)

                for obj_id in sorted(all_obj_ids):
                    obj_masks = []
                    for fi in range(n_vid_frames):
                        mask_frame = np.zeros((h, w), dtype=np.uint8)
                        if fi in video_segments and obj_id in video_segments[fi]:
                            mask_tensor = video_segments[fi][obj_id]
                            if hasattr(mask_tensor, 'cpu'):
                                mask_np = mask_tensor.cpu().float().numpy().squeeze()
                            else:
                                mask_np = mask_tensor
                            if mask_np.ndim > 2:
                                mask_np = mask_np[0]
                            mask_frame = (mask_np > 0).astype(np.uint8)
                        obj_masks.append(mask_frame)
                    obj_masks = np.array(obj_masks)
                    if obj_masks.sum() > 0:
                        self.save_mask_video(
                            obj_masks,
                            parent_dir / f"{stem}_obj{obj_id}{suffix}",
                            fps=pred_src_fps,
                        )

        # ================================================================
        # 8. IoU comparison – sample pred_masks and gt_masks at aligned indices
        # ================================================================
        if self.video_mask_metrics and pred_masks is not None:
            sampled_pred_masks = []
            sampled_gt_masks = []
            for i in range(n_gt):
                vid_idx = sampled_video_idx[i]
                sampled_pred_masks.append(pred_masks[vid_idx])
                npz_idx = min(i + 1, gt_masks.shape[0] - 1)
                sampled_gt_masks.append((gt_masks[npz_idx] > 0).astype(np.uint8))

            pred_m = np.array(sampled_pred_masks)
            gt_m = np.array(sampled_gt_masks)
            for metric in self.video_mask_metrics:
                score = metric.compute(pred_m, gt_m)
                results[metric.name] = score

        # ================================================================
        # 9. Velocity metric (uses full video internally)
        # ================================================================
        if (self._velocity_metric is not None and video_segments is not None):
            try:
                aligned_depths = self.generate_aligned_depth(
                    video_path, case_dir=case_dir,
                    sampled_video_idx=sampled_video_idx,
                    n_gt=n_gt,
                    save_depth_dir=save_depth_dir,
                )
                if aligned_depths is not None:
                    gt_velocity_path = os.path.join(case_dir, "velocity", "maps.npz")
                    cam_params_dir = os.path.join(case_dir, "camera_parameters")
                    gt_ins_seg_path_vel = os.path.join(
                        case_dir, "instance_segmentation", "maps.npz")

                    obj_ids = list(video_segments[0].keys())

                    if obj_ids:
                        velocity_error = self._velocity_metric.compute_velocity_error(
                            aligned_depths=aligned_depths,
                            video_segments=video_segments,
                            sampled_video_idx=sampled_video_idx,
                            n_gt=n_gt,
                            valid_gt_ids=valid_gt_ids,
                            gt_velocity_path=gt_velocity_path,
                            cam_params_dir=cam_params_dir,
                            obj_ids=obj_ids,
                            gt_ins_seg_path=gt_ins_seg_path_vel,
                            save_velocity_dir=save_velocity_dir,
                            gt_fps=gt_fps,
                        )
                        if velocity_error is not None:
                            results[self._velocity_metric.name] = velocity_error

            except Exception as e:
                print(f"Error computing velocity metric: {e}")
                import traceback
                traceback.print_exc()

        return results

    def save_mask_video(self, masks, output_path, fps=24):
        """Save a sequence of masks as a video."""
        if len(masks) == 0:
             return
        h, w = masks[0].shape
        # Use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), isColor=False)
        for mask in masks:
            # Ensure mask is 0-255 uint8
            if mask.max() <= 1:
                frame = (mask * 255).astype(np.uint8)
            else:
                frame = mask.astype(np.uint8)
            out.write(frame)
        out.release()

    def save_mask_image(self, mask, output_path):
        """Save a single mask frame."""
        # Ensure mask is 0-255 uint8
        if mask.max() <= 1:
            img = (mask * 255).astype(np.uint8)
        else:
            img = mask.astype(np.uint8)
        cv2.imwrite(str(output_path), img)

    def generate_mask(self, in_path: str, case_dir: str = None,
                      is_realworld: bool = False):
        """
        Generate binary masks for a video using SAM3 video tracker.

        Args:
            in_path: Path to the input video file.
            case_dir: Path to the case directory.
            is_realworld: If True, treat as real-world data (no mapping json,
                background=0, all non-zero IDs are valid objects).

        Returns:
            (masks, video_segments, valid_gt_ids) where *masks* has shape
            ``(T+1, H, W)`` (initial frame + T video frames) with values 0/1,
            *valid_gt_ids* is a list mapping SAM3 obj index to the original GT
            instance-segmentation ID, or ``(None, None, None)`` on failure.
        """
        try:
            assert case_dir is not None, "case_dir not provided"

            initial_mask_list, valid_gt_ids = self._extract_valid_objects(
                case_dir, is_realworld=is_realworld)

            if not initial_mask_list:
                print(f"Warning: No valid objects found in {case_dir}")
                return None, None, None

            # Load video frames; match GT resolution (same as initial instance seg)
            video_frames, _ = load_video(in_path)  # shape: [T, H, W, 3]
            gt_h, gt_w = initial_mask_list[0].shape[:2]
            if video_frames.shape[1:3] != (gt_h, gt_w):
                video_frames = np.stack([
                    cv2.resize(video_frames[ti], (gt_w, gt_h), interpolation=cv2.INTER_LANCZOS4)
                    for ti in range(video_frames.shape[0])
                ])

            model, processor = self._get_sam3_model()

            inference_session = processor.init_video_session(
                video=video_frames,
                inference_device=self._device,
                dtype=torch.bfloat16,
            )

            # Prepare prompts (center points of each object)
            ann_frame_idx = 0
            obj_ids = []
            all_objs_points = []
            all_objs_labels = []

            for idx, mask in enumerate(initial_mask_list):
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) == 0:
                    continue
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                obj_ids.append(idx)
                all_objs_points.append([[center_x, center_y]])
                all_objs_labels.append([1])

            if not obj_ids:
                print(f"Warning: No valid object centres found in {case_dir}")
                return None, None, None

            input_points = [all_objs_points]
            input_labels = [all_objs_labels]

            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=ann_frame_idx,
                obj_ids=obj_ids,
                input_points=input_points,
                input_labels=input_labels,
            )

            outputs = model(
                inference_session=inference_session,
                frame_idx=ann_frame_idx,
            )

            # Propagate through video
            video_segments = {}
            for sam3_out in model.propagate_in_video_iterator(inference_session):
                video_res_masks = processor.post_process_masks(
                    [sam3_out.pred_masks],
                    original_sizes=[[inference_session.video_height, inference_session.video_width]],
                    binarize=False,
                )[0]
                video_segments[sam3_out.frame_idx] = {
                    oid: video_res_masks[i]
                    for i, oid in enumerate(inference_session.obj_ids)
                }

            # Combine per-object masks into a single binary mask per frame
            num_frames = len(video_frames)
            h, w = inference_session.video_height, inference_session.video_width
            masks = []
            for i in range(num_frames):
                combined = np.zeros((h, w), dtype=np.uint8)
                if i in video_segments:
                    for _oid, mask_tensor in video_segments[i].items():
                        mask_np = mask_tensor.cpu().float().numpy().squeeze()
                        if mask_np.ndim > 2:
                            mask_np = mask_np[0]
                        combined = np.logical_or(combined, mask_np > 0).astype(np.uint8)
                masks.append(combined)

            result = np.array(masks)
            expected_shape = (num_frames, h, w)
            assert result.shape == expected_shape, (
                f"Mask shape mismatch: got {result.shape}, expected {expected_shape}"
            )
            return result, video_segments, valid_gt_ids

        except Exception as e:
            print(f"Error processing video {in_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
