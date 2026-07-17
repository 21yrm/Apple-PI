"""SAM3-based segmentation IoU evaluator for PERCEPTION_GRAPHIC subtrack.

Computes IoU between SAM3-predicted object masks on the generated image
and ground-truth object masks, using GT mask centroids as SAM3 query points.

Supports two data sources:
  - Simulator: uses instance_segmentation_mapping JSON + mask files in initial_state/
  - Realworld: uses instance_segmentation_0000.npy (non-zero = object)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from apple_pi.constants import SAM3_MODEL_REPO, SAM3_MODEL_REVISION

logger = logging.getLogger(__name__)

_sam3_model = None
_sam3_processor = None


def _load_sam3(device: torch.device):
    """Lazy-load SAM3 model (singleton)."""
    global _sam3_model, _sam3_processor
    if _sam3_model is not None:
        return _sam3_model, _sam3_processor

    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    logger.info("Loading SAM3 model ...")
    _sam3_model = Sam3TrackerVideoModel.from_pretrained(
        SAM3_MODEL_REPO,
        revision=SAM3_MODEL_REVISION,
    ).to(device, dtype=torch.bfloat16)
    _sam3_processor = Sam3TrackerVideoProcessor.from_pretrained(
        SAM3_MODEL_REPO,
        revision=SAM3_MODEL_REVISION,
    )
    logger.info("SAM3 model loaded.")
    return _sam3_model, _sam3_processor


def _mask_centroid(binary_mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (x, y) centroid of a binary mask, or None if empty."""
    ys, xs = np.where(binary_mask > 0)
    if len(ys) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))


def _get_query_points_simulator(initial_state_dir: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Extract query points and GT combined mask from simulator data.

    Returns:
        query_points: list of (x, y) centroids for each valid object
        gt_mask: binary (H, W) mask combining all valid objects
    """
    d = Path(initial_state_dir)

    mapping_path = d / "instance_segmentation_mapping_0000.json"
    ins_seg_path = d / "instance_segmentation_0000.npy"
    mask_path = d / "mask_0000.npy"

    with open(mapping_path) as f:
        mapping = json.load(f)

    ins_seg = np.load(str(ins_seg_path))  # (H, W) uint16
    gt_mask = np.load(str(mask_path))  # (H, W) uint8, 0/1

    valid_ids = []
    for str_id, label in mapping.items():
        int_id = int(str_id)
        label_lower = label.lower()
        if "invalid" in label_lower or "ground" in label_lower:
            continue
        valid_ids.append(int_id)

    query_points = []
    for obj_id in valid_ids:
        obj_mask = (ins_seg == obj_id)
        centroid = _mask_centroid(obj_mask)
        if centroid is not None:
            query_points.append(centroid)

    return query_points, gt_mask


def _get_query_points_realworld(first_frame_mask_path: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Extract query points and GT combined mask from realworld data.

    Returns:
        query_points: list of (x, y) centroids for each unique non-zero object
        gt_mask: binary (H, W) mask combining all non-zero objects
    """
    maps = np.load(first_frame_mask_path)

    gt_mask = (maps > 0).astype(np.uint8)

    obj_ids = [v for v in np.unique(maps) if v != 0]
    query_points = []
    for obj_id in obj_ids:
        centroid = _mask_centroid(maps == obj_id)
        if centroid is not None:
            query_points.append(centroid)

    return query_points, gt_mask


def _run_sam3_single_image(
    model,
    processor,
    image_pil: Image.Image,
    query_points: List[Tuple[int, int]],
    device: torch.device,
) -> np.ndarray:
    """Run SAM3 on a single image with point prompts, return combined binary mask.

    SAM3 expects a video (list of PIL images). We pass a single-frame "video".
    Each query point is treated as a separate object with a positive label.
    """
    frames_pil = [image_pil]

    obj_ids = list(range(len(query_points)))
    formatted_points = [[[pt[0], pt[1]] for pt in [qp]] for qp in query_points]
    input_points = [formatted_points]
    input_labels = [[[1] for _ in query_points]]

    inference_session = processor.init_video_session(
        video=frames_pil, inference_device=device, dtype=torch.bfloat16,
    )
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=obj_ids,
        input_points=input_points,
        input_labels=input_labels,
    )
    _ = model(inference_session=inference_session, frame_idx=0)

    h, w = inference_session.video_height, inference_session.video_width
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for out in model.propagate_in_video_iterator(inference_session):
        masks = processor.post_process_masks(
            [out.pred_masks],
            original_sizes=[[h, w]],
            binarize=False,
        )[0]
        for i, oid in enumerate(inference_session.obj_ids):
            mask_np = masks[i].cpu().float().numpy().squeeze()
            if mask_np.ndim > 2:
                mask_np = mask_np[0]
            combined_mask[mask_np > 0] = 1

    return combined_mask


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred_bin = pred_mask.astype(bool)
    gt_bin = gt_mask.astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _get_query_points_velocity_simulator(velocity_dir: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Extract query points and GT combined mask from instantaneous_velocity/ for sim cases.

    Uses mapping.json to filter out INVALID and ground objects from mask.npy.

    Returns:
        query_points: list of (x, y) centroids for each valid object
        gt_mask: binary (H, W) mask combining all valid objects
    """
    d = Path(velocity_dir)

    mapping_path = d / "mapping.json"
    mask_path = d / "mask.npy"

    with open(mapping_path) as f:
        mapping = json.load(f)

    mask = np.load(str(mask_path))  # (H, W) with instance IDs

    valid_ids = []
    for str_id, label in mapping.items():
        int_id = int(str_id)
        label_lower = label.lower()
        if "invalid" in label_lower or "ground" in label_lower:
            continue
        valid_ids.append(int_id)

    gt_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    query_points = []
    for obj_id in valid_ids:
        obj_mask = (mask == obj_id)
        gt_mask[obj_mask] = 1
        centroid = _mask_centroid(obj_mask)
        if centroid is not None:
            query_points.append(centroid)

    return query_points, gt_mask


def _get_query_points_velocity_realworld(velocity_dir: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Extract query points and GT combined mask from instantaneous_velocity/ for real-world cases.

    Uses mask.npy where non-zero IDs are target objects.

    Returns:
        query_points: list of (x, y) centroids for each unique non-zero object
        gt_mask: binary (H, W) mask combining all non-zero objects
    """
    mask_path = Path(velocity_dir) / "mask.npy"
    maps = np.load(str(mask_path))

    gt_mask = (maps > 0).astype(np.uint8)

    obj_ids = [v for v in np.unique(maps) if v != 0]
    query_points = []
    for obj_id in obj_ids:
        centroid = _mask_centroid(maps == obj_id)
        if centroid is not None:
            query_points.append(centroid)

    return query_points, gt_mask


def compute_segmentation_iou(
    gen_image: np.ndarray,
    case_dir: str,
    is_realworld: bool = False,
    mask_subdir: str = "initial_state",
) -> float:
    """Compute SAM3-based segmentation IoU for a generated image.

    Args:
        gen_image: generated image as numpy array (H, W, 3), RGB uint8.
        case_dir: path to the case directory.
        is_realworld: if True, use realworld mask format; else simulator.
        mask_subdir: subdirectory containing masks. "initial_state" for
            perception_graphic, "instantaneous_velocity" for formulation_graphic.

    Returns:
        IoU score (0.0 to 1.0).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_sam3(device)

    if mask_subdir == "instantaneous_velocity":
        velocity_dir = str(Path(case_dir) / "instantaneous_velocity")
        if is_realworld:
            query_points, gt_mask = _get_query_points_velocity_realworld(velocity_dir)
        else:
            query_points, gt_mask = _get_query_points_velocity_simulator(velocity_dir)
    elif is_realworld:
        mask_path = str(Path(case_dir) / "initial_state" / "instance_segmentation_0000.npy")
        query_points, gt_mask = _get_query_points_realworld(mask_path)
    else:
        initial_state_dir = str(Path(case_dir) / "initial_state")
        query_points, gt_mask = _get_query_points_simulator(initial_state_dir)

    if not query_points:
        logger.warning("No valid query points found, returning IoU=0.0")
        return 0.0

    logger.info(f"SAM3 segmentation: {len(query_points)} query points")

    gen_pil = Image.fromarray(gen_image)
    if gen_pil.size != (gt_mask.shape[1], gt_mask.shape[0]):
        gen_pil = gen_pil.resize(
            (gt_mask.shape[1], gt_mask.shape[0]), Image.BILINEAR
        )

    pred_mask = _run_sam3_single_image(model, processor, gen_pil, query_points, device)

    iou = _compute_iou(pred_mask, gt_mask)
    logger.info(f"SAM3 segmentation IoU: {iou:.4f}")
    return iou


def visualize_segmentation_debug(
    gen_image: np.ndarray,
    case_dir: str,
    is_realworld: bool,
    save_path: str,
    mask_subdir: str = "initial_state",
) -> None:
    """可视化 SAM3 分割调试信息并保存成图片.

    可视化内容（水平拼接三列）：
      1. 生成图片 + query 点位置标注
      2. SAM3 分割结果 mask
      3. GT mask

    Args:
        gen_image: 生成图片 (H, W, 3), RGB uint8
        case_dir: case 目录
        is_realworld: True 表示 realworld 格式，否则 simulator
        save_path: 输出可视化 PNG 路径
        mask_subdir: "initial_state" 或 "instantaneous_velocity"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_sam3(device)

    if mask_subdir == "instantaneous_velocity":
        velocity_dir = str(Path(case_dir) / "instantaneous_velocity")
        if is_realworld:
            query_points, gt_mask = _get_query_points_velocity_realworld(velocity_dir)
        else:
            query_points, gt_mask = _get_query_points_velocity_simulator(velocity_dir)
    elif is_realworld:
        mask_path = str(Path(case_dir) / "initial_state" / "instance_segmentation_0000.npy")
        query_points, gt_mask = _get_query_points_realworld(mask_path)
    else:
        initial_state_dir = str(Path(case_dir) / "initial_state")
        query_points, gt_mask = _get_query_points_simulator(initial_state_dir)

    if not query_points:
        logger.warning("No valid query points found, skip visualization.")
        return

    gen_pil = Image.fromarray(gen_image)
    if gen_pil.size != (gt_mask.shape[1], gt_mask.shape[0]):
        gen_pil = gen_pil.resize(
            (gt_mask.shape[1], gt_mask.shape[0]), Image.BILINEAR
        )

    # 1) 生成图 + query 点
    vis_img = gen_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(vis_img)
    for x, y in query_points:
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)

    # 2) SAM3 预测 mask
    pred_mask = _run_sam3_single_image(model, processor, vis_img, query_points, device)
    h, w = gt_mask.shape

    def _mask_to_rgb(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[mask > 0] = np.array(color, dtype=np.uint8)
        return base

    pred_rgb = _mask_to_rgb(pred_mask, (255, 0, 0))  # red
    gt_rgb = _mask_to_rgb(gt_mask, (0, 255, 0))      # green

    vis_np = np.asarray(vis_img)
    panel = np.concatenate([vis_np, pred_rgb, gt_rgb], axis=1)

    out = Image.fromarray(panel)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(save_path)
    logger.info(f"Saved SAM3 debug visualization to: {save_path}")
