"""
core.py - Flooring Visualizer Core Logic

This module provides functions for:
- Loading the SegFormer model (ADE20K trained from Hugging Face)
- Segmenting the floor from an image automatically
- Applying perspective transform to a texture
- Blending the texture with original lighting
"""

import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from typing import Tuple, Optional

# ADE20K class indices after reduce_labels (0-indexed from original 1-150)
# Floor = original index 4, after reduce_labels = 3
# Also include: rug/carpet = 28, mat = 29 (commonly on floors)
FLOOR_CLASS_IDS = [3]  # floor/flooring

# Global model cache
_model = None
_processor = None
_device = None


def load_model(device: Optional[str] = None):
    """
    Load the SegFormer model with ADE20K weights from Hugging Face.
    Uses nvidia/segformer-b5-finetuned-ade-640-640 for best accuracy.

    Args:
        device: Device to run on ('cuda' or 'cpu'). Auto-detects if None.

    Returns:
        Tuple of (model, processor, device).
    """
    global _model, _processor, _device
    
    if _model is not None:
        return _model, _processor, _device
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SegFormer model on {device}...")
    
    # Use the largest SegFormer variant for best accuracy
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    
    _processor = SegformerImageProcessor.from_pretrained(model_name)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    _model.to(device)
    _model.eval()
    _device = device
    
    print("SegFormer model loaded successfully!")
    return _model, _processor, _device


def get_floor_mask(
    image: np.ndarray,
    model=None,
    processor=None,
    device: str = None
) -> np.ndarray:
    """
    Segment the floor from an image using SegFormer.
    
    Uses ADE20K floor class (index 3) for precise floor detection.

    Args:
        image: RGB image as numpy array (H, W, 3).
        model: Loaded SegFormer model (optional, uses cached if None).
        processor: SegFormer processor (optional, uses cached if None).
        device: Device string (optional, uses cached if None).

    Returns:
        Binary mask (H, W) where 1 indicates floor.
    """
    global _model, _processor, _device
    
    # Use cached model if not provided
    if model is None:
        if _model is None:
            load_model()
        model, processor, device = _model, _processor, _device
    
    h, w = image.shape[:2]
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, num_classes, h/4, w/4)
    
    # Upsample to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )
    
    # Get class predictions
    predictions = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # Create floor mask from floor class IDs
    mask = np.zeros((h, w), dtype=np.uint8)
    for class_id in FLOOR_CLASS_IDS:
        mask = np.logical_or(mask, predictions == class_id).astype(np.uint8)
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest connected component (main floor area)
    mask = _keep_largest_component(mask)
    
    # If mask is too small, try expanding to nearby similar regions
    if mask.sum() < (h * w * 0.05):  # Less than 5% of image
        mask = _expand_floor_mask(image, mask, predictions)
    
    return mask


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in the mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask
    
    # Find largest component (excluding background = label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    return (labels == largest_label).astype(np.uint8)


def _expand_floor_mask(image: np.ndarray, mask: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """
    Expand a small floor mask by including nearby regions with similar appearance.
    """
    h, w = image.shape[:2]
    
    # If we have some floor pixels, use their color as reference
    if mask.sum() > 0:
        floor_pixels = image[mask > 0]
        reference_color = np.median(floor_pixels, axis=0)
    else:
        # Fallback: sample from bottom-center
        sample_region = image[int(h * 0.8):int(h * 0.95), int(w * 0.3):int(w * 0.7)]
        reference_color = np.median(sample_region.reshape(-1, 3), axis=0)
    
    # Convert to LAB for perceptual color distance
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(np.uint8([[reference_color]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    
    # Calculate color distance
    diff = np.sqrt(np.sum((image_lab - ref_lab) ** 2, axis=2))
    
    # Find bottom region class (most likely floor)
    bottom_region = predictions[int(h * 0.7):, :]
    unique, counts = np.unique(bottom_region, return_counts=True)
    if len(unique) > 0:
        dominant_class = unique[np.argmax(counts)]
        class_mask = (predictions == dominant_class).astype(np.uint8)
        
        # Combine with color similarity
        color_threshold = np.percentile(diff, 25)
        color_mask = (diff < color_threshold).astype(np.uint8)
        
        # Focus on bottom half
        color_mask[:int(h * 0.3), :] = 0
        
        # Combine masks
        combined = np.logical_or(class_mask, color_mask).astype(np.uint8)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return _keep_largest_component(combined)
    
    return mask


def apply_perspective_transform(
    texture_img: np.ndarray,
    target_shape: Tuple[int, int],
    mask: np.ndarray
) -> np.ndarray:
    """
    Warp a texture image to fit the perspective of the floor mask.

    Args:
        texture_img: RGB texture image (H, W, 3).
        target_shape: (height, width) of the target room image.
        mask: Binary floor mask (H, W).

    Returns:
        Warped texture image (H, W, 3) with same shape as target.
    """
    h, w = target_shape

    # Find contours of the floor mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # No floor found, return tiled texture
        return tile_texture(texture_img, (h, w))

    # Get the largest contour (main floor area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate to polygon (4 corners for perspective)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Get bounding box as fallback
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Try to get 4 corners for proper perspective
    if len(approx) >= 4:
        # Use convex hull and get 4 extreme points
        hull = cv2.convexHull(largest_contour)
        corners = get_four_corners(hull)
    else:
        # Fallback to bounding rectangle corners
        corners = np.array([
            [x, y],
            [x + bw, y],
            [x + bw, y + bh],
            [x, y + bh]
        ], dtype=np.float32)

    # Sort corners: top-left, top-right, bottom-right, bottom-left
    corners = order_points(corners)

    # Source points (texture corners)
    th, tw = texture_img.shape[:2]
    src_pts = np.array([
        [0, 0],
        [tw, 0],
        [tw, th],
        [0, th]
    ], dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, corners)

    if H is None:
        return tile_texture(texture_img, (h, w))

    # Tile texture to ensure enough coverage
    tiled = tile_texture(texture_img, (h * 2, w * 2))

    # Warp the texture
    warped = cv2.warpPerspective(tiled, H, (w, h))

    return warped


def get_four_corners(hull: np.ndarray) -> np.ndarray:
    """
    Extract 4 corner points from a convex hull.
    """
    hull = hull.reshape(-1, 2)

    # Find extreme points
    top_left_idx = np.argmin(hull[:, 0] + hull[:, 1])
    top_right_idx = np.argmax(hull[:, 0] - hull[:, 1])
    bottom_right_idx = np.argmax(hull[:, 0] + hull[:, 1])
    bottom_left_idx = np.argmin(hull[:, 0] - hull[:, 1])

    corners = np.array([
        hull[top_left_idx],
        hull[top_right_idx],
        hull[bottom_right_idx],
        hull[bottom_left_idx]
    ], dtype=np.float32)

    return corners


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def tile_texture(texture: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Tile a texture to fill the target size.

    Args:
        texture: Source texture image.
        target_size: (height, width) to fill.

    Returns:
        Tiled texture image.
    """
    th, tw = texture.shape[:2]
    target_h, target_w = target_size

    # Calculate how many tiles needed
    tiles_y = (target_h // th) + 2
    tiles_x = (target_w // tw) + 2

    # Tile the texture
    tiled = np.tile(texture, (tiles_y, tiles_x, 1))

    # Crop to target size
    return tiled[:target_h, :target_w]


def blend_with_lighting(
    original_room: np.ndarray,
    warped_texture: np.ndarray,
    mask: np.ndarray,
    blend_strength: float = 0.7
) -> np.ndarray:
    """
    Blend the warped texture with the original room, preserving lighting.

    Uses LAB color space to preserve luminance from the original floor.

    Args:
        original_room: Original RGB room image (H, W, 3).
        warped_texture: Warped texture RGB (H, W, 3).
        mask: Binary floor mask (H, W).
        blend_strength: How much of the original lighting to preserve (0-1).

    Returns:
        Blended result RGB image (H, W, 3).
    """
    # Convert to LAB color space
    room_lab = cv2.cvtColor(original_room, cv2.COLOR_RGB2LAB).astype(np.float32)
    texture_lab = cv2.cvtColor(warped_texture, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Create blended LAB
    blended_lab = texture_lab.copy()

    # Blend L channel (luminance) to preserve original lighting
    original_L = room_lab[:, :, 0]
    texture_L = texture_lab[:, :, 0]
    blended_L = blend_strength * original_L + (1 - blend_strength) * texture_L
    blended_lab[:, :, 0] = blended_L

    # Convert back to RGB
    blended_rgb = cv2.cvtColor(blended_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Apply mask - only replace floor area
    mask_3ch = np.stack([mask] * 3, axis=-1)
    result = np.where(mask_3ch, blended_rgb, original_room)

    # Optional: Feather edges for smoother blending
    result = feather_edges(result, original_room, mask)

    return result


def feather_edges(
    composite: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    feather_amount: int = 5
) -> np.ndarray:
    """
    Feather the edges of the mask for smoother blending.
    """
    # Create a blurred mask for feathering
    kernel = np.ones((feather_amount, feather_amount), np.float32)
    blurred_mask = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    blurred_mask = np.clip(blurred_mask, 0, 1)

    # Apply feathered blend
    blurred_mask_3ch = np.stack([blurred_mask] * 3, axis=-1)
    result = (blurred_mask_3ch * composite +
              (1 - blurred_mask_3ch) * original).astype(np.uint8)

    return result


def process_room(
    room_image: np.ndarray,
    texture_image: np.ndarray,
    model=None,
    processor=None,
    device: str = None,
    blend_strength: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: segment floor, warp texture, blend.

    Args:
        room_image: RGB room image.
        texture_image: RGB texture image.
        model: Loaded SegFormer model (optional).
        processor: SegFormer processor (optional).
        device: Device string (optional).
        blend_strength: Lighting preservation strength.

    Returns:
        Tuple of (result_image, floor_mask).
    """
    # Step 1: Segment floor automatically
    mask = get_floor_mask(room_image, model, processor, device)

    # Step 2: Warp texture to floor perspective
    warped = apply_perspective_transform(
        texture_image,
        room_image.shape[:2],
        mask
    )

    # Step 3: Blend with lighting preservation
    result = blend_with_lighting(room_image, warped, mask, blend_strength)

    return result, mask
