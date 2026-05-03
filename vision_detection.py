import os

import cv2
import numpy as np
import scipy.io as scio

from action_planner import normalize_grasp_angle
from experiment_config import GRASP_WIDTH


COLOR_BGR_MAP = {
    "红色": np.array([51, 51, 217], dtype=np.uint8),
    "绿色": np.array([64, 191, 64], dtype=np.uint8),
    "蓝色": np.array([217, 89, 51], dtype=np.uint8),
    "黄色": np.array([51, 204, 242], dtype=np.uint8),
    "紫色": np.array([204, 76, 166], dtype=np.uint8),
}


def analyze_colored_object_contour(image_path, target_color):
    if not image_path or not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    target_bgr = COLOR_BGR_MAP.get(target_color)
    if target_bgr is None:
        return None

    color_diff = np.linalg.norm(
        image.astype(np.int16) - target_bgr.reshape(1, 1, 3).astype(np.int16),
        axis=2,
    )
    mask = (color_diff < 95).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 80:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = max(w * h, 1)
    fill_ratio = area / rect_area
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    return {
        "fill_ratio": fill_ratio,
        "vertices": len(approx),
        "bbox": (x, y, w, h),
        "area": area,
    }


def infer_triangle_grasp_angle_from_side_views(image_dir, target_color, base_angle):
    candidate_direct = normalize_grasp_angle(base_angle)
    candidate_perpendicular = normalize_grasp_angle(base_angle + np.pi / 2)

    analyses = []
    for name in ["camera_rgb_left.png", "camera_rgb_right.png"]:
        info = analyze_colored_object_contour(os.path.join(image_dir, name), target_color)
        if info:
            analyses.append((name, info))

    if not analyses:
        return candidate_direct, "侧视图未稳定识别到目标颜色，保留顶视图抓取方向作为默认方向"

    triangle_like_detected = any(
        info["fill_ratio"] < 0.72 or info["vertices"] <= 4
        for _, info in analyses
    )

    if triangle_like_detected:
        reason = "侧视轮廓更像三角面，优先尝试与当前顶视方向垂直的夹取方向"
        return candidate_perpendicular, reason

    reason = "侧视轮廓更接近平整侧面，保持当前顶视抓取方向"
    return candidate_direct, reason


def get_positions(path, object_ids=None, max_grasp_width=GRASP_WIDTH):
    mask_file = os.path.join(path, "camera_mask.mat")
    if not os.path.exists(mask_file) or not object_ids:
        return {}

    mask_data = scio.loadmat(mask_file).get("A")
    if mask_data is None:
        return {}

    raw_mask = np.asarray(mask_data, dtype=np.int64)
    height, width = raw_mask.shape[:2]
    scale_x = 0.4 / (width / 2)
    scale_y = scale_x
    object_uid_mask = np.bitwise_and(raw_mask, (1 << 24) - 1)

    positions = {}
    debug_mask = np.zeros((height, width), dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    min_area = 80
    max_area = height * width * 0.12

    for obj_index, obj_id in enumerate(object_ids):
        binary = ((raw_mask == obj_id) | (object_uid_mask == obj_id)).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Mask 未检测到物块{obj_index + 1}，等待重新渲染。")
            continue

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            print(f"Mask 过滤物块{obj_index + 1}: 轮廓面积异常 area={area:.1f}")
            continue

        rect = cv2.minAreaRect(contour)
        (x_img, y_img), (w_px, h_px), angle_deg = rect
        if w_px <= 1 or h_px <= 1:
            print(f"Mask 过滤物块{obj_index + 1}: 外接矩形异常 w={w_px:.1f}, h={h_px:.1f}")
            continue

        x_sim = (x_img - width / 2) * scale_x
        y_sim = (height / 2 - y_img) * scale_y
        grasp_width = np.clip(max(w_px, h_px) * scale_x, 0.03, max_grasp_width)
        grasp_angle = np.deg2rad(angle_deg)
        if w_px < h_px:
            grasp_angle += np.pi / 2
        grasp_angle = normalize_grasp_angle(grasp_angle)

        positions[obj_index] = (x_sim, y_sim, 0.02, grasp_angle, grasp_width)
        debug_mask[binary > 0] = min(255, 40 + obj_index * 40)

    cv2.imwrite(os.path.join(path, "camera_mask_objects.png"), debug_mask)
    return positions
