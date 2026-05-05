"""
视觉感知模块批量测试脚本

连续测试多次并汇总统计结果

用法:
    python test_vision_batch.py
    python test_vision_batch.py --trials 10
    python test_vision_batch.py --trials 20 --no-gui
"""

import os
import sys
import argparse
import json
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
from datetime import datetime
import cv2
import scipy.io as scio

COLOR_NAMES = ["红色", "绿色", "蓝色", "黄色", "紫色"]
COLOR_RGBA_MAP = {
    "红色": [0.8, 0.2, 0.2, 1.0],
    "绿色": [0.2, 0.75, 0.25, 1.0],
    "蓝色": [0.2, 0.35, 0.85, 1.0],
    "黄色": [0.95, 0.8, 0.2, 1.0],
    "紫色": [0.8, 0.3, 0.65, 1.0],
}

IMAGEWIDTH = 640
IMAGEHEIGHT = 480


class VisionTestEnv:
    """视觉测试仿真环境"""
    
    def __init__(self, gui=True):
        self.gui = gui
        self.connection_mode = p.GUI if gui else p.DIRECT
        
        try:
            p.getConnectionInfo()
            p.disconnect()
        except:
            pass
        
        p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.object_ids = []
        self.object_info = []
        
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 0.5],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0]
        )
        fov = 60
        aspect = IMAGEWIDTH / IMAGEHEIGHT
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=0.1,
            farVal=100
        )
        
        self.output_dir = "test_vision_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_cube(self, size=0.045, position=[0, 0, 0], color_name="红色"):
        half_size = size / 2
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size] * 3)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_size] * 3,
            rgbaColor=COLOR_RGBA_MAP.get(color_name, [0.5, 0.5, 0.5, 1])
        )
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        return obj_id
    
    def create_cylinder(self, radius=0.025, height=0.05, position=[0, 0, 0], color_name="红色"):
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=COLOR_RGBA_MAP.get(color_name, [0.5, 0.5, 0.5, 1])
        )
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        return obj_id
    
    def spawn_objects(self, num_objects=5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.object_ids = []
        self.object_info = []
        
        shapes = ["cube", "cube", "cylinder", "cylinder", "cube"]
        
        spawn_region = {
            "x_min": -0.18, "x_max": 0.18,
            "y_min": -0.14, "y_max": 0.14,
            "z": 0.022
        }
        
        for i in range(num_objects):
            color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
            shape = shapes[i % len(shapes)]
            
            x = np.random.uniform(spawn_region["x_min"], spawn_region["x_max"])
            y = np.random.uniform(spawn_region["y_min"], spawn_region["y_max"])
            z = spawn_region["z"]
            
            yaw = np.random.uniform(-np.pi, np.pi)
            
            if shape == "cube":
                size = np.random.uniform(0.035, 0.055)
                obj_id = self.create_cube(size=size, position=[x, y, z], color_name=color_name)
                obj_info = {
                    "index": i,
                    "shape": "cube",
                    "color": color_name,
                    "size": size,
                    "expected_position": [x, y, z],
                    "expected_yaw": yaw
                }
            else:
                radius = np.random.uniform(0.02, 0.04)
                height = np.random.uniform(0.03, 0.06)
                obj_id = self.create_cylinder(radius=radius, height=height, position=[x, y, z], color_name=color_name)
                obj_info = {
                    "index": i,
                    "shape": "cylinder",
                    "color": color_name,
                    "radius": radius,
                    "height": height,
                    "expected_position": [x, y, z],
                    "expected_yaw": yaw
                }
            
            p.resetBasePositionAndOrientation(obj_id, [x, y, z], p.getQuaternionFromEuler([0, 0, yaw]))
            
            self.object_ids.append(obj_id)
            self.object_info.append(obj_info)
        
        for _ in range(200):
            p.stepSimulation()
        
        for i, obj_id in enumerate(self.object_ids):
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            self.object_info[i]["actual_position"] = list(pos)
            self.object_info[i]["actual_yaw"] = p.getEulerFromQuaternion(orn)[2]
        
        return self.object_info
    
    def render_images(self):
        img_camera = p.getCameraImage(
            IMAGEWIDTH, IMAGEHEIGHT,
            self.viewMatrix, self.projectionMatrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb = np.reshape(img_camera[2], (IMAGEHEIGHT, IMAGEWIDTH, 4))[:, :, :3]
        depth = np.reshape(img_camera[3], (IMAGEHEIGHT, IMAGEWIDTH))
        segmentation = np.reshape(img_camera[4], (IMAGEHEIGHT, IMAGEWIDTH))
        
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        
        mask_data = np.zeros((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.int64)
        for i, obj_id in enumerate(self.object_ids):
            mask_data[segmentation == obj_id] = i + 1
        
        mask_path = os.path.join(self.output_dir, "camera_mask.mat")
        scio.savemat(mask_path, {"A": mask_data})
        
        return mask_path
    
    def get_ground_truth(self):
        ground_truth = []
        for i, obj_id in enumerate(self.object_ids):
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            yaw = p.getEulerFromQuaternion(orn)[2]
            aabb = p.getAABB(obj_id)
            size = [aabb[1][j] - aabb[0][j] for j in range(3)]
            
            ground_truth.append({
                "index": i,
                "object_id": obj_id,
                "position": list(pos),
                "yaw": yaw,
                "size": size,
                "info": self.object_info[i]
            })
        return ground_truth
    
    def cleanup(self):
        p.disconnect()


def run_vision_detection(img_path, object_ids=None):
    mask_file = os.path.join(img_path, "camera_mask.mat")
    
    if not os.path.exists(mask_file):
        return []
    
    mask_data = scio.loadmat(mask_file).get("A")
    if mask_data is None:
        return []
    
    raw_mask = np.asarray(mask_data, dtype=np.int64)
    height, width = raw_mask.shape[:2]
    
    scale_x = 0.4 / (width / 2)
    scale_y = scale_x
    
    unique_labels = np.unique(raw_mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    detected_objects = []
    
    for label in unique_labels:
        obj_mask = (raw_mask == label).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 80:
            continue
        
        rect = cv2.minAreaRect(contour)
        (x_img, y_img), (w_px, h_px), angle_deg = rect
        
        x_sim = (x_img - width / 2) * scale_x
        y_sim = (height / 2 - y_img) * scale_y
        
        detected_objects.append({
            "label": int(label),
            "position": [x_sim, y_sim, 0.02],
        })
    
    return detected_objects


def calculate_metrics(ground_truth, detected_objects):
    metrics = {
        "total_objects": len(ground_truth),
        "detected_objects": len(detected_objects),
        "detection_rate": 0.0,
        "position_errors": [],
        "mean_position_error": 0.0,
        "max_position_error": 0.0,
    }
    
    if len(ground_truth) == 0:
        return metrics
    
    metrics["detection_rate"] = len(detected_objects) / len(ground_truth)
    
    for gt in ground_truth:
        gt_pos = gt["position"][:2]
        
        best_error = float('inf')
        
        for det in detected_objects:
            det_pos = det["position"][:2]
            error = np.sqrt((gt_pos[0] - det_pos[0])**2 + (gt_pos[1] - det_pos[1])**2)
            
            if error < best_error:
                best_error = error
        
        if best_error < float('inf'):
            metrics["position_errors"].append(best_error)
    
    if metrics["position_errors"]:
        metrics["mean_position_error"] = np.mean(metrics["position_errors"])
        metrics["max_position_error"] = np.max(metrics["position_errors"])
    
    return metrics


def run_single_test(trial_index, num_objects=5, gui=False):
    """运行单次测试"""
    seed = 42 + trial_index * 1000
    
    env = VisionTestEnv(gui=gui)
    
    try:
        env.spawn_objects(num_objects=num_objects, seed=seed)
        env.render_images()
        ground_truth = env.get_ground_truth()
        detected_objects = run_vision_detection(env.output_dir)
        metrics = calculate_metrics(ground_truth, detected_objects)
    finally:
        env.cleanup()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="视觉感知模块批量测试")
    parser.add_argument("--trials", type=int, default=10, help="测试次数")
    parser.add_argument("--num-objects", type=int, default=5, help="每次测试物体数")
    parser.add_argument("--no-gui", action="store_true", help="无GUI模式")
    
    args = parser.parse_args()
    
    print("="*60)
    print("视觉感知模块批量测试")
    print("="*60)
    print(f"测试次数: {args.trials}")
    print(f"每次物体数: {args.num_objects}")
    print(f"GUI模式: {'关闭' if args.no_gui else '开启'}")
    print("="*60)
    
    all_metrics = []
    
    for i in range(args.trials):
        print(f"\n[测试 {i+1}/{args.trials}] 运行中...", end=" ")
        metrics = run_single_test(i, args.num_objects, gui=not args.no_gui)
        all_metrics.append(metrics)
        print(f"检测率: {metrics['detection_rate']*100:.1f}%, "
              f"平均误差: {metrics['mean_position_error']*100:.2f}cm, "
              f"最大误差: {metrics['max_position_error']*100:.2f}cm")
    
    detection_rates = [m["detection_rate"] for m in all_metrics]
    mean_errors = [m["mean_position_error"] for m in all_metrics if m["position_errors"]]
    max_errors = [m["max_position_error"] for m in all_metrics if m["position_errors"]]
    
    all_position_errors = []
    for m in all_metrics:
        all_position_errors.extend(m["position_errors"])
    
    print("\n" + "="*60)
    print("批量测试统计结果")
    print("="*60)
    
    print(f"\n【检测成功率】")
    print(f"  平均检测率: {np.mean(detection_rates)*100:.1f}%")
    print(f"  检测率标准差: {np.std(detection_rates)*100:.1f}%")
    print(f"  最高检测率: {np.max(detection_rates)*100:.1f}%")
    print(f"  最低检测率: {np.min(detection_rates)*100:.1f}%")
    
    print(f"\n【位置误差】")
    if all_position_errors:
        print(f"  平均位置误差: {np.mean(all_position_errors)*100:.2f} cm")
        print(f"  最大位置误差: {np.max(all_position_errors)*100:.2f} cm")
        print(f"  位置误差标准差: {np.std(all_position_errors)*100:.2f} cm")
        print(f"  最小位置误差: {np.min(all_position_errors)*100:.2f} cm")
    else:
        print("  无位置误差数据")
    
    print(f"\n【单次测试统计】")
    print(f"  单次平均误差的均值: {np.mean(mean_errors)*100:.2f} cm")
    print(f"  单次最大误差的均值: {np.mean(max_errors)*100:.2f} cm")
    
    print("\n" + "="*60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "trials": args.trials,
        "num_objects": args.num_objects,
        "summary": {
            "avg_detection_rate": float(np.mean(detection_rates)),
            "detection_rate_std": float(np.std(detection_rates)),
            "avg_position_error_cm": float(np.mean(all_position_errors) * 100) if all_position_errors else 0,
            "max_position_error_cm": float(np.max(all_position_errors) * 100) if all_position_errors else 0,
            "position_error_std_cm": float(np.std(all_position_errors) * 100) if all_position_errors else 0,
        },
        "trials_data": [
            {
                "trial": i+1,
                "detection_rate": m["detection_rate"],
                "mean_position_error": m["mean_position_error"],
                "max_position_error": m["max_position_error"]
            }
            for i, m in enumerate(all_metrics)
        ]
    }
    
    output_path = "test_reports/vision_batch_test_results.json"
    os.makedirs("test_reports", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
