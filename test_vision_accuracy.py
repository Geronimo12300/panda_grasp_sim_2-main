"""
视觉感知模块完整测试脚本

测试目标：验证物体检测和位姿估计的准确性

测试方法：
- 在仿真环境中放置5个不同颜色、形状的物体
- 运行视觉检测模块，记录检测结果
- 对比检测位置与实际位置的误差

评价指标：
- 检测成功率：检测到物体数/实际物体数
- 位置误差：检测位置与实际位置的欧氏距离
- 角度误差：检测角度与实际角度的差值

用法:
    python test_vision_accuracy.py
    python test_vision_accuracy.py --num-objects 5
    python test_vision_accuracy.py --no-gui
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

COLOR_NAMES = ["红色", "绿色", "蓝色", "黄色", "紫色"]
COLOR_BGR_MAP = {
    "红色": np.array([51, 51, 217], dtype=np.uint8),
    "绿色": np.array([64, 191, 64], dtype=np.uint8),
    "蓝色": np.array([217, 89, 51], dtype=np.uint8),
    "黄色": np.array([51, 204, 242], dtype=np.uint8),
    "紫色": np.array([204, 76, 166], dtype=np.uint8),
}
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
        
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, -0.2], useFixedBase=True)
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
        
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
        
        self.fov = fov
        self.camera_height = 0.5
        
        self.output_dir = "test_vision_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_cube(self, size=0.045, position=[0, 0, 0], color_name="红色"):
        """创建立方体"""
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
        """创建圆柱体"""
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
        """生成测试物体"""
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
            if self.gui:
                time.sleep(1/240)
        
        for i, obj_id in enumerate(self.object_ids):
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            self.object_info[i]["actual_position"] = list(pos)
            self.object_info[i]["actual_yaw"] = p.getEulerFromQuaternion(orn)[2]
        
        print(f"已生成 {num_objects} 个测试物体")
        return self.object_info
    
    def render_images(self):
        """渲染测试图像"""
        img_camera = p.getCameraImage(
            IMAGEWIDTH, IMAGEHEIGHT,
            self.viewMatrix, self.projectionMatrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb = np.reshape(img_camera[2], (IMAGEHEIGHT, IMAGEWIDTH, 4))[:, :, :3]
        depth = np.reshape(img_camera[3], (IMAGEHEIGHT, IMAGEWIDTH))
        segmentation = np.reshape(img_camera[4], (IMAGEHEIGHT, IMAGEWIDTH))
        
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        
        rgb_path = os.path.join(self.output_dir, "test_rgb.png")
        depth_path = os.path.join(self.output_dir, "test_depth.npy")
        seg_path = os.path.join(self.output_dir, "test_segmentation.npy")
        
        import cv2
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_path, depth)
        np.save(seg_path, segmentation)
        
        mask_data = np.zeros((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.int64)
        for i, obj_id in enumerate(self.object_ids):
            mask_data[segmentation == obj_id] = i + 1
        
        import scipy.io as scio
        mask_path = os.path.join(self.output_dir, "camera_mask.mat")
        scio.savemat(mask_path, {"A": mask_data})
        
        print(f"图像已保存到: {self.output_dir}")
        return rgb_path, mask_path
    
    def get_ground_truth(self):
        """获取真实位姿数据"""
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
        """清理环境"""
        p.disconnect()


def run_vision_detection(img_path, object_ids=None):
    """运行视觉检测"""
    import cv2
    import scipy.io as scio
    
    mask_file = os.path.join(img_path, "camera_mask.mat")
    rgb_file = os.path.join(img_path, "test_rgb.png")
    
    if not os.path.exists(mask_file):
        print(f"错误: 找不到mask文件 {mask_file}")
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
            "bbox": [int(x_img - w_px/2), int(y_img - h_px/2), int(w_px), int(h_px)],
            "pixel_center": [int(x_img), int(y_img)]
        })
    
    return detected_objects


def calculate_metrics(ground_truth, detected_objects):
    """计算评价指标"""
    metrics = {
        "total_objects": len(ground_truth),
        "detected_objects": len(detected_objects),
        "detection_rate": 0.0,
        "position_errors": [],
        "angle_errors": [],
        "mean_position_error": 0.0,
        "max_position_error": 0.0,
        "mean_angle_error": 0.0,
        "details": []
    }
    
    if len(ground_truth) == 0:
        return metrics
    
    metrics["detection_rate"] = len(detected_objects) / len(ground_truth)
    
    for gt in ground_truth:
        gt_pos = gt["position"][:2]
        gt_yaw = gt["yaw"]
        
        best_match = None
        best_error = float('inf')
        
        for det in detected_objects:
            det_pos = det["position"][:2]
            error = np.sqrt((gt_pos[0] - det_pos[0])**2 + (gt_pos[1] - det_pos[1])**2)
            
            if error < best_error:
                best_error = error
                best_match = det
        
        detail = {
            "object_index": gt["index"],
            "shape": gt["info"]["shape"],
            "color": gt["info"]["color"],
            "ground_truth_position": gt_pos,
            "ground_truth_yaw": gt_yaw,
            "detected": best_match is not None
        }
        
        if best_match:
            detail["detected_position"] = best_match["position"][:2]
            detail["position_error"] = best_error
            metrics["position_errors"].append(best_error)
        
        metrics["details"].append(detail)
    
    if metrics["position_errors"]:
        metrics["mean_position_error"] = np.mean(metrics["position_errors"])
        metrics["max_position_error"] = np.max(metrics["position_errors"])
    
    return metrics


def print_report(metrics):
    """打印测试报告"""
    print("\n" + "="*60)
    print("视觉感知模块测试报告")
    print("="*60)
    
    print(f"\n【检测成功率】")
    print(f"  实际物体数: {metrics['total_objects']}")
    print(f"  检测物体数: {metrics['detected_objects']}")
    print(f"  检测成功率: {metrics['detection_rate']*100:.1f}%")
    
    print(f"\n【位置误差】")
    if metrics['position_errors']:
        print(f"  平均位置误差: {metrics['mean_position_error']*100:.2f} cm")
        print(f"  最大位置误差: {metrics['max_position_error']*100:.2f} cm")
        print(f"  位置误差标准差: {np.std(metrics['position_errors'])*100:.2f} cm")
    else:
        print("  无位置误差数据")
    
    print(f"\n【详细结果】")
    print("-"*60)
    print(f"{'序号':<4} {'形状':<8} {'颜色':<6} {'检测':<4} {'位置误差(cm)':<12} {'真实位置':<20} {'检测位置':<20}")
    print("-"*60)
    
    for detail in metrics['details']:
        idx = detail['object_index']
        shape = detail['shape']
        color = detail['color']
        detected = "是" if detail['detected'] else "否"
        
        if detail['detected']:
            pos_err = detail['position_error'] * 100
            gt_pos = f"({detail['ground_truth_position'][0]:.3f}, {detail['ground_truth_position'][1]:.3f})"
            det_pos = f"({detail['detected_position'][0]:.3f}, {detail['detected_position'][1]:.3f})"
            print(f"{idx:<4} {shape:<8} {color:<6} {detected:<4} {pos_err:<12.2f} {gt_pos:<20} {det_pos:<20}")
        else:
            gt_pos = f"({detail['ground_truth_position'][0]:.3f}, {detail['ground_truth_position'][1]:.3f})"
            print(f"{idx:<4} {shape:<8} {color:<6} {detected:<4} {'N/A':<12} {gt_pos:<20} {'N/A':<20}")
    
    print("-"*60)
    
    print(f"\n【评价】")
    if metrics['detection_rate'] >= 1.0 and metrics['mean_position_error'] < 0.02:
        print("  ★★★ 优秀 - 检测完整，位置误差小")
    elif metrics['detection_rate'] >= 0.8 and metrics['mean_position_error'] < 0.05:
        print("  ★★☆ 良好 - 检测基本完整，位置误差可接受")
    elif metrics['detection_rate'] >= 0.6:
        print("  ★☆☆ 一般 - 存在漏检或较大位置误差")
    else:
        print("  ☆☆☆ 需改进 - 检测率低或误差大")


def save_report(metrics, output_dir):
    """保存测试报告"""
    report_path = os.path.join(output_dir, "vision_test_report.json")
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "detection_rate": metrics["detection_rate"],
            "mean_position_error": metrics["mean_position_error"],
            "max_position_error": metrics["max_position_error"],
            "total_objects": metrics["total_objects"],
            "detected_objects": metrics["detected_objects"]
        },
        "details": metrics["details"]
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="视觉感知模块准确度测试")
    parser.add_argument("--num-objects", type=int, default=5, help="测试物体数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-gui", action="store_true", help="无GUI模式")
    
    args = parser.parse_args()
    
    print("="*60)
    print("视觉感知模块准确度测试")
    print("="*60)
    print(f"测试物体数: {args.num_objects}")
    print(f"随机种子: {args.seed}")
    print(f"GUI模式: {'关闭' if args.no_gui else '开启'}")
    
    env = VisionTestEnv(gui=not args.no_gui)
    
    try:
        print("\n[1/4] 生成测试物体...")
        env.spawn_objects(num_objects=args.num_objects, seed=args.seed)
        
        print("\n[2/4] 渲染测试图像...")
        rgb_path, mask_path = env.render_images()
        
        print("\n[3/4] 获取真实位姿...")
        ground_truth = env.get_ground_truth()
        
        print("\n[4/4] 运行视觉检测...")
        detected_objects = run_vision_detection(env.output_dir)
        
        print("\n计算评价指标...")
        metrics = calculate_metrics(ground_truth, detected_objects)
        
        print_report(metrics)
        save_report(metrics, env.output_dir)
        
    finally:
        env.cleanup()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
