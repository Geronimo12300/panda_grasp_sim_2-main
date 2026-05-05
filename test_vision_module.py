"""
视觉感知模块独立测试脚本

用法:
    python test_vision_module.py --test mask      # 测试mask检测
    python test_vision_module.py --test color     # 测试颜色识别
    python test_vision_module.py --test triangle  # 测试三角体角度推断
    python test_vision_module.py --test all       # 运行所有测试
"""

import os
import sys
import argparse
import cv2
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

COLOR_BGR_MAP = {
    "红色": np.array([51, 51, 217], dtype=np.uint8),
    "绿色": np.array([64, 191, 64], dtype=np.uint8),
    "蓝色": np.array([217, 89, 51], dtype=np.uint8),
    "黄色": np.array([51, 204, 242], dtype=np.uint8),
    "紫色": np.array([204, 76, 166], dtype=np.uint8),
}


def test_mask_detection(img_path):
    """测试mask检测功能"""
    print("\n" + "="*50)
    print("【测试Mask检测】")
    print("="*50)
    
    mask_file = os.path.join(img_path, "camera_mask.mat")
    rgb_file = os.path.join(img_path, "camera_rgb.png")
    
    if not os.path.exists(mask_file):
        print(f"错误: 找不到mask文件 {mask_file}")
        return False
    
    mask_data = scio.loadmat(mask_file).get("A")
    if mask_data is None:
        print("错误: mask文件格式不正确")
        return False
    
    raw_mask = np.asarray(mask_data, dtype=np.int64)
    print(f"Mask尺寸: {raw_mask.shape}")
    
    unique_labels = np.unique(raw_mask)
    unique_labels = unique_labels[unique_labels > 0]
    print(f"检测到 {len(unique_labels)} 个物体")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if os.path.exists(rgb_file):
        rgb_img = cv2.imread(rgb_file)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb_img)
        axes[0].set_title("RGB图像")
    else:
        axes[0].text(0.5, 0.5, "无RGB图像", ha='center', va='center')
        axes[0].set_title("RGB图像 (未找到)")
    
    colored_mask = np.zeros((*raw_mask.shape, 3), dtype=np.uint8)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3] * 255
    
    for i, label in enumerate(unique_labels):
        colored_mask[raw_mask == label] = colors[i % 10].astype(np.uint8)
    
    axes[1].imshow(colored_mask)
    axes[1].set_title(f"Mask分割结果 ({len(unique_labels)}个物体)")
    
    plt.tight_layout()
    plt.savefig("test_mask_result.png", dpi=150)
    print(f"结果已保存到: test_mask_result.png")
    plt.show()
    
    return True


def test_color_detection(img_path, target_color=None):
    """测试颜色识别功能"""
    print("\n" + "="*50)
    print("【测试颜色识别】")
    print("="*50)
    
    rgb_file = os.path.join(img_path, "camera_rgb.png")
    if not os.path.exists(rgb_file):
        print(f"错误: 找不到RGB文件 {rgb_file}")
        return False
    
    image = cv2.imread(rgb_file)
    if image is None:
        print("错误: 无法读取图像")
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("原始图像")
    
    colors_to_test = list(COLOR_BGR_MAP.keys()) if target_color is None else [target_color]
    
    for idx, color_name in enumerate(colors_to_test[:5]):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
        target_bgr = COLOR_BGR_MAP.get(color_name)
        if target_bgr is None:
            continue
        
        color_diff = np.linalg.norm(
            image.astype(np.int16) - target_bgr.reshape(1, 1, 3).astype(np.int16),
            axis=2,
        )
        mask = (color_diff < 95).astype(np.uint8) * 255
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = image_rgb.copy()
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 80:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        axes[row, col].imshow(result)
        axes[row, col].set_title(f"{color_name} - 检测到{len([c for c in contours if cv2.contourArea(c) > 80])}个区域")
    
    plt.tight_layout()
    plt.savefig("test_color_result.png", dpi=150)
    print(f"结果已保存到: test_color_result.png")
    plt.show()
    
    return True


def test_triangle_angle(img_path, target_color="红色"):
    """测试三角体角度推断"""
    print("\n" + "="*50)
    print("【测试三角体角度推断】")
    print("="*50)
    
    left_file = os.path.join(img_path, "camera_rgb_left.png")
    right_file = os.path.join(img_path, "camera_rgb_right.png")
    top_file = os.path.join(img_path, "camera_rgb.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, path) in enumerate([("俯视图", top_file), ("左视图", left_file), ("右视图", right_file)]):
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img)
        else:
            axes[idx].text(0.5, 0.5, f"未找到{name}", ha='center', va='center')
        axes[idx].set_title(name)
    
    plt.tight_layout()
    plt.savefig("test_triangle_views.png", dpi=150)
    print(f"视图已保存到: test_triangle_views.png")
    plt.show()
    
    for color_name in COLOR_BGR_MAP.keys():
        print(f"\n分析颜色: {color_name}")
        
        for view_name, file_name in [("左视图", "camera_rgb_left.png"), ("右视图", "camera_rgb_right.png")]:
            file_path = os.path.join(img_path, file_name)
            if not os.path.exists(file_path):
                print(f"  {view_name}: 文件不存在")
                continue
            
            image = cv2.imread(file_path)
            if image is None:
                print(f"  {view_name}: 无法读取")
                continue
            
            target_bgr = COLOR_BGR_MAP.get(color_name)
            if target_bgr is None:
                continue
            
            color_diff = np.linalg.norm(
                image.astype(np.int16) - target_bgr.reshape(1, 1, 3).astype(np.int16),
                axis=2,
            )
            mask = (color_diff < 95).astype(np.uint8) * 255
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area > 80:
                    x, y, w, h = cv2.boundingRect(contour)
                    rect_area = max(w * h, 1)
                    fill_ratio = area / rect_area
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    
                    is_triangle_like = fill_ratio < 0.72 or len(approx) <= 4
                    print(f"  {view_name}: 面积={area:.0f}, 填充率={fill_ratio:.3f}, 顶点数={len(approx)}, 三角体特征={is_triangle_like}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="视觉感知模块测试")
    parser.add_argument("--test", choices=["mask", "color", "triangle", "all"], default="all",
                        help="测试类型: mask, color, triangle, all")
    parser.add_argument("--path", type=str, default="img/img_urdf",
                        help="图像路径 (默认: img/img_urdf)")
    parser.add_argument("--color", type=str, default=None,
                        help="目标颜色 (用于颜色和三角体测试)")
    
    args = parser.parse_args()
    
    img_path = args.path
    if not os.path.exists(img_path):
        print(f"错误: 图像路径不存在 {img_path}")
        print("请先运行实验生成图像，或指定正确的路径")
        return
    
    print(f"图像路径: {os.path.abspath(img_path)}")
    
    if args.test in ["mask", "all"]:
        test_mask_detection(img_path)
    
    if args.test in ["color", "all"]:
        test_color_detection(img_path, args.color)
    
    if args.test in ["triangle", "all"]:
        test_triangle_angle(img_path, args.color)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
