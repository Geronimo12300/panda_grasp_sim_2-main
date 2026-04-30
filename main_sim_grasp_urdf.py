import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import cv2
import base64
from simEnv import SimEnv
import panda_sim_grasp as panda_sim
import requests
import json

BAILIAN_API_KEY = "sk-4ea064d0eb6b4c39b6ae8479e8975443"
BAILIAN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

GRASP_GAP = 0.005
GRASP_DEPTH = 0.005
GRASP_WIDTH = 0.08

def encode_image_to_data_url(image_path):
    if not image_path or not os.path.exists(image_path):
        return None

    suffix = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


COLOR_BGR_MAP = {
    "红色": np.array([51, 51, 217], dtype=np.uint8),
    "绿色": np.array([64, 191, 64], dtype=np.uint8),
    "蓝色": np.array([217, 89, 51], dtype=np.uint8),
    "黄色": np.array([51, 204, 242], dtype=np.uint8),
    "紫色": np.array([204, 76, 166], dtype=np.uint8),
}


def normalize_grasp_angle(angle):
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return angle


def analyze_colored_object_contour(image_path, target_color):
    if not image_path or not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    target_bgr = COLOR_BGR_MAP.get(target_color)
    if target_bgr is None:
        return None

    color_diff = np.linalg.norm(image.astype(np.int16) - target_bgr.reshape(1, 1, 3).astype(np.int16), axis=2)
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
        return candidate_perpendicular, "侧视图未识别到目标颜色，回退为夹取平整侧面的默认方向"

    triangle_like_detected = any(
        info["fill_ratio"] < 0.72 or info["vertices"] <= 4
        for _, info in analyses
    )

    if triangle_like_detected:
        reason = "侧视轮廓更接近三角面，采用垂直于三角面法向的夹取方向"
        return candidate_perpendicular, reason

    reason = "侧视轮廓更接近平整矩形侧面，保持当前方向抓取"
    return candidate_direct, reason


def ask_bailian_for_stacking_order(cubes_info, image_paths=None):
    """
    询问阿里云百炼 / Qwen-VL 模型最优的堆叠顺序
    cubes_info: 物块信息列表，每个元素至少包含 {index, color}
    返回: 抓取顺序的索引列表
    """
    cube_labels = []
    for i, cube in enumerate(cubes_info):
        color = cube.get('color', f'物块{i+1}')
        cube_labels.append(f"物块{i+1}={color}")
    
    cube_count = len(cubes_info)
    cube_indices_text = ", ".join([f"物块{i+1}" for i in range(cube_count)])
    json_example = '{"order": [' + ", ".join([f"物块编号{i+1}" for i in range(cube_count)]) + ']}'

    prompt = f"""你是一个机器人抓取规划专家。现在有{cube_count}个物块需要被抓取并堆叠在一起。

物块编号与颜色对应关系:
{", ".join(cube_labels)}

我会提供三张场景截图，分别来自俯视相机和两个侧视相机。请你只根据这三张图片中各个物块的外观、形状、相对大小和顶部/底部特征，判断怎样堆叠最稳、最高。

【重要规则】：
1. 你只能根据三张图片进行判断，不要使用额外假设
2. 你需要重点判断哪个物块更适合放在底层，哪个物块更适合放在上层
3. 底部更宽、更稳、顶部更平整、承托能力更强的物块更适合放在下层
4. 顶部尖、顶部斜、顶部不平整的物块不适合承托其他物体，应尽量放在上层
5. 第一个抓取的物块会放在最下面，最后一个抓取的物块会放在最上面
6. 返回顺序时，必须严格使用给定的物块编号，不要创造新的编号

请直接返回一个JSON格式的抓取顺序，格式如下：
{json_example}

其中物块编号是指 {cube_indices_text}，请只返回JSON，不要有其他内容。"""

    user_content = [{"type": "text", "text": prompt}]
    for image_path in image_paths or []:
        image_data_url = encode_image_to_data_url(image_path)
        if image_data_url:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_data_url}
            })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BAILIAN_API_KEY}"
    }
    
    data = {
        "model": "qwen-vl-max-latest",
        "messages": [
            {"role": "system", "content": "你是一个机器人抓取规划专家，请结合图像直接返回JSON格式的结果。"},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 160
    }
    
    try:
        response = requests.post(BAILIAN_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        print("\n" + "="*50)
        print("【阿里云百炼 / Qwen-VL 原始返回内容】")
        print(content)
        print("="*50)
        
        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]
        
        print(f"提取的JSON字符串: {json_match.strip()}")
        
        order_result = json.loads(json_match.strip())
        order = order_result["order"]
        
        print(f"解析出的order字段: {order}")
        
        order_indices = []
        for item in order:
            if isinstance(item, int):
                order_indices.append(item - 1)
            elif isinstance(item, str):
                num = int(''.join(filter(str.isdigit, item)))
                order_indices.append(num - 1)
        
        order_indices = enforce_stacking_constraints(order_indices, cubes_info)
        print(f"解析后的抓取顺序索引: {order_indices}")
        
        print("\n" + "="*50)
        print("【阿里云百炼 / Qwen-VL 生成的抓取顺序】")
        for i, idx in enumerate(order_indices):
            if idx < len(cubes_info):
                cube = cubes_info[idx]
                color = cube.get('color', f'物块{idx+1}')
                shape = cube.get('shape')
                volume = cube.get('volume')

                details = [color]
                if shape:
                    details.append(shape)
                if volume is not None:
                    details.append(f"体积={volume:.8f}m^3")

                print(f"  第{i+1}个抓取: 物块{idx+1} ({', '.join(details)})")
        print("="*50 + "\n")
        
        return order_indices
        
    except Exception as e:
        print(f"调用阿里云百炼 / Qwen-VL API失败: {e}")
        print("使用默认排序（按体积从大到小）")
        default_order = sorted(
            range(len(cubes_info)),
            key=lambda i: (cubes_info[i].get('top_only', False), -cubes_info[i].get('volume', 0.0))
        )
        return enforce_stacking_constraints(default_order, cubes_info)

def enforce_stacking_constraints(order_indices, cubes_info):
    valid_indices = [idx for idx in order_indices if 0 <= idx < len(cubes_info)]
    missing_indices = [i for i in range(len(cubes_info)) if i not in valid_indices]
    merged_order = valid_indices + missing_indices

    top_only_indices = [
        idx for idx in merged_order
        if cubes_info[idx].get('top_only', False) or cubes_info[idx].get('is_triangle', False)
    ]
    normal_indices = [
        idx for idx in merged_order
        if not (cubes_info[idx].get('top_only', False) or cubes_info[idx].get('is_triangle', False))
    ]

    normal_indices.sort(key=lambda idx: cubes_info[idx].get('volume', 0.0), reverse=True)
    top_only_indices.sort(key=lambda idx: cubes_info[idx].get('volume', 0.0), reverse=True)

    return normal_indices + top_only_indices


def push_triangle_objects_to_end(order_indices, urdf_filenames):
    normal_indices = []
    triangle_indices = []

    for idx in order_indices:
        filename = os.path.basename(urdf_filenames[idx]) if 0 <= idx < len(urdf_filenames) else ""
        if filename.startswith('cone_top'):
            triangle_indices.append(idx)
        else:
            normal_indices.append(idx)

    return normal_indices + triangle_indices

def get_positions(path):
    """
    获取所有小球的位置
    :param path: 从上方拍摄的容器内物体的图片所在路径
    :return: 物体位置列表 [(x1, y1, z1, angle1, width1), (x2, y2, z2, angle2, width2), ...]
    """

    # 读取图像（假设路径为目录，图片名为 'capture.png'）
    img_file = os.path.join(path, 'camera_mask.png')
    if not os.path.exists(img_file):
        return []

    image = cv2.imread(img_file)
    cv2.imwrite(path + '/camera_rgb_new.png', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path + '/camera_rgb_gray.png', gray)
    # 图像预处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(path + '/camera_rgb_thresh.png', thresh)

    # 形态学操作（可选）
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.bitwise_not(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2))
    cv2.imwrite(path + '/camera_rgb_cleaned.png', cleaned)


    # 检测轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    positions = []
    height, width = image.shape[:2]
    scale_x = 0.4 / (width / 2)
    scale_y = scale_x

    for cnt in contours:
        # 过滤小面积噪声
        if cv2.contourArea(cnt) < 100:
            continue

        # 计算最小外接矩形
        rect = cv2.minAreaRect(cnt)
        (x_img, y_img), (w_px, h_px), angle_deg = rect

        # 坐标转换
        x_sim = (x_img - width / 2) * scale_x
        y_sim = (height / 2 - y_img) * scale_y  # Y轴方向翻转
        z_sim = 0.02  # 假设物体在平面上

        # 动态计算抓取宽度（取外接矩形长边）
        grasp_width = max(w_px, h_px) * scale_x  # 转换为仿真环境单位
        grasp_width = np.clip(grasp_width, 0.03, 0.20)  # 限制抓取宽度范围

        # 计算抓取角度（弧度）
        grasp_angle = np.deg2rad(angle_deg)
        if w_px < h_px:
            grasp_angle += np.pi / 2  # 修正长边方向
        if grasp_angle > np.pi / 2:
            grasp_angle -= np.pi

        positions.append((x_sim, y_sim, z_sim, grasp_angle, grasp_width))

    return positions

def run():
    # 数据库路径
    database_path = [
        'cube',
        'cylinder',
        'cone_top'
    ]

    # 连接 PyBullet 服务器
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置 PyBullet 数据路径

    # 初始化虚拟环境
    env = SimEnv(p, database_path)

    # 初始化 Panda 机器人
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])

    # 抓取状态
    GRASP_STATE = False
    grasp_config = {'x': 0, 'y': 0, 'z': 0.05, 'angle': 0, 'width': GRASP_WIDTH}

    # 放置状态
    PLACE_STATE = False

    # 输入状态
    IN_STATE = False
    pressed_char = None

    # 图像保存路径
    img_path = 'img/img_urdf'

    # 加载物体
    obj_nums = 5  # 每次加载的物体个数
    env.loadObjsInURDF(0, obj_nums)

    # 全局变量
    ball_positions = []
    grasp_obj_index = 0
    grasp_order_indices = []

    while True:
        p.stepSimulation()
        time.sleep(1. / 250.)

        # 检测按键
        keys = p.getKeyboardEvents()

        # 按 1 渲染图像并捕捉小球位置
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            env.renderURDFImage(save_path=img_path)
            detected_positions = get_positions(img_path)
            print(f"图像检测到的位置数量: {len(detected_positions)}")
            
            if len(detected_positions) > 0:
                cubes_info = []
                matched_positions = []
                
                print("\n" + "="*50)
                print("【仿真环境中的物块信息】")
                
                for i in range(len(env.urdfs_id)):
                    scale = env.urdfs_scale[i] if i < len(env.urdfs_scale) else 1.0
                    obj_pos, _ = p.getBasePositionAndOrientation(env.urdfs_id[i])
                    color = env.urdfs_colors[i] if i < len(env.urdfs_colors) else f'物块{i+1}'
                    shape = env.urdfs_shapes[i] if hasattr(env, 'urdfs_shapes') and i < len(env.urdfs_shapes) else '立方体'
                    source_filename = os.path.basename(env.urdfs_filename[i]) if i < len(env.urdfs_filename) else ""
                    is_triangle = source_filename.startswith('cone_top')
                    top_only = is_triangle or shape == '尖锥顶物块'
                    aabb = p.getAABB(env.urdfs_id[i])
                    size_x = aabb[1][0] - aabb[0][0]
                    size_y = aabb[1][1] - aabb[0][1]
                    size_z = aabb[1][2] - aabb[0][2]

                    shape_params = {}
                    if shape == '圆柱体':
                        diameter = max(size_x, size_y)
                        height = size_z
                        volume = np.pi * (diameter / 2.0) ** 2 * height
                        shape_params.update({
                            'diameter': diameter,
                            'height': height,
                            'volume': volume
                        })
                        shape_summary = f"直径={diameter:.4f}m, 高度={height:.4f}m, 体积={volume:.8f}m^3"
                    elif shape == '尖锥顶物块':
                        base_diameter = max(size_x, size_y)
                        height = size_z
                        volume = (np.pi * (base_diameter / 2.0) ** 2 * height) / 3.0
                        shape_params.update({
                            'base_diameter': base_diameter,
                            'height': height,
                            'volume': volume
                        })
                        shape_summary = f"底面直径={base_diameter:.4f}m, 高度={height:.4f}m, 体积={volume:.8f}m^3"
                    else:
                        edge_length = max(size_x, size_y, size_z)
                        volume = edge_length ** 3
                        shape_params.update({
                            'edge_length': edge_length,
                            'volume': volume
                        })
                        shape_summary = f"边长={edge_length:.4f}m, 体积={volume:.8f}m^3"
                    
                    print(
                        f"  物块{i+1}: 形状={shape}, 颜色={color}, 缩放比例={scale:.2f}, "
                        f"{shape_summary}, 位置=({obj_pos[0]:.3f}, {obj_pos[1]:.3f})"
                    )
                    
                    min_dist = float('inf')
                    matched_pos = None
                    for det_pos in detected_positions:
                        dist = ((det_pos[0] - obj_pos[0])**2 + (det_pos[1] - obj_pos[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            matched_pos = det_pos
                    
                    if matched_pos:
                        matched_positions.append(matched_pos)
                        cube_info = {
                            'index': i,
                            'scale': scale,
                            'color': color,
                            'shape': shape,
                            'top_only': top_only,
                            'is_triangle': is_triangle,
                            'source_filename': source_filename,
                            'position': [obj_pos[0], obj_pos[1]]
                        }
                        cube_info.update(shape_params)
                        cubes_info.append(cube_info)
                        print(f"    -> 匹配到检测位置: ({matched_pos[0]:.3f}, {matched_pos[1]:.3f})")
                
                print("="*50 + "\n")
                print("正在询问阿里云百炼 / Qwen-VL 最优堆叠顺序...")
                
                scene_image_paths = [
                    os.path.join(img_path, 'camera_rgb.png'),
                    os.path.join(img_path, 'camera_rgb_left.png'),
                    os.path.join(img_path, 'camera_rgb_right.png')
                ]
                order_indices = ask_bailian_for_stacking_order(cubes_info, image_paths=scene_image_paths)
                order_indices = push_triangle_objects_to_end(order_indices, env.urdfs_filename)
                grasp_order_indices = order_indices
                
                ball_positions = [matched_positions[i] for i in order_indices if i < len(matched_positions)]
                
                print(f"最终抓取顺序: {ball_positions}")
                print(f"物块ID顺序: {grasp_order_indices}")

        # 按 2 开始抓取
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            if not ball_positions:
                print("未找到物体位置，请先按 1 渲染图像并捕捉位置。")
                continue

            grasp_order_indices = push_triangle_objects_to_end(grasp_order_indices, env.urdfs_filename)

            detected_x, detected_y, detected_z, detected_angle, detected_width = ball_positions[0]
            grasp_config['angle'] = detected_angle
            grasp_config['width'] = detected_width

            target_obj_id = None
            if grasp_obj_index < len(grasp_order_indices):
                obj_idx = grasp_order_indices[grasp_obj_index]
                if obj_idx < len(env.urdfs_id):
                    target_obj_id = env.urdfs_id[obj_idx]

            if target_obj_id is not None:
                aabb_min, aabb_max = p.getAABB(target_obj_id)
                size_x = aabb_max[0] - aabb_min[0]
                size_y = aabb_max[1] - aabb_min[1]
                grasp_config['x'] = (aabb_min[0] + aabb_max[0]) / 2.0
                grasp_config['y'] = (aabb_min[1] + aabb_max[1]) / 2.0
                grasp_config['z'] = (aabb_min[2] + aabb_max[2]) / 2.0
                _, obj_orn = p.getBasePositionAndOrientation(target_obj_id)
                obj_yaw = p.getEulerFromQuaternion(obj_orn)[2]
                if obj_yaw > np.pi / 2:
                    obj_yaw -= np.pi
                elif obj_yaw < -np.pi / 2:
                    obj_yaw += np.pi
                grasp_config['angle'] = obj_yaw

                target_filename = os.path.basename(env.urdfs_filename[obj_idx]) if obj_idx < len(env.urdfs_filename) else ""
                if target_filename.startswith('cone_top'):
                    flat_face_span = min(size_x, size_y)
                    grasp_config['width'] = np.clip(flat_face_span + 2 * GRASP_GAP, 0.02, GRASP_WIDTH)
                    target_color = env.urdfs_colors[obj_idx] if obj_idx < len(env.urdfs_colors) else None
                    adjusted_angle, angle_reason = infer_triangle_grasp_angle_from_side_views(
                        img_path,
                        target_color,
                        grasp_config['angle']
                    )
                    grasp_config['angle'] = adjusted_angle
                    print(
                        f"????????: ???????????, "
                        f"????={flat_face_span:.4f}m, ????={grasp_config['width']:.4f}m, "
                        f"????={grasp_config['angle']:.4f} rad, {angle_reason}"
                    )

                print(
                    f"???????????????: "
                    f"({grasp_config['x']:.4f}, {grasp_config['y']:.4f}, {grasp_config['z']:.4f}), "
                    f"???????: {grasp_config['angle']:.4f} rad"
                )
            else:
                grasp_config['x'] = detected_x
                grasp_config['y'] = detected_y
                grasp_config['z'] = detected_z

            GRASP_STATE = True
            ball_positions.pop(0)
        # 执行抓取
        if GRASP_STATE:
            target_position = [grasp_config['x'], grasp_config['y'], grasp_config['z'] - GRASP_DEPTH]
            current_held_obj_id = None
            if grasp_obj_index < len(grasp_order_indices):
                current_obj_idx = grasp_order_indices[grasp_obj_index]
                if current_obj_idx < len(env.urdfs_id):
                    current_held_obj_id = env.urdfs_id[current_obj_idx]

            if panda.grasp_step(target_position, grasp_config['angle'], grasp_config['width'], current_held_obj_id):
                GRASP_STATE = False
                IN_STATE = True
                print("抓取完成！")

        if IN_STATE:
            for char in ['7', '8', '9', '0']:
                key_code = ord(char)
                if key_code in keys and keys[key_code] & p.KEY_WAS_TRIGGERED:
                    pressed_char = char
                    panda.start_place()
                    IN_STATE = False
                    PLACE_STATE = True

        # 执行放置
        if PLACE_STATE:
            if grasp_obj_index < len(grasp_order_indices):
                obj_idx = grasp_order_indices[grasp_obj_index]
                held_obj_id = env.urdfs_id[obj_idx] if obj_idx < len(env.urdfs_id) else None
            else:
                held_obj_id = None
            if panda.place_step(pressed_char, held_obj_id):
                PLACE_STATE = False
                grasp_obj_index += 1
                print("放置完成！")

        # 按 3 重置环境
        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            env.loadObjsInURDF(0, obj_nums)
            ball_positions = []
            grasp_obj_index = 0
            grasp_order_indices = []
            panda.place_count = 0
            panda.placed_objects = []
            panda.stack_center = None
            print("环境已重置")

if __name__ == "__main__":
    run()
