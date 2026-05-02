import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import cv2
import base64
import shutil
import scipy.io as scio
from simEnv import SimEnv
import panda_sim_grasp as panda_sim
import requests
import json
import re

BAILIAN_API_KEY = "sk-4ea064d0eb6b4c39b6ae8479e8975443"
BAILIAN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

GRASP_GAP = 0.005
GRASP_DEPTH = 0.005
GRASP_WIDTH = 0.08
SCENE_SETTLE_LINEAR_VEL = 0.015
SCENE_SETTLE_ANGULAR_VEL = 0.08
SCENE_SETTLE_STEPS = 45

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


def ask_bailian_for_stack_success(image_paths=None, expected_count=None):
    count_text = f"{expected_count}个物块" if expected_count is not None else "这些物块"
    prompt = f"""你是一个机器人堆叠结果验收助手。

我会提供当前堆叠完成后的场景图片，请你判断 {count_text} 是否已经成功堆叠。

判定标准：
1. 主要关注目标物块是否形成了明显的竖向堆叠，而不是散落在桌面
2. 如果大部分物块已经叠在一起且整体稳定，没有明显倒塌，判定为成功
3. 如果物块散落、明显滑落、倒塌，或者没有形成堆叠，判定为失败
4. 只根据图片判断，不要补充额外假设

请只返回 JSON，格式如下：
{{"success": true, "reason": "一句简短中文说明"}}
"""

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
            {"role": "system", "content": "你是一个机器人堆叠结果验收助手，请直接返回JSON格式结果。"},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 120
    }

    try:
        response = requests.post(BAILIAN_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        print("\n" + "=" * 50)
        print("【阿里云百炼 / Qwen-VL 堆叠验收原始返回】")
        print(content)
        print("=" * 50)

        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        verdict = json.loads(json_match.strip())
        success = bool(verdict.get("success", False))
        reason = verdict.get("reason", "未提供原因")
        return success, reason
    except Exception as e:
        print(f"调用阿里云百炼 / Qwen-VL 堆叠验收失败: {e}")
        return False, f"模型验收失败: {e}"


def clamp_value(value, lower, upper):
    return max(lower, min(upper, value))


def parse_object_index(value, object_count):
    if isinstance(value, int):
        idx = value - 1
    elif isinstance(value, str):
        digits = ''.join(ch for ch in value if ch.isdigit())
        if not digits:
            return None
        idx = int(digits) - 1
    else:
        return None

    if 0 <= idx < object_count:
        return idx
    return None


def ask_bailian_for_pick_place_actions(cubes_info, image_paths=None, stack_target=None):
    """
    让大模型直接输出结构化抓取/放置动作。
    返回原始动作列表；后续仍需做本地安全裁剪。
    """
    stack_target = stack_target or {"x": 0.5, "y": 0.0, "z": 0.0}
    object_lines = []
    for cube in cubes_info:
        grasp_pose = cube.get("default_grasp_pose", {})
        object_lines.append(
            f"物块{cube['index'] + 1}: "
            f"颜色={cube.get('color', '未知')}, "
            f"形状={cube.get('shape', '未知')}, "
            f"位置=({cube['position'][0]:.3f}, {cube['position'][1]:.3f}), "
            f"默认抓取候选=(x={grasp_pose.get('x', 0.0):.3f}, y={grasp_pose.get('y', 0.0):.3f}, "
            f"z={grasp_pose.get('z', 0.0):.3f}, yaw={grasp_pose.get('yaw', 0.0):.3f}, "
            f"width={grasp_pose.get('width', 0.05):.3f})"
        )

    json_example = """{
  "actions": [
    {
      "target_object": "物块1",
      "grasp_pose": {"x": 0.10, "y": -0.02, "z": 0.03, "yaw": 0.0, "width": 0.05},
      "place_pose": {"x": 0.50, "y": 0.00, "z": 0.00},
      "reason": "适合做底层"
    }
  ]
}"""

    prompt = f"""你是一个机械臂抓取与堆叠规划专家。

现在有 {len(cubes_info)} 个物块，你需要直接给出“抓谁、怎么抓、放到哪里”的动作计划。

场景中物块信息如下：
{os.linesep.join(object_lines)}

我还会给你三张场景图：俯视图、左侧视图、右侧视图。

堆叠目标点建议为：
place_pose = (x={stack_target['x']:.3f}, y={stack_target['y']:.3f}, z={stack_target['z']:.3f})

规则：
1. 每个物块只能出现一次。
2. 第一个 action 放在最底层，最后一个 action 放在最上层。
3. 请优先使用我提供的默认抓取候选，只在必要时做小幅调整。
4. 你输出的 grasp_pose 必须是机械臂可以执行的单次抓取位姿。
5. 所有 place_pose 应该围绕同一个堆叠中心，便于竖直堆叠。
6. 如果三角体不适合承托其他物块，应尽量放在上层。
7. 只能使用给定的物块编号，不要创造新编号。

请只返回 JSON，不要添加其他解释。格式如下：
{json_example}
"""

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
            {"role": "system", "content": "你是机器人抓取规划专家，请直接返回结构化 JSON 动作计划。"},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 600
    }

    try:
        response = requests.post(BAILIAN_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        print("\n" + "=" * 50)
        print("【阿里云百炼 / Qwen-VL 动作计划原始返回】")
        print(content)
        print("=" * 50)

        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        plan = json.loads(json_match.strip())
        actions = plan.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError("actions 字段不是列表")
        return actions
    except Exception as e:
        print(f"调用阿里云百炼 / Qwen-VL 动作规划失败: {e}")
        return None

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

def get_positions(path, object_ids=None):
    """
    从 PyBullet 原始 segmentation mask 中提取每个真实物块的抓取候选。
    返回值为 {物块索引: (x, y, z, angle, width)}。
    """
    mask_file = os.path.join(path, 'camera_mask.mat')
    if not os.path.exists(mask_file) or not object_ids:
        return {}

    mask_data = scio.loadmat(mask_file).get('A')
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
        grasp_width = np.clip(max(w_px, h_px) * scale_x, 0.03, GRASP_WIDTH)
        grasp_angle = np.deg2rad(angle_deg)
        if w_px < h_px:
            grasp_angle += np.pi / 2
        grasp_angle = normalize_grasp_angle(grasp_angle)

        positions[obj_index] = (x_sim, y_sim, 0.02, grasp_angle, grasp_width)
        debug_mask[binary > 0] = min(255, 40 + obj_index * 40)

    cv2.imwrite(os.path.join(path, 'camera_mask_objects.png'), debug_mask)
    return positions

def run():
    AUTO_RUN = True
    AUTO_PLACE_CHAR = '7'
    AUTO_CAPTURE_DELAY_STEPS = 240
    TRIAL_TIMEOUT_STEPS = 30000
    GRASP_ACTION_TIMEOUT_STEPS = 2200
    PLACE_ACTION_TIMEOUT_STEPS = 6000
    RESULTS_MARKDOWN_PATH = 'experiment_results.md'
    SCREENSHOT_DIR = '1'
    FINAL_RENDER_DIR = 'img/img_urdf'
    TRIALS_PER_GROUP = 10
    RESUME_FROM_GROUP_INDEX = 3
    RESUME_FROM_TRIAL_INDEX = 5
    RESTART_FROM_RESUME_GROUP = True
    EXPERIMENT_GROUPS = [
        {'name': '第一组：5个正方体', 'shapes': ['cube'] * 5},
        {'name': '第二组：5个圆柱体', 'shapes': ['cylinder'] * 5},
        {'name': '第三组：3个正方体 + 2个圆柱体', 'shapes': ['cube', 'cube', 'cube', 'cylinder', 'cylinder']},
        {'name': '第四组：2个正方体 + 2个圆柱体 + 1个三角体', 'shapes': ['cube', 'cube', 'cylinder', 'cylinder', 'cone_top']},
    ]

    database_path = ['cube', 'cylinder', 'cone_top']

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    env = SimEnv(p, database_path)
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])

    GRASP_STATE = False
    grasp_config = {'x': 0, 'y': 0, 'z': 0.05, 'angle': 0, 'width': GRASP_WIDTH}
    PLACE_STATE = False
    IN_STATE = False
    pressed_char = None
    auto_capture_pending = AUTO_RUN
    auto_capture_countdown = AUTO_CAPTURE_DELAY_STEPS
    img_path = FINAL_RENDER_DIR
    obj_nums = 5

    ball_positions = []
    grasp_obj_index = 0
    grasp_order_indices = []
    planned_actions = []
    stack_evaluation_done = False
    auto_scene_settle_counter = 0
    trial_step_counter = 0
    grasp_action_step_counter = 0
    place_action_step_counter = 0
    current_group_index = 0
    current_trial_index = 0
    batch_finished = False
    experiment_results = [{'name': group['name'], 'records': []} for group in EXPERIMENT_GROUPS]

    def queue_auto_capture():
        nonlocal auto_capture_pending, auto_capture_countdown
        auto_capture_pending = AUTO_RUN
        auto_capture_countdown = AUTO_CAPTURE_DELAY_STEPS

    def ensure_output_paths():
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)

    def render_group_links(records):
        if not records:
            return ['- 暂无截图']

        lines = []
        for record in records:
            links = record['links']
            lines.append(
                f"- 第{record['trial']}次："
                f"[顶视图]({links['top']}) / "
                f"[左视图]({links['left']}) / "
                f"[右视图]({links['right']})"
            )
        return lines

    def write_markdown_report():
        lines = [
            '# 自动化堆叠实验结果',
            '',
            f'- 每组实验次数：{TRIALS_PER_GROUP}',
            f'- 物块总数：{obj_nums}',
            '',
        ]

        for group_result in experiment_results:
            records = group_result['records']
            success_count = sum(1 for record in records if record['success'])
            lines.append(f"## {group_result['name']}")
            lines.append('')
            if records:
                lines.append('| 次数 | 大模型判定 | 原因 |')
                lines.append('| --- | --- | --- |')
                for record in records:
                    verdict = '成功' if record['success'] else '失败'
                    reason = str(record['reason']).replace('\n', ' ').strip()
                    lines.append(f"| {record['trial']} | {verdict} | {reason} |")
                lines.append('')
                lines.append(f'成功次数：{success_count}/{len(records)}')
            else:
                lines.append('当前组还没有完成的实验结果。')
            lines.append('')
            lines.append('截图链接：')
            lines.extend(render_group_links(records))
            lines.append('')

        with open(RESULTS_MARKDOWN_PATH, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write('\n'.join(lines))

    def default_trial_links(group_number, trial_number):
        return {
            'top': f'{SCREENSHOT_DIR}/group{group_number}_trial{trial_number:02d}_top.png',
            'left': f'{SCREENSHOT_DIR}/group{group_number}_trial{trial_number:02d}_left.png',
            'right': f'{SCREENSHOT_DIR}/group{group_number}_trial{trial_number:02d}_right.png',
        }

    def load_existing_markdown_report():
        if not os.path.exists(RESULTS_MARKDOWN_PATH):
            return

        group_by_name = {group['name']: index for index, group in enumerate(EXPERIMENT_GROUPS)}
        parsed_records = {index: {} for index in range(len(EXPERIMENT_GROUPS))}
        current_group_index_for_parse = None

        with open(RESULTS_MARKDOWN_PATH, 'r', encoding='utf-8') as markdown_file:
            for raw_line in markdown_file:
                line = raw_line.strip()
                if line.startswith('## '):
                    group_name = line[3:].strip()
                    current_group_index_for_parse = group_by_name.get(group_name)
                    continue

                if current_group_index_for_parse is None:
                    continue

                if line.startswith('|') and not line.startswith('| ---') and '次数' not in line:
                    cells = [cell.strip() for cell in line.strip('|').split('|')]
                    if len(cells) >= 3 and cells[0].isdigit():
                        trial_number = int(cells[0])
                        parsed_records[current_group_index_for_parse][trial_number] = {
                            'trial': trial_number,
                            'success': cells[1] == '成功',
                            'reason': cells[2],
                            'links': default_trial_links(current_group_index_for_parse + 1, trial_number),
                        }
                    continue

                link_match = re.match(
                    r'- 第(\d+)次：\[顶视图\]\(([^)]+)\) / \[左视图\]\(([^)]+)\) / \[右视图\]\(([^)]+)\)',
                    line,
                )
                if link_match:
                    trial_number = int(link_match.group(1))
                    record = parsed_records[current_group_index_for_parse].setdefault(
                        trial_number,
                        {
                            'trial': trial_number,
                            'success': False,
                            'reason': '从已有报告恢复，未解析到判定原因',
                            'links': default_trial_links(current_group_index_for_parse + 1, trial_number),
                        },
                    )
                    record['links'] = {
                        'top': link_match.group(2),
                        'left': link_match.group(3),
                        'right': link_match.group(4),
                    }

        for group_index, records_by_trial in parsed_records.items():
            experiment_results[group_index]['records'] = [
                records_by_trial[trial_number]
                for trial_number in sorted(records_by_trial)
                if 1 <= trial_number <= TRIALS_PER_GROUP
            ]

    def prepare_resume_state():
        nonlocal current_group_index, current_trial_index, batch_finished

        resume_group_index = max(0, min(RESUME_FROM_GROUP_INDEX, len(EXPERIMENT_GROUPS) - 1))
        resume_trial_index = max(0, min(RESUME_FROM_TRIAL_INDEX, TRIALS_PER_GROUP - 1))
        if RESTART_FROM_RESUME_GROUP:
            experiment_results[resume_group_index]['records'] = [
                record
                for record in experiment_results[resume_group_index]['records']
                if record['trial'] <= resume_trial_index
            ]
            for group_index in range(resume_group_index + 1, len(EXPERIMENT_GROUPS)):
                experiment_results[group_index]['records'] = []
            current_group_index = resume_group_index
            current_trial_index = resume_trial_index
            print(
                f"已保留第{resume_group_index}组及以前的记录，并保留当前组前{resume_trial_index}次记录，将从 "
                f"{EXPERIMENT_GROUPS[current_group_index]['name']} 第{current_trial_index + 1}次重新开始。"
            )
            return

        for group_index in range(resume_group_index, len(EXPERIMENT_GROUPS)):
            completed_trials = {record['trial'] for record in experiment_results[group_index]['records']}
            for trial_index in range(TRIALS_PER_GROUP):
                if trial_index + 1 not in completed_trials:
                    current_group_index = group_index
                    current_trial_index = trial_index
                    print(
                        f"已加载已有记录，将从 {EXPERIMENT_GROUPS[current_group_index]['name']} "
                        f"第{current_trial_index + 1}次继续。"
                    )
                    return

        batch_finished = True
        print('从第二组开始的实验记录已经全部完成。')

    def copy_trial_screenshots(group_number, trial_number):
        screenshot_targets = {
            'top': ('camera_rgb.png', f'group{group_number}_trial{trial_number:02d}_top.png'),
            'left': ('camera_rgb_left.png', f'group{group_number}_trial{trial_number:02d}_left.png'),
            'right': ('camera_rgb_right.png', f'group{group_number}_trial{trial_number:02d}_right.png'),
        }
        copied_links = {}
        for view_name, (source_name, target_name) in screenshot_targets.items():
            source_path = os.path.join(img_path, source_name)
            target_path = os.path.join(SCREENSHOT_DIR, target_name)
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
            copied_links[view_name] = os.path.join(SCREENSHOT_DIR, target_name).replace('\\', '/')
        return copied_links

    def reset_robot_and_runtime():
        nonlocal GRASP_STATE, PLACE_STATE, IN_STATE, pressed_char
        nonlocal ball_positions, grasp_obj_index, grasp_order_indices, planned_actions
        nonlocal stack_evaluation_done, auto_scene_settle_counter, trial_step_counter
        nonlocal grasp_action_step_counter, place_action_step_counter

        panda.reset_to_initial_pose(settle_steps=40)
        GRASP_STATE = False
        PLACE_STATE = False
        IN_STATE = False
        pressed_char = None
        ball_positions = []
        grasp_obj_index = 0
        grasp_order_indices = []
        planned_actions = []
        stack_evaluation_done = False
        auto_scene_settle_counter = 0
        trial_step_counter = 0
        grasp_action_step_counter = 0
        place_action_step_counter = 0
        queue_auto_capture()

    def clear_world_before_trial():
        panda.reset_to_initial_pose(settle_steps=120)
        if getattr(env, 'urdfs_id', None):
            env.removeObjsInURDF()
        for _ in range(60):
            p.stepSimulation()
        panda.reset_to_initial_pose(settle_steps=20)

    def start_current_trial():
        group = EXPERIMENT_GROUPS[current_group_index]
        print('\n' + '=' * 60)
        print(f"开始实验：{group['name']}，第 {current_trial_index + 1}/{TRIALS_PER_GROUP} 次")
        print(f"物块组合：{group['shapes']}")
        print('=' * 60)
        clear_world_before_trial()
        env.loadObjsInURDF(0, obj_nums, shape_sequence=group['shapes'])
        reset_robot_and_runtime()

    def build_default_grasp_pose(obj_idx, detected_pos):
        detected_x, detected_y, detected_z, detected_angle, detected_width = detected_pos
        pose = {
            'x': float(detected_x),
            'y': float(detected_y),
            'z': float(detected_z),
            'yaw': float(detected_angle),
            'width': float(detected_width),
        }

        if obj_idx >= len(env.urdfs_id):
            return pose

        target_obj_id = env.urdfs_id[obj_idx]
        aabb_min, aabb_max = p.getAABB(target_obj_id)
        size_x = aabb_max[0] - aabb_min[0]
        size_y = aabb_max[1] - aabb_min[1]
        size_z = aabb_max[2] - aabb_min[2]
        pose['x'] = float((aabb_min[0] + aabb_max[0]) / 2.0)
        pose['y'] = float((aabb_min[1] + aabb_max[1]) / 2.0)
        pose['z'] = float((aabb_min[2] + aabb_max[2]) / 2.0)

        _, obj_orn = p.getBasePositionAndOrientation(target_obj_id)
        obj_yaw = p.getEulerFromQuaternion(obj_orn)[2]
        if obj_yaw > np.pi / 2:
            obj_yaw -= np.pi
        elif obj_yaw < -np.pi / 2:
            obj_yaw += np.pi
        pose['yaw'] = float(obj_yaw)

        target_filename = os.path.basename(env.urdfs_filename[obj_idx]) if obj_idx < len(env.urdfs_filename) else ''
        if target_filename.startswith('cube'):
            cube_edge = size_z
            pose['width'] = float(np.clip(cube_edge + 2 * GRASP_GAP, 0.025, GRASP_WIDTH))
            pose['z'] = float(aabb_min[2] + 0.38 * size_z)
        elif target_filename.startswith('cylinder'):
            cylinder_diameter = max(size_x, size_y)
            pose['width'] = float(np.clip(cylinder_diameter + 0.025, 0.045, GRASP_WIDTH))
            pose['z'] = float(aabb_min[2] + 0.42 * size_z)
            pose['yaw'] = 0.0
        elif target_filename.startswith('cone_top'):
            flat_face_span = min(size_x, size_y)
            pose['width'] = float(np.clip(flat_face_span + 2 * GRASP_GAP, 0.02, GRASP_WIDTH))
            target_color = env.urdfs_colors[obj_idx] if obj_idx < len(env.urdfs_colors) else None
            adjusted_angle, _ = infer_triangle_grasp_angle_from_side_views(img_path, target_color, pose['yaw'])
            pose['yaw'] = float(adjusted_angle)
        return pose

    def build_default_action_plan(cubes_info, detected_positions, stack_target_xy=(0.5, 0.0)):
        ordered_indices = enforce_stacking_constraints(
            list(range(len(cubes_info))),
            cubes_info,
        )
        actions = []
        for order_rank, obj_idx in enumerate(ordered_indices):
            default_grasp_pose = cubes_info[obj_idx]['default_grasp_pose']
            actions.append({
                'target_index': obj_idx,
                'grasp_pose': dict(default_grasp_pose),
                'place_pose': {
                    'x': float(stack_target_xy[0]),
                    'y': float(stack_target_xy[1]),
                    'z': float(order_rank * 0.04),
                },
                'reason': '默认规则规划：按稳定性与形状约束堆叠',
            })
        return actions

    def sanitize_action_plan(raw_actions, default_actions, object_count, stack_target_xy=(0.5, 0.0)):
        if not raw_actions:
            return list(default_actions)

        def get_float(source, key, fallback):
            try:
                return float(source.get(key, fallback))
            except (TypeError, ValueError, AttributeError):
                return float(fallback)

        default_by_index = {action['target_index']: action for action in default_actions}
        planned = []
        used_indices = set()

        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                continue
            obj_idx = parse_object_index(raw_action.get('target_object'), object_count)
            if obj_idx is None or obj_idx in used_indices or obj_idx not in default_by_index:
                continue

            default_action = default_by_index[obj_idx]
            default_grasp = default_action['grasp_pose']
            raw_grasp = raw_action.get('grasp_pose', {}) or {}
            raw_place = raw_action.get('place_pose', {}) or {}

            grasp_pose = {
                'x': clamp_value(get_float(raw_grasp, 'x', default_grasp['x']), default_grasp['x'] - 0.05, default_grasp['x'] + 0.05),
                'y': clamp_value(get_float(raw_grasp, 'y', default_grasp['y']), default_grasp['y'] - 0.05, default_grasp['y'] + 0.05),
                'z': clamp_value(get_float(raw_grasp, 'z', default_grasp['z']), max(0.0, default_grasp['z'] - 0.02), default_grasp['z'] + 0.03),
                'yaw': float(normalize_grasp_angle(get_float(raw_grasp, 'yaw', default_grasp['yaw']))),
                'width': clamp_value(get_float(raw_grasp, 'width', default_grasp['width']), 0.02, GRASP_WIDTH),
            }
            place_pose = {
                'x': clamp_value(get_float(raw_place, 'x', stack_target_xy[0]), stack_target_xy[0] - 0.02, stack_target_xy[0] + 0.02),
                'y': clamp_value(get_float(raw_place, 'y', stack_target_xy[1]), stack_target_xy[1] - 0.02, stack_target_xy[1] + 0.02),
                'z': clamp_value(get_float(raw_place, 'z', default_action['place_pose']['z']), 0.0, 0.25),
            }

            planned.append({
                'target_index': obj_idx,
                'grasp_pose': grasp_pose,
                'place_pose': place_pose,
                'reason': str(raw_action.get('reason', '大模型动作规划')).strip() or '大模型动作规划',
            })
            used_indices.add(obj_idx)

        for default_action in default_actions:
            if default_action['target_index'] not in used_indices:
                planned.append(default_action)

        return planned

    def advance_to_next_trial():
        nonlocal current_group_index, current_trial_index, batch_finished

        if current_trial_index + 1 < TRIALS_PER_GROUP:
            current_trial_index += 1
            start_current_trial()
            return

        if current_group_index + 1 < len(EXPERIMENT_GROUPS):
            current_group_index += 1
            current_trial_index = 0
            start_current_trial()
            return

        batch_finished = True
        print('\n所有自动化实验均已完成。')
        print(f"实验结果已写入：{os.path.abspath(RESULTS_MARKDOWN_PATH)}")
        print(f"截图目录：{os.path.abspath(SCREENSHOT_DIR)}")

    def finalize_current_trial(forced_failure_reason=None):
        nonlocal stack_evaluation_done

        if stack_evaluation_done:
            return

        stack_evaluation_done = True
        env.renderURDFImage(save_path=img_path)
        evaluation_images = [
            os.path.join(img_path, 'camera_rgb.png'),
            os.path.join(img_path, 'camera_rgb_left.png'),
            os.path.join(img_path, 'camera_rgb_right.png')
        ]
        model_success, model_reason = ask_bailian_for_stack_success(
            image_paths=evaluation_images,
            expected_count=obj_nums
        )
        success = model_success if forced_failure_reason is None else False
        reason = model_reason if forced_failure_reason is None else f"{forced_failure_reason}; 大模型判断：{model_reason}"
        links = copy_trial_screenshots(current_group_index + 1, current_trial_index + 1)
        experiment_results[current_group_index]['records'].append({
            'trial': current_trial_index + 1,
            'success': success,
            'reason': reason,
            'links': links,
        })
        write_markdown_report()

        print('\n' + '=' * 50)
        print('【堆叠验收结果】')
        print(f"实验组别: {EXPERIMENT_GROUPS[current_group_index]['name']}")
        print(f"实验次数: 第 {current_trial_index + 1} 次")
        print(f"是否成功: {'成功' if success else '失败'}")
        print(f"原因: {reason}")
        print('=' * 50 + '\n')

        advance_to_next_trial()

    def scene_is_settled():
        if not env.urdfs_id:
            return False

        for obj_id in env.urdfs_id:
            linear_vel, angular_vel = p.getBaseVelocity(obj_id)
            if np.linalg.norm(linear_vel) > SCENE_SETTLE_LINEAR_VEL:
                return False
            if np.linalg.norm(angular_vel) > SCENE_SETTLE_ANGULAR_VEL:
                return False
        return True

    def plan_scene():
        nonlocal ball_positions, grasp_order_indices, grasp_obj_index, planned_actions, stack_evaluation_done

        env.renderURDFImage(save_path=img_path)
        detected_positions = get_positions(img_path, env.urdfs_id)
        print(f"Mask 检测到的物块数量: {len(detected_positions)}/{len(env.urdfs_id)}")

        ball_positions = []
        grasp_order_indices = []
        planned_actions = []
        grasp_obj_index = 0
        stack_evaluation_done = False
        if len(detected_positions) != len(env.urdfs_id):
            missing = [i + 1 for i in range(len(env.urdfs_id)) if i not in detected_positions]
            print(f"Mask 未检测到全部物块，缺失物块编号: {missing}，等待重新渲染。")
            return False

        cubes_info = []

        print('\n' + '=' * 50)
        print('【仿真环境中的物块信息】')

        for i in range(len(env.urdfs_id)):
            scale = env.urdfs_scale[i] if i < len(env.urdfs_scale) else 1.0
            obj_pos, _ = p.getBasePositionAndOrientation(env.urdfs_id[i])
            color = env.urdfs_colors[i] if i < len(env.urdfs_colors) else f'物块{i+1}'
            shape = env.urdfs_shapes[i] if hasattr(env, 'urdfs_shapes') and i < len(env.urdfs_shapes) else '正方体'
            source_filename = os.path.basename(env.urdfs_filename[i]) if i < len(env.urdfs_filename) else ''
            is_triangle = source_filename.startswith('cone_top')
            top_only = is_triangle or shape == '三角体'
            aabb = p.getAABB(env.urdfs_id[i])
            size_x = aabb[1][0] - aabb[0][0]
            size_y = aabb[1][1] - aabb[0][1]
            size_z = aabb[1][2] - aabb[0][2]

            shape_params = {}
            if shape == '圆柱体':
                diameter = max(size_x, size_y)
                height = size_z
                volume = np.pi * (diameter / 2.0) ** 2 * height
                shape_params.update({'diameter': diameter, 'height': height, 'volume': volume})
                shape_summary = f"直径={diameter:.4f}m, 高度={height:.4f}m, 体积={volume:.8f}m^3"
            elif shape == '三角体':
                base_diameter = max(size_x, size_y)
                height = size_z
                volume = (np.pi * (base_diameter / 2.0) ** 2 * height) / 3.0
                shape_params.update({'base_diameter': base_diameter, 'height': height, 'volume': volume})
                shape_summary = f"底面直径={base_diameter:.4f}m, 高度={height:.4f}m, 体积={volume:.8f}m^3"
            else:
                edge_length = max(size_x, size_y, size_z)
                volume = edge_length ** 3
                shape_params.update({'edge_length': edge_length, 'volume': volume})
                shape_summary = f"边长={edge_length:.4f}m, 体积={volume:.8f}m^3"

            print(
                f"  物块{i+1}: 形状={shape}, 颜色={color}, 缩放比例={scale:.2f}, "
                f"{shape_summary}, 位置=({obj_pos[0]:.3f}, {obj_pos[1]:.3f})"
            )

            mask_pos = detected_positions[i]
            cube_info = {
                'index': i,
                'scale': scale,
                'color': color,
                'shape': shape,
                'top_only': top_only,
                'is_triangle': is_triangle,
                'source_filename': source_filename,
                'position': [obj_pos[0], obj_pos[1]],
            }
            cube_info['default_grasp_pose'] = build_default_grasp_pose(i, mask_pos)
            cube_info.update(shape_params)
            cubes_info.append(cube_info)
            print(f"    -> Mask 抓取候选 ({mask_pos[0]:.3f}, {mask_pos[1]:.3f})")

        print('=' * 50 + '\n')
        print('正在询问阿里云百炼 / Qwen-VL 直接生成抓取放置动作...')

        if not cubes_info:
            return False

        stack_target_xy = (0.5, 0.0)
        scene_image_paths = [
            os.path.join(img_path, 'camera_rgb.png'),
            os.path.join(img_path, 'camera_rgb_left.png'),
            os.path.join(img_path, 'camera_rgb_right.png')
        ]
        default_actions = build_default_action_plan(cubes_info, detected_positions, stack_target_xy=stack_target_xy)
        raw_actions = ask_bailian_for_pick_place_actions(
            cubes_info,
            image_paths=scene_image_paths,
            stack_target={'x': stack_target_xy[0], 'y': stack_target_xy[1], 'z': 0.0},
        )
        planned_actions = sanitize_action_plan(
            raw_actions,
            default_actions,
            len(cubes_info),
            stack_target_xy=stack_target_xy,
        )
        grasp_order_indices = [action['target_index'] for action in planned_actions]
        ball_positions = [
            (
                action['grasp_pose']['x'],
                action['grasp_pose']['y'],
                action['grasp_pose']['z'],
                action['grasp_pose']['yaw'],
                action['grasp_pose']['width'],
            )
            for action in planned_actions
        ]

        print("\n" + "=" * 50)
        print("【最终动作计划】")
        for action_idx, action in enumerate(planned_actions, 1):
            grasp_pose = action['grasp_pose']
            place_pose = action['place_pose']
            print(
                f"  Action {action_idx}: 物块{action['target_index'] + 1}, "
                f"grasp=({grasp_pose['x']:.3f}, {grasp_pose['y']:.3f}, {grasp_pose['z']:.3f}, "
                f"yaw={grasp_pose['yaw']:.3f}, width={grasp_pose['width']:.3f}), "
                f"place=({place_pose['x']:.3f}, {place_pose['y']:.3f}, {place_pose['z']:.3f}), "
                f"reason={action['reason']}"
            )
        print("=" * 50 + "\n")
        return len(planned_actions) > 0

    def begin_next_grasp():
        nonlocal GRASP_STATE, grasp_action_step_counter

        if not planned_actions:
            print('未生成动作计划，请先完成场景规划。')
            return False

        if grasp_obj_index >= len(planned_actions):
            return False

        current_action = planned_actions[grasp_obj_index]
        grasp_pose = current_action['grasp_pose']
        grasp_config['x'] = grasp_pose['x']
        grasp_config['y'] = grasp_pose['y']
        grasp_config['z'] = grasp_pose['z']
        grasp_config['angle'] = grasp_pose['yaw']
        grasp_config['width'] = grasp_pose['width']

        target_obj_id = None
        obj_idx = current_action['target_index']
        if obj_idx < len(env.urdfs_id):
            target_obj_id = env.urdfs_id[obj_idx]

        if target_obj_id is not None:
            print(
                f"执行动作: 抓取物块{obj_idx + 1}, "
                f"({grasp_config['x']:.4f}, {grasp_config['y']:.4f}, {grasp_config['z']:.4f}), "
                f"抓取角度={grasp_config['angle']:.4f} rad, "
                f"夹爪宽度={grasp_config['width']:.4f}m, "
                f"放置点=({current_action['place_pose']['x']:.4f}, {current_action['place_pose']['y']:.4f}, {current_action['place_pose']['z']:.4f})"
            )

        GRASP_STATE = True
        grasp_action_step_counter = 0
        return True

    def abort_current_grasp(reason):
        nonlocal GRASP_STATE, ball_positions, grasp_order_indices, planned_actions, auto_scene_settle_counter
        nonlocal grasp_action_step_counter

        print(reason)
        panda.force_release_held_object()
        panda.reset()
        GRASP_STATE = False
        grasp_action_step_counter = 0
        if AUTO_RUN:
            ball_positions = []
            grasp_order_indices = []
            planned_actions = []
            auto_scene_settle_counter = 0
            queue_auto_capture()

    def abort_current_place(reason):
        nonlocal PLACE_STATE, place_action_step_counter

        print(reason)
        panda.force_release_held_object()
        for _ in range(80):
            p.stepSimulation()
        PLACE_STATE = False
        place_action_step_counter = 0
        if AUTO_RUN:
            finalize_current_trial(forced_failure_reason=reason)

    ensure_output_paths()
    load_existing_markdown_report()
    prepare_resume_state()
    write_markdown_report()
    if not batch_finished:
        start_current_trial()

    if AUTO_RUN:
        print('Auto run enabled: experiment scheduler is active.')

    while True:
        if batch_finished:
            break

        p.stepSimulation()
        time.sleep(1. / 250.)
        trial_step_counter += 1

        keys = p.getKeyboardEvents()

        if scene_is_settled():
            auto_scene_settle_counter += 1
        else:
            auto_scene_settle_counter = 0

        if AUTO_RUN and auto_capture_pending and not GRASP_STATE and not PLACE_STATE and not IN_STATE:
            auto_capture_countdown -= 1
            if auto_capture_countdown <= 0 and auto_scene_settle_counter >= SCENE_SETTLE_STEPS:
                if plan_scene():
                    auto_capture_pending = False
                    auto_scene_settle_counter = 0
                else:
                    auto_capture_countdown = AUTO_CAPTURE_DELAY_STEPS

        if not AUTO_RUN and ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            plan_scene()

        if AUTO_RUN and not auto_capture_pending and not GRASP_STATE and not PLACE_STATE and not IN_STATE:
            if planned_actions and grasp_obj_index < len(planned_actions) and auto_scene_settle_counter >= SCENE_SETTLE_STEPS:
                begin_next_grasp()
                auto_scene_settle_counter = 0

        if not AUTO_RUN and ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            begin_next_grasp()

        if GRASP_STATE:
            target_position = [grasp_config['x'], grasp_config['y'], grasp_config['z'] - GRASP_DEPTH]
            current_held_obj_id = None
            if grasp_obj_index < len(grasp_order_indices):
                current_obj_idx = grasp_order_indices[grasp_obj_index]
                if current_obj_idx < len(env.urdfs_id):
                    current_held_obj_id = env.urdfs_id[current_obj_idx]

            if panda.grasp_step(target_position, grasp_config['angle'], grasp_config['width'], current_held_obj_id):
                GRASP_STATE = False
                if panda.held_object_id is None:
                    panda.reset()
                    print('抓取失败：未抓稳目标物块，重新规划当前场景。')
                    if AUTO_RUN:
                        ball_positions = []
                        grasp_order_indices = []
                        auto_scene_settle_counter = 0
                        queue_auto_capture()
                    else:
                        IN_STATE = False
                else:
                    if AUTO_RUN:
                        pressed_char = AUTO_PLACE_CHAR
                        panda.set_place_target_override(planned_actions[grasp_obj_index]['place_pose'])
                        panda.start_place()
                        place_action_step_counter = 0
                        PLACE_STATE = True
                    else:
                        IN_STATE = True
                    print('抓取完成。')

        if IN_STATE:
            for char in ['7', '8', '9', '0']:
                key_code = ord(char)
                if key_code in keys and keys[key_code] & p.KEY_WAS_TRIGGERED:
                    pressed_char = char
                    panda.set_place_target_override(None)
                    panda.start_place()
                    place_action_step_counter = 0
                    IN_STATE = False
                    PLACE_STATE = True

        if PLACE_STATE:
            if grasp_obj_index < len(grasp_order_indices):
                obj_idx = grasp_order_indices[grasp_obj_index]
                held_obj_id = env.urdfs_id[obj_idx] if obj_idx < len(env.urdfs_id) else None
            else:
                held_obj_id = None
            if panda.place_step(pressed_char, held_obj_id):
                PLACE_STATE = False
                place_action_step_counter = 0
                grasp_obj_index += 1
                print('放置完成。')
                if AUTO_RUN and grasp_obj_index >= len(grasp_order_indices):
                    print('Auto run finished: all objects processed.')
                if grasp_obj_index >= len(grasp_order_indices):
                    finalize_current_trial()

        if GRASP_STATE:
            grasp_action_step_counter += 1
            if grasp_action_step_counter > GRASP_ACTION_TIMEOUT_STEPS:
                abort_current_grasp('抓取动作超时：已强制释放并重新规划当前场景。')

        if PLACE_STATE:
            place_action_step_counter += 1
            if place_action_step_counter > PLACE_ACTION_TIMEOUT_STEPS:
                abort_current_place('放置动作超时：已强制松爪，当前小次实验按失败记录并进入下一次。')

        if AUTO_RUN and trial_step_counter > TRIAL_TIMEOUT_STEPS:
            print('当前实验超过时长上限，按失败记录并自动进入下一次。')
            finalize_current_trial(forced_failure_reason='实验超时，未在限定步数内完成堆叠')

        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            start_current_trial()
            print('环境已重置。')

    p.disconnect()

if __name__ == "__main__":
    run()
