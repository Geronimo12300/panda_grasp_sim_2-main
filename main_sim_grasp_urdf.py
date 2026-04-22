import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import cv2
from simEnv import SimEnv
import panda_sim_grasp as panda_sim
import requests
import json

DEEPSEEK_API_KEY = "sk-a76e539391214387b356aac52b38391f"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

GRASP_GAP = 0.005
GRASP_DEPTH = 0.005
GRASP_WIDTH = 0.08

def ask_deepseek_for_stacking_order(cubes_info):
    """
    询问DeepSeek大模型最优的堆叠顺序
    cubes_info: 物块信息列表，每个元素包含 {index, scale, color, position}
    返回: 抓取顺序的索引列表
    """
    cube_descriptions = []
    for i, cube in enumerate(cubes_info):
        desc = f"物块{i+1}: 缩放比例={cube['scale']:.2f}, 颜色={cube['color']}, 位置=({cube['position'][0]:.3f}, {cube['position'][1]:.3f})"
        cube_descriptions.append(desc)
    
    prompt = f"""你是一个机器人抓取规划专家。现在有3个立方体物块需要被抓取并堆叠在一起。

物块信息:
{chr(10).join(cube_descriptions)}

请分析这些物块的参数，确定最优的抓取堆叠顺序，使得堆叠后的稳定性最好。

【重要规则】：
1. 堆叠时必须遵循"大在下，小在上"的原则：缩放比例大的物块必须放在最下面，缩放比例小的物块放在上面
2. 第一个抓取的物块会放在最下面，最后一个抓取的物块会放在最上面
3. 因此抓取顺序应该是：缩放比例最大的物块最先抓取，缩放比例最小的物块最后抓取

请直接返回一个JSON格式的抓取顺序，格式如下：
{{"order": [物块编号1, 物块编号2, 物块编号3]}}

其中物块编号是指物块1、物块2、物块3，请只返回JSON，不要有其他内容。"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个机器人抓取规划专家，请直接返回JSON格式的结果。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        print("\n" + "="*50)
        print("【DeepSeek大模型原始返回内容】")
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
        
        print(f"解析后的抓取顺序索引: {order_indices}")
        
        print("\n" + "="*50)
        print("【大模型生成的抓取顺序】")
        for i, idx in enumerate(order_indices):
            if idx < len(cubes_info):
                cube = cubes_info[idx]
                print(f"  第{i+1}个抓取: 物块{idx+1} ({cube['color']}, 缩放={cube['scale']:.2f})")
        print("="*50 + "\n")
        
        return order_indices
        
    except Exception as e:
        print(f"调用DeepSeek API失败: {e}")
        print("使用默认排序（从大到小）")
        return sorted(range(len(cubes_info)), key=lambda i: cubes_info[i]['scale'], reverse=True)

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
        'cube'
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
    obj_nums = 3  # 每次加载的物体个数
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
                    actual_size = 0.04 * scale
                    
                    print(f"  物块{i+1}: 颜色={color}, 缩放比例={scale:.2f}, 实际尺寸={actual_size:.3f}m, 位置=({obj_pos[0]:.3f}, {obj_pos[1]:.3f})")
                    
                    min_dist = float('inf')
                    matched_pos = None
                    for det_pos in detected_positions:
                        dist = ((det_pos[0] - obj_pos[0])**2 + (det_pos[1] - obj_pos[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            matched_pos = det_pos
                    
                    if matched_pos:
                        matched_positions.append(matched_pos)
                        cubes_info.append({
                            'index': i,
                            'scale': scale,
                            'color': color,
                            'position': [obj_pos[0], obj_pos[1]]
                        })
                        print(f"    -> 匹配到检测位置: ({matched_pos[0]:.3f}, {matched_pos[1]:.3f})")
                
                print("="*50 + "\n")
                print("正在询问DeepSeek大模型最优堆叠顺序...")
                
                order_indices = ask_deepseek_for_stacking_order(cubes_info)
                grasp_order_indices = order_indices
                
                ball_positions = [matched_positions[i] for i in order_indices if i < len(matched_positions)]
                
                print(f"最终抓取顺序: {ball_positions}")
                print(f"物块ID顺序: {grasp_order_indices}")

        # 按 2 开始抓取
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            if not ball_positions:
                print("未找到物体位置，请先按 1 渲染图像并捕捉位置。")
                continue

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
                print(
                    f"抓取中心已修正到物块包围盒中心: "
                    f"({grasp_config['x']:.4f}, {grasp_config['y']:.4f}, {grasp_config['z']:.4f}), "
                    f"抓取角度修正为: {grasp_config['angle']:.4f} rad"
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
            if panda.grasp_step(target_position, grasp_config['angle'], grasp_config['width']):
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
