import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import cv2
from simEnv import SimEnv
import panda_sim_grasp as panda_sim

# 抓取参数
GRASP_GAP = 0.005  # 抓取间隙
GRASP_DEPTH = 0.005  # 抓取深度
GRASP_WIDTH = 0.08  # 抓取宽度

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

    positions.sort(key=lambda x: x[4], reverse=True)
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

    while True:
        p.stepSimulation()
        time.sleep(1. / 250.)

        # 检测按键
        keys = p.getKeyboardEvents()

        # 按 1 渲染图像并捕捉小球位置
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            env.renderURDFImage(save_path=img_path)
            ball_positions = get_positions(img_path)
            print(ball_positions)

        # 按 2 开始抓取
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            if not ball_positions:
                print("未找到物体位置，请先按 1 渲染图像并捕捉位置。")
                continue

            grasp_config['x'], grasp_config['y'], grasp_config['z'], grasp_config['angle'], grasp_config['width'] = ball_positions[0]
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
                    IN_STATE = False
                    PLACE_STATE = True

        # 执行放置
        if PLACE_STATE:
            held_obj_id = env.urdfs_id[grasp_obj_index] if grasp_obj_index < len(env.urdfs_id) else None
            if panda.place_step(pressed_char, held_obj_id):
                PLACE_STATE = False
                grasp_obj_index += 1
                print("放置完成！")

        # 按 3 重置环境
        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            env.loadObjsInURDF(0, obj_nums)
            ball_positions = []
            grasp_obj_index = 0
            panda.place_count = 0
            panda.placed_objects = []
            panda.stack_center = None
            print("环境已重置")

if __name__ == "__main__":
    run()