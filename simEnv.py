"""
虚拟环境文件
初始化虚拟环境，加载物体，渲染图像，保存图像

(待写) ！！ 保存虚拟环境状态，以便离线抓取测试
"""

import pybullet as p
import pybullet_data
import time
import math
import os
import glob
import random
import cv2
import shutil
import numpy as np
import scipy.io as scio
from mesh import Mesh
import tool

# 容器尺寸（单位：米）
container_size = (0.8, 0.8)  # 容器的宽度和高度
# 图像尺寸
IMAGEWIDTH = 640
IMAGEHEIGHT = 480

# 计算容器在图像中的像素范围
def get_container_roi(container_size, image_width, image_height, fov, aspect):
    """
    计算容器在图像中的 ROI（感兴趣区域）
    :param container_size: 容器的物理尺寸 (width, height)
    :param image_width: 图像的宽度（像素）
    :param image_height: 图像的高度（像素）
    :param fov: 相机的垂直视场角（度）
    :param aspect: 图像的宽高比
    :return: ROI 的像素范围 (x_min, y_min, x_max, y_max)
    """
    # 计算容器在图像中的像素尺寸
    fov_rad = math.radians(fov)
    container_width_pixels = int((container_size[0] / (2 * math.tan(fov_rad / 2) * aspect)) * image_width)
    container_height_pixels = int((container_size[1] / (2 * math.tan(fov_rad / 2))) * image_height)

    # 计算 ROI 的边界
    x_min = (image_width - container_width_pixels) // 2
    x_max = x_min + container_width_pixels
    y_min = (image_height - container_height_pixels) // 2
    y_max = y_min + container_height_pixels

    return x_min, y_min, x_max, y_max

nearPlane = 0.01
farPlane = 10

fov = 60    # 垂直视场 图像高tan(30) * 0.7 *2 = 0.8082903m
aspect = IMAGEWIDTH / IMAGEHEIGHT

size=(0.8, 0.8)     # 桌面深度图实际尺寸 m
unit=0.0002          # 每个像素的长度 0.1mm


def get_urdf_xyz(filename):
    """
    获取urdfs_xyz
    filename: urdf文件名
    """
    with open(filename) as f:
        line = f.readlines()[15][32:-5]
        strs = line.split(" ")
        return [float(strs[0]), float(strs[1]), float(strs[2])]

def get_urdf_scale(filename):
    """
    获取urdfs_scale
    filename: urdf文件名
    """
    with open(filename) as f:
        line = f.readlines()[17]
        idx = line.find('scale') + 7
        strs = line[idx:-5].split(" ")
        return float(strs[0])



class SimEnv(object):
    """
    虚拟环境类
    """
    def __init__(self, bullet_client, path):
        """
        path: 模型路径
        """
        self.p = bullet_client
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加路径


        ground_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0])
        ground_id = p.createMultiBody(0, ground_shape)
        p.changeDynamics(ground_id, -1, rollingFriction=0, restitution=1.0)

        self.second_container_position = [0.5, -0.05, 0.15]
        self.second_container_id = self.p.loadURDF('models/urdf/cola.urdf', self.second_container_position,globalScaling=0.5)
        self.third_container_position = [0.5, 0.2, 0]  # 第三个容器的位置，可以根据需要调整
        self.third_container_id = self.p.loadURDF('models/urdf/cola.urdf', self.third_container_position,globalScaling=0.6)
        tilt_angle1 = 0  # 倾斜弧度
        tilt_angle2 = 0  # 倾斜弧度
        tilt_orientation1 = self.p.getQuaternionFromEuler([tilt_angle1, 0, 0])  # 将欧拉角转换为四元数
        tilt_orientation2 = self.p.getQuaternionFromEuler([tilt_angle2, 0, 0])  # 将欧拉角转换为四元数   
        self.p.resetBasePositionAndOrientation(
        self.second_container_id,
        posObj=[0.5, -0.05, 0],  # 容器的位置
        ornObj=tilt_orientation1  # 容器的姿态
        )
        self.p.resetBasePositionAndOrientation(
        self.third_container_id,
        posObj=[0.5, 0.2, 0],  # 容器的位置
        ornObj=tilt_orientation2  # 容器的姿态
        )

        self.p.setGravity(0, 0, -9.8) # 设置重力

        self.flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)

        # 加载相机
        self.viewMatrix = self.p.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [0, 1, 0])
        self.leftViewMatrix = self.p.computeViewMatrix([-0.45, 0, 0.18], [0, 0, 0.05], [0, 0, 1])
        self.rightViewMatrix = self.p.computeViewMatrix([0.45, 0, 0.18], [0, 0, 0.05], [0, 0, 1])
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        # 获取urdf物体列表
        if isinstance(path, str):
            self.urdfs_list = self._find_urdf_files(path)
        elif isinstance(path, list):
            self.urdfs_list = []
            for pth in path:
                self.urdfs_list.extend(self._find_urdf_files(pth))
            self.urdfs_list.sort()

        if not self.urdfs_list:
            raise ValueError("No URDF files found in the specified path(s). Please check the path and ensure URDF files exist.")
        
        self.num_urdf = 0
        self.urdfs_id = []
        self.obj_ids = self.urdfs_id  # 添加 obj_ids 属性，指向 urdfs_id
        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]

        # 获取obj物体列表
        # self.objs_list = glob.glob(os.path.join(path, '*.obj'))
        # self.objs_list.sort()
        # self.num_obj = 0
        # self.objs_id = []

    def _find_urdf_files(self, path):
        """
        递归查找指定路径及其子文件夹中的 URDF 文件
        :param path: 要搜索的路径
        :return: 找到的 URDF 文件列表
        """
        urdf_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.urdf'):
                    urdf_files.append(os.path.join(root, file))
        return urdf_files
    
    def _urdf_nums(self):
        return len(self.urdfs_list)


    def init_single_mesh(self, urdfname, quaternion):
        """
        初始化mesh
        """
        # 获取obj当前位姿
        offset = [0, 0, 0]
        # quaternion = [0, 0, 0, 1]

        # 计算从obj坐标系到URDF坐标系的变换矩阵
        # 平移：self.xyz [-0.019, 0.019, -0.019]  旋转: 欧拉角[1.570796, 0, 0]
        # (1) 欧拉角->四元数
        orn = self.p.getQuaternionFromEuler([1.570796, 0, 0])
        # (2) 四元数->旋转矩阵
        rot = tool.quaternion_to_rotation_matrix(orn)
        # (3) 计算变换矩阵
        urdf_xyz = get_urdf_xyz(urdfname)
        mat = tool.getTransfMat(urdf_xyz, rot)

        # 获取obj文件路径
        objURDF_name = urdfname.replace('.urdf', '.obj')      # 单物体时使用

        # 读取obj文件，并根据scale缩放
        urdf_scale = get_urdf_scale(urdfname)
        mesh = Mesh(objURDF_name, urdf_scale)

        # 计算物体的变换矩阵(从URDF坐标系到物体坐标系)
        rotate_mat = tool.quaternion_to_rotation_matrix(quaternion)  # 四元数转旋转矩阵
        transMat = tool.getTransfMat(offset, rotate_mat)

        transMat = np.matmul(transMat, mat) # !!!! 注意乘的顺序, 使用

        # 根据旋转矩阵调整mesh顶点坐标
        mesh.transform(transMat)

        return mesh




    """
    原始加载函数
    """
    def loadObjsInURDF(self, idx, num, shape_sequence=None, scale_mode="random"):
        """
        以URDF的格式加载多个obj物体

        num: 加载物体的个数
        idx: 开始的id
            idx为负数时，随机加载num个物体
            idx为非负数时，从id开始加载num个物体
        """
        assert idx >= 0, f"idx 的值无效: {idx} (必须为非负数)"
        assert num >= 0, f"num 的值无效: {num} (必须为非负数)"

        # 检查 self.urdfs_list 的长度
        if len(self.urdfs_list) == 0:
            raise ValueError("self.urdfs_list 为空，无法加载物体")

    # 检查 idx 是否超出范围
        if idx >= len(self.urdfs_list):
            raise ValueError(f"idx ({idx}) 超出了 urdfs_list 的范围 (0-{len(self.urdfs_list) - 1})")

    # 计算 self.num_urdf
        if self.urdfs_id:
            self.removeObjsInURDF()

        self.num_urdf = num

    # 获取物体文件，允许从现有立方体模板中复用生成更多实例
        available_urdfs = self.urdfs_list[idx:]
        if not available_urdfs:
            raise ValueError("可用 URDF 列表为空，无法加载物体")

        cube_urdfs = [path for path in available_urdfs if os.path.basename(path).startswith('cube')]
        cylinder_urdfs = [path for path in available_urdfs if os.path.basename(path).startswith('cylinder')]
        cone_top_urdfs = [path for path in available_urdfs if os.path.basename(path).startswith('cone_top')]
        cuboid_bar_urdfs = [path for path in available_urdfs if os.path.basename(path).startswith('cuboid_bar')]

        if shape_sequence is not None:
            if len(shape_sequence) != self.num_urdf:
                raise ValueError(f"shape_sequence 长度({len(shape_sequence)})必须等于 num({self.num_urdf})")

            shape_pools = {
                'cube': cube_urdfs,
                'cylinder': cylinder_urdfs,
                'cone_top': cone_top_urdfs,
                'cuboid_bar': cuboid_bar_urdfs,
            }
            shape_counters = {key: 0 for key in shape_pools}
            self.urdfs_filename = []
            for shape_name in shape_sequence:
                if shape_name not in shape_pools:
                    raise ValueError(f"不支持的 shape_sequence 类型: {shape_name}")
                candidates = shape_pools[shape_name]
                if not candidates:
                    raise ValueError(f"当前资源中缺少 {shape_name} 对应的 URDF 文件")
                candidate_idx = shape_counters[shape_name] % len(candidates)
                self.urdfs_filename.append(candidates[candidate_idx])
                shape_counters[shape_name] += 1
        elif cube_urdfs and cylinder_urdfs and cone_top_urdfs:
            self.urdfs_filename = []
            for i in range(self.num_urdf):
                if i == self.num_urdf - 1:
                    self.urdfs_filename.append(cone_top_urdfs[0])
                elif i >= 2:
                    cylinder_offset = i - 2
                    self.urdfs_filename.append(cylinder_urdfs[cylinder_offset % len(cylinder_urdfs)])
                else:
                    self.urdfs_filename.append(cube_urdfs[i % len(cube_urdfs)])
        else:
            self.urdfs_filename = [
                available_urdfs[i % len(available_urdfs)]
                for i in range(self.num_urdf)
            ]

        print('self.urdfs_filename = ', self.urdfs_filename)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        self.urdfs_colors = []
        self.urdfs_shapes = []
        
        spawn_region = {
            "x_min": -0.18,
            "x_max": 0.18,
            "y_min": -0.14,
            "y_max": 0.14,
            "z": 0.022
        }
        min_spawn_distance = 0.09

        fixed_scale_palette = [0.75, 0.85, 1.0, 1.15, 1.25]
        shared_shape_scales = {}
        randomized_scales = []
        special_three_block_mode = (
            scale_mode == "fixed"
            and self.num_urdf == 3
            and len([name for name in self.urdfs_filename if os.path.basename(name).startswith('cube')]) == 1
            and (
                len([name for name in self.urdfs_filename if os.path.basename(name).startswith('cuboid_bar')]) == 2
                or len([name for name in self.urdfs_filename if os.path.basename(name).startswith('cone_top')]) == 2
            )
        )
        random_special_three_block_mode = (
            scale_mode == "random"
            and self.num_urdf == 3
            and len([name for name in self.urdfs_filename if os.path.basename(name).startswith('cube')]) == 1
            and (
                len([name for name in self.urdfs_filename if os.path.basename(name).startswith('cuboid_bar')]) == 2
                or len([name for name in self.urdfs_filename if os.path.basename(name).startswith('cone_top')]) == 2
            )
        )
        for i in range(self.num_urdf):
            filename = os.path.basename(self.urdfs_filename[i]) if i < len(self.urdfs_filename) else ""
            if scale_mode == "fixed":
                if special_three_block_mode:
                    if filename.startswith('cube'):
                        scale_value = 0.8
                    elif filename.startswith('cuboid_bar'):
                        scale_value = shared_shape_scales.setdefault('cuboid_bar', 1.0)
                    elif filename.startswith('cone_top'):
                        scale_value = shared_shape_scales.setdefault('cone_top', 1.0)
                    else:
                        scale_value = 1.0
                else:
                    scale_value = fixed_scale_palette[i % len(fixed_scale_palette)]
            elif filename.startswith('cuboid_bar'):
                scale_value = shared_shape_scales.setdefault('cuboid_bar', round(random.uniform(0.9, 1.15), 2))
            elif filename.startswith('cone_top'):
                scale_value = shared_shape_scales.setdefault('cone_top', round(random.uniform(0.75, 1.1), 2))
            else:
                scale_value = round(random.uniform(0.6, 1.2), 2)
            randomized_scales.append(scale_value)
        randomized_yaws = [random.uniform(-math.pi, math.pi) for _ in range(self.num_urdf)]
        cube_rgba_colors = [
            [0.85, 0.20, 0.20, 1.0],
            [0.20, 0.75, 0.25, 1.0],
            [0.20, 0.35, 0.85, 1.0],
            [0.95, 0.80, 0.20, 1.0],
            [0.65, 0.30, 0.80, 1.0]
        ]
        cube_colors = ['红色', '绿色', '蓝色', '黄色', '紫色']

        print(f"随机缩放比例: {randomized_scales}")
        placed_positions = []

        for i in range(self.num_urdf):
            filename = os.path.basename(self.urdfs_filename[i]) if i < len(self.urdfs_filename) else ""
            
            if filename.startswith('cuboid_bar') and special_three_block_mode:
                bar_spawn_region = {
                    "x_min": 0.08,
                    "x_max": 0.18,
                    "y_min": 0.08,
                    "y_max": 0.14,
                    "z": spawn_region["z"]
                }
            elif filename.startswith('cube') and special_three_block_mode:
                bar_spawn_region = {
                    "x_min": -0.18,
                    "x_max": 0.18,
                    "y_min": -0.14,
                    "y_max": -0.06,
                    "z": spawn_region["z"]
                }
            elif random_special_three_block_mode:
                bar_spawn_region = {
                    "x_min": 0.06,
                    "x_max": 0.20,
                    "y_min": -0.14,
                    "y_max": -0.06,
                    "z": spawn_region["z"]
                }
            else:
                bar_spawn_region = spawn_region
            
            for _ in range(220):
                candidate_position = [
                    random.uniform(bar_spawn_region["x_min"], bar_spawn_region["x_max"]),
                    random.uniform(bar_spawn_region["y_min"], bar_spawn_region["y_max"]),
                    bar_spawn_region["z"]
                ]
                if all(
                    ((candidate_position[0] - pos[0]) ** 2 + (candidate_position[1] - pos[1]) ** 2) ** 0.5 >= min_spawn_distance
                    for pos in placed_positions
                ):
                    basePosition = candidate_position
                    break
            else:
                grid_candidates = [
                    [x, y, bar_spawn_region["z"]]
                    for x in np.linspace(bar_spawn_region["x_min"], bar_spawn_region["x_max"], 4)
                    for y in np.linspace(bar_spawn_region["y_min"], bar_spawn_region["y_max"], 3)
                ]
                basePosition = max(
                    grid_candidates,
                    key=lambda candidate: min(
                        [((candidate[0] - pos[0]) ** 2 + (candidate[1] - pos[1]) ** 2) ** 0.5 for pos in placed_positions]
                        or [float("inf")]
                    )
                )

            yaw = randomized_yaws[i] if i < len(randomized_yaws) else random.uniform(-math.pi, math.pi)
            baseOrientation = self.p.getQuaternionFromEuler([0, 0, yaw])

            if i < len(randomized_scales):
                scaling_factor = randomized_scales[i]
            else:
                scaling_factor = 1.0 if scale_mode == "fixed" else round(random.uniform(0.6, 1.2), 2)
            if filename.startswith('cuboid_bar'):
                basePosition[2] = 0.04 * scaling_factor + 0.002
            else:
                basePosition[2] = 0.02 * scaling_factor + 0.002
            urdf_id = self.p.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation, globalScaling=scaling_factor)
            placed_positions.append(basePosition)
            self.p.changeVisualShape(urdf_id, -1, rgbaColor=cube_rgba_colors[i % len(cube_rgba_colors)])
            if filename.startswith('cube'):
                lateral_friction = 2.8
                spinning_friction = 0.05
                rolling_friction = 0.001
            elif filename.startswith('cuboid_bar'):
                lateral_friction = 2.6
                spinning_friction = 0.06
                rolling_friction = 0.001
            else:
                lateral_friction = 1.8
                spinning_friction = 0.02
                rolling_friction = 0.002

            self.p.changeDynamics(
                urdf_id,
                -1,
                lateralFriction=lateral_friction,
                spinningFriction=spinning_friction,
                rollingFriction=rolling_friction,
                restitution=0.0
            )

            # 获取可视形状信息
            inf = self.p.getVisualShapeData(urdf_id)[0]

            if filename.startswith('cone_top'):
                shape_name = '三角体'
            elif filename.startswith('cuboid_bar'):
                shape_name = '细长长方体'
            elif filename.startswith('cylinder'):
                shape_name = '圆柱体'
            else:
                shape_name = '正方体'

            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5])
            self.urdfs_scale.append(scaling_factor)
            self.urdfs_colors.append(cube_colors[i] if i < len(cube_colors) else f'物块{i+1}')
            self.urdfs_shapes.append(shape_name)
            print(
                f"物块{i+1} 已加载: 位置=({basePosition[0]:.3f}, {basePosition[1]:.3f}, {basePosition[2]:.3f}), "
                f"偏航角={yaw:.3f} rad"
            )

        self.obj_ids = self.urdfs_id

    def removeObjsInURDF(self):
        """
        移除objs
        """
        for i in range(self.num_urdf):
            self.p.removeBody(self.urdfs_id[i])

        # 清空 urdfs_id 并更新 obj_ids
        self.urdfs_id = []
        self.obj_ids = self.urdfs_id


    def renderURDFImage(self, save_path):
        """
        渲染图像
        """
        os.makedirs(save_path, exist_ok=True)



        # ======================== 渲染相机深度图 ========================
        print('>> 渲染相机深度图...')
        # 渲染图像
        # 在DIRECT模式下使用软件渲染器，GUI模式下使用硬件渲染器
        try:
            # 先尝试使用硬件渲染器
            img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            # 检查渲染结果是否有效
            dep = img_camera[3]
            depth = np.reshape(dep, (IMAGEHEIGHT, IMAGEWIDTH))
            A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
            B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
            C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
            im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))
            unique_depths = len(np.unique(np.round(im_depthCamera, 6)))
            if unique_depths <= 1:
                # 硬件渲染失败，使用软件渲染器
                print('>> 硬件渲染结果异常，切换到软件渲染器...')
                img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_TINY_RENDERER)
        except Exception as e:
            print(f'>> 硬件渲染异常: {e}，使用软件渲染器...')
            img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_TINY_RENDERER)
        
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        rgba = img_camera[2]    # color data RGB
        dep = img_camera[3]    # depth data
        mask = img_camera[4]    # mask data

        # 获取彩色图像
        im_rgb = np.reshape(rgba, (h, w, 4))[:, :, [2, 1, 0]]
        im_rgb = im_rgb.astype(np.uint8)

        # 获取深度图像
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # 单位 m
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m
        im_depthCamera_rev = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * im_depthCamera.max() - im_depthCamera # 反转深度
        print(
            f'>> depth stats: min={np.nanmin(im_depthCamera):.6f}, '
            f'max={np.nanmax(im_depthCamera):.6f}, '
            f'unique~={len(np.unique(np.round(im_depthCamera, 6)))}'
        )

        # 获取分割图像
        im_mask = np.reshape(mask, (h, w))


        # 保存图像
        # print('>> 保存相机深度图')
        scio.savemat(os.path.join(save_path, 'camera_rgb.mat'), {'A':im_rgb})
        scio.savemat(os.path.join(save_path, 'camera_depth.mat'), {'A':im_depthCamera})
        scio.savemat(os.path.join(save_path, 'camera_depth_rev.mat'), {'A':im_depthCamera_rev})
        scio.savemat(os.path.join(save_path, 'camera_mask.mat'), {'A':im_mask})

        cv2.imwrite(os.path.join(save_path, 'camera_rgb.png'), im_rgb)
        cv2.imwrite(os.path.join(save_path, 'camera_mask.png'), im_mask*20)
        cv2.imwrite(os.path.join(save_path, 'camera_depth.png'), tool.depth2Gray(im_depthCamera))
        cv2.imwrite(os.path.join(save_path, 'camera_depth_rev.png'), tool.depth2Gray(im_depthCamera_rev))

        side_views = [
            ('left', self.leftViewMatrix),
            ('right', self.rightViewMatrix),
        ]
        for view_name, view_matrix in side_views:
            # 在DIRECT模式下，先尝试硬件渲染器，如果失败则使用软件渲染器
            side_camera = None
            try:
                side_camera = self.p.getCameraImage(
                    IMAGEWIDTH,
                    IMAGEHEIGHT,
                    view_matrix,
                    self.projectionMatrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL
                )
                # 检查渲染结果是否有效（不是全黑）
                side_rgba = side_camera[2]
                side_rgb_test = np.reshape(side_rgba, (IMAGEHEIGHT, IMAGEWIDTH, 4))[:, :, :3]
                if np.max(side_rgb_test) < 5:  # 几乎全黑
                    side_camera = None
            except Exception:
                side_camera = None
            
            if side_camera is None:
                side_camera = self.p.getCameraImage(
                    IMAGEWIDTH,
                    IMAGEHEIGHT,
                    view_matrix,
                    self.projectionMatrix,
                    renderer=self.p.ER_TINY_RENDERER
                )
            
            side_width = side_camera[0]
            side_height = side_camera[1]
            side_rgba = side_camera[2]
            side_rgb = np.reshape(side_rgba, (side_height, side_width, 4))[:, :, [2, 1, 0]]
            side_rgb = np.ascontiguousarray(side_rgb.astype(np.uint8))

            mat_path = os.path.join(save_path, f'camera_rgb_{view_name}.mat')
            png_path = os.path.join(save_path, f'camera_rgb_{view_name}.png')
            scio.savemat(mat_path, {'A': side_rgb})
            saved = cv2.imwrite(png_path, side_rgb)
            print(f'>> 侧视相机 {view_name} RGB 保存: {png_path}, success={saved}')

        print('>> 渲染结束')


