import pybullet as p
import numpy as np

from spring_visualizer import SpringVisualizer


class ObjectLauncher:
    def __init__(self, robot_id, link_index, impulse_magnitude=20.0):
        self.spring = SpringVisualizer(robot_id, link_index)
        self.impulse_mag = impulse_magnitude
        self.charge_level = 0.0  # 蓄力比例

    def start_charging(self):
        """开始蓄力"""
        self.charge_level = 0.0

    def update_charging(self, dt):
        """更新蓄力状态"""
        if self.charge_level < 1.0:
            self.charge_level += dt * 2.0  # 每秒充满
            self.spring.update_spring(self.charge_level)

    def launch_object(self, obj_id):
        """执行弹射"""
        if obj_id is None:
            return False

        # 获取机械爪当前方向
        link_state = p.getLinkState(self.spring.robot_id, self.spring.link_index)
        rot_matrix = np.array(p.getMatrixFromQuaternion(link_state[1])).reshape(3, 3)

        # 计算弹射方向（局部坐标系Z轴方向）
        launch_dir = rot_matrix @ np.array([0, 0, 1])
        launch_dir /= np.linalg.norm(launch_dir)

        # 施加冲量（蓄力比例影响力度）
        impulse = launch_dir * self.impulse_mag * self.charge_level
        p.applyExternalForce(
            obj_id,
            -1,
            forceObj=impulse,
            posObj=link_state[0],
            flags=p.WORLD_FRAME
        )

        # 重置弹簧
        self.spring.update_spring(0)
        self.charge_level = 0.0
        return True