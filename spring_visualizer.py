import pybullet as p
import numpy as np


class SpringVisualizer:
    def __init__(self, robot_id, link_index):
        self.robot_id = robot_id
        self.link_index = link_index
        self.spring_id = None
        self.rest_length = 0.15  # 弹簧自然长度
        self.max_compression = 0.3

        # 创建弹簧视觉形状（红色半透明圆柱）
        self.spring_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.02,
            length=self.rest_length,
            rgbaColor=[1, 0, 0, 0.5],  # 红色半透明
            visualFramePosition=[0, 0, -self.rest_length / 2]
        )

        # 创建弹簧刚体（无质量）
        self.spring_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.spring_shape,
            basePosition=[0, 0, 0]
        )

        # 将弹簧固定到机械爪末端
        p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=link_index,
            childBodyUniqueId=self.spring_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )

    def update_spring(self, compression_ratio):
        """动态更新弹簧压缩状态"""
        # 限制压缩比例在0~1之间
        ratio = np.clip(compression_ratio, 0, 1)
        current_length = self.rest_length * (1 - ratio)

        # 更新弹簧视觉属性
        p.changeVisualShape(
            self.spring_id,
            -1,
            rgbaColor=[1, 0.2 * ratio, 0.2 * ratio, 0.7],  # 压缩越大颜色越深
            length=current_length,
            visualFramePosition=[0, 0, -current_length / 2]
        )