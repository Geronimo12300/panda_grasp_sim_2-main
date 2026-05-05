# -*- coding: utf-8 -*-
"""
Panda机械臂仿真控制模块

本模块实现了Franka Panda机械臂在PyBullet仿真环境中的抓取和放置控制。
主要功能包括：
1. 机械臂运动学计算（逆运动学IK）
2. 夹爪控制（抓取、释放）
3. 抓取动作状态机（6个状态）
4. 放置动作状态机（5个状态）
5. 物体物理接触检测
6. 堆叠高度计算
"""

import numpy as np
import math
import pybullet as p

# ======================== 逆运动学配置 ========================
useNullSpace = 1  # 使用零空间优化
ikSolver = 0  # IK求解器类型
pandaEndEffectorIndex = 11  # 末端执行器链接索引
pandaNumDofs = 7  # 机械臂自由度数量

# 关节限位参数
ll = [-7] * pandaNumDofs  # 关节下限
ul = [7] * pandaNumDofs   # 关节上限
jr = [7] * pandaNumDofs   # 关节范围

# 初始关节位置（机械臂默认姿态）
jointPositions = (
    0.8045609285966308,
    0.525471701354679,
    -0.02519566900946519,
    -1.3925086098003587,
    0.013443782914225877,
    1.9178323512245277,
    -0.007207024243406651,
    0.01999436579245478,   # 左夹爪
    0.019977024051412193,  # 右夹爪
)
rp = jointPositions  # 参考姿态

# ======================== 夹爪参数 ========================
GRIPPER_MAX_VELOCITY = 0.25           # 夹爪最大速度
GRIPPER_RELEASE_VELOCITY = 0.08       # 释放时夹爪速度
GRIPPER_RESET_VELOCITY = 0.5          # 重置时夹爪速度
GRIPPER_OPEN_WIDTH = 0.04             # 夹爪完全张开宽度
GRIPPER_CLOSED_WIDTH = 0.0            # 夹爪完全闭合宽度
GRIPPER_PLACE_HOLD_WIDTH = 0.004      # 放置时保持宽度
GRIPPER_GRASP_FORCE = 90              # 抓取力
GRIPPER_RELEASE_FORCE = 45            # 释放力
GRIPPER_RESET_FORCE = 120             # 重置力

# ======================== 机械臂运动参数 ========================
ARM_MOTOR_FORCE = 4 * 240.0           # 机械臂电机力矩
ARM_MAX_VELOCITY = 0.8                # 最大移动速度
ARM_CARRY_MAX_VELOCITY = 0.55         # 搬运时移动速度
# ARM_GRASP_APPROACH_MAX_VELOCITY = 0.50    # 抓取接近速度
# ARM_GRASP_DESCEND_MAX_VELOCITY = 0.15     # 抓取下降速度
# ARM_SMALL_GRASP_APPROACH_MAX_VELOCITY = 0.42  # 小物体抓取接近速度
# RM_SMALL_GRASP_DESCEND_MAX_VELOCITY = 0.12    # 小物体抓取下降速度
ARM_DESCEND_MAX_VELOCITY = 0.38       # 放置下降速度

# ======================== 运输模式配置 ========================
DETERMINISTIC_OBJECT_TRANSPORT = False  # 确定性搬运模式（物体跟随夹爪移动）
FREEZE_PLACED_OBJECTS = False           # 冻结已放置物体

# ======================== 位置容差参数 ========================
EE_XY_TOLERANCE = 0.003               # 末端执行器XY容差
EE_Z_TOLERANCE = 0.005                # 末端执行器Z容差
PLACE_XY_TOLERANCE = 0.015            # 放置XY容差
PLACE_Z_TOLERANCE = 0.020             # 放置Z容差
PLACE_OBJECT_XY_TOLERANCE = 0.008     # 放置物体XY容差
PLACE_GRIPPER_OBJECT_MAX_OFFSET = 0.08    # 夹爪与物体最大偏移
PLACE_OFFSET_COMPENSATION_LIMIT = 0.045   # 偏移补偿限制
PLACE_REGRIP_XY_OFFSET = 0.018        # 重抓XY偏移
PLACE_SUPPORT_XY_MARGIN = 0.026       # 支撑XY边界

# ======================== 抓取参数 ========================
GRASP_CENTER_OBJECT_ON_ATTACH = True              # 附加时居中物体
GRASP_CENTER_ATTACH_MAX_XY_CORRECTION = 0.075     # 最大XY修正
GRASP_CAPTURE_ZONE_XY = 0.022         # 抓取捕获区域XY范围
GRASP_CAPTURE_ZONE_Z_MIN = 0.015      # 抓取捕获区域Z最小值
GRASP_CAPTURE_ZONE_Z_MAX = 0.090      # 抓取捕获区域Z最大值
GRASP_OBJECT_EE_MIN_Z_OFFSET = 0.045  # 物体与末端执行器最小Z偏移
GRASP_OBJECT_EE_MAX_Z_OFFSET = 0.070  # 物体与末端执行器最大Z偏移
GRASP_OBJECT_MIN_BOTTOM_CLEARANCE = 0.006  # 物体底部最小间隙
GRASP_REGRIP_MIN_BOTTOM_Z = 0.055     # 重抓最小底部Z
GRASP_CONSTRAINT_FORCE = 520          # 抓取约束力
GRASP_ATTACH_DELAY = 0.12             # 附加延迟时间
GRASP_CONTACT_TIMEOUT = 0.45          # 接触超时时间
GRASP_RETRY_LOWERING = 0.006          # 重试下降距离
SMALL_OBJECT_GRASP_WIDTH_THRESHOLD = 0.038  # 小物体抓取宽度阈值
SMALL_OBJECT_GRASP_RETRY_LOWERING = 0.003   # 小物体重试下降距离
GRASP_CLOSE_MAX_VELOCITY = 0.08       # 夹爪闭合最大速度
SMALL_OBJECT_GRASP_CLOSE_MAX_VELOCITY = 0.06  # 小物体夹爪闭合速度
GRASP_CONTACT_XY_TOLERANCE = 0.025    # 抓取接触XY容差
GRASP_CONTACT_Z_TOLERANCE = 0.06      # 抓取接触Z容差
GRASP_POSE_ATTACH_XY_TOLERANCE = 0.04 # 姿态附加XY容差
GRASP_POSE_ATTACH_Z_TOLERANCE = 0.09  # 姿态附加Z容差
PHYSICAL_GRASP_LIFT_CLEARANCE = 0.014 # 物理抓取提升间隙
PHYSICAL_GRASP_HOLD_XY_TOLERANCE = 0.04   # 物理抓取保持XY容差
PHYSICAL_GRASP_HOLD_Z_TOLERANCE = 0.11    # 物理抓取保持Z容差

# ======================== 放置参数 ========================
PLACE_SETTLE_DURATION = 0.35          # 放置稳定时间
PLACE_APPROACH_TIMEOUT = 1.2          # 接近超时
PLACE_DESCENT_TIMEOUT = 1.4           # 下降超时
PLACE_RELEASE_TIMEOUT = 1.0           # 释放超时
PLACE_HARD_RELEASE_TIMEOUT = 3.0      # 硬释放超时
PLACE_SLOW_RELEASE_DURATION = 1.2     # 慢速释放持续时间
FIRST_PLACE_EXTRA_CLEARANCE = 0.012   # 首次放置额外间隙
GRIPPER_RELEASE_START_WIDTH = 0.007   # 夹爪释放起始宽度
PLACE_OBJECT_MOVE_DURATION = 0.9      # 物体移动持续时间
PLACE_DETERMINISTIC_APPROACH_DURATION = 0.35   # 确定性接近持续时间
PLACE_DETERMINISTIC_DESCENT_DURATION = 0.35    # 确定性下降持续时间
PLACE_DETERMINISTIC_HOLD_DURATION = 0.12       # 确定性保持持续时间
PLACE_FORCE_RELEASE_XY_TOLERANCE = 0.025    # 强制释放XY容差
PLACE_LOOSE_XY_TOLERANCE = 0.045      # 宽松XY容差
SAFE_CARRY_X_MIN = 0.22               # 安全搬运X最小值

# ======================== 摩擦力参数 ========================
GRIPPER_GRASP_LATERAL_FRICTION = 4.0      # 抓取侧向摩擦力
GRIPPER_GRASP_SPINNING_FRICTION = 0.08    # 抓取旋转摩擦力
GRIPPER_RELEASE_LATERAL_FRICTION = 0.9    # 释放侧向摩擦力
GRIPPER_RELEASE_SPINNING_FRICTION = 0.01  # 释放旋转摩擦力
GRIPPER_RIGHT_RELEASE_LATERAL_FRICTION = 0.45   # 右夹爪释放侧向摩擦力
GRIPPER_RIGHT_RELEASE_SPINNING_FRICTION = 0.005 # 右夹爪释放旋转摩擦力

# ======================== 放置目标位置 ========================
# 键盘按键对应的放置位置 (x, y, z)
PLACE_TARGETS = {
    "7": [0.5, 0.0, 0.4],
    "8": [0.0, 0.5, 0.4],
    "9": [-0.5, 0.0, 0.4],
    "0": [0.5, 0.2, 0.4],
}


class PandaSim(object):
    """
    Panda机械臂仿真类
    
    实现机械臂的基本控制功能，包括：
    - 运动学计算
    - 夹爪控制
    - 抓取/放置状态机
    """
    
    def __init__(self, bullet_client, offset):
        """
        初始化机械臂
        
        参数:
            bullet_client: PyBullet客户端
            offset: 机械臂位置偏移
        """
        self.p = bullet_client
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)

        # 加载机械臂URDF模型
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn = [0, 0, 0, 1]
        self.panda = self.p.loadURDF(
            "franka_panda/panda_1.urdf",
            np.array([0, 0, 0]) + self.offset,
            orn,
            useFixedBase=True,
            flags=flags,
        )

        # 设置夹爪动力学参数
        for i in [9, 10]:  # 左右夹爪链接索引
            self.p.changeDynamics(self.panda, i, mass=0.2)
            self.p.changeDynamics(
                self.panda,
                i,
                lateralFriction=4.0,
                spinningFriction=0.08,
                rollingFriction=0.003,
                restitution=0.0,
            )
            self.p.changeDynamics(self.panda, i, collisionMargin=0.001)

        # 初始化状态变量
        index = 0
        self.state = 0
        self.control_dt = 1.0 / 240.0  # 控制周期（240Hz）
        self.finger_target = 0
        self.gripper_height = 0.2
        self.pose_controlled_states = {1, 2, 4, 5, 7, 8, 9, 10, 11}  # 姿态控制状态集合
        self.current_place_char = None
        self.active_grasp_constraint = None
        self.held_object_id = None
        self.stack_anchor = None
        self.place_target_snapshot = None
        self.place_target_override = None
        self.release_snapped = False
        self.release_open_start_t = None
        self.place_animation = None
        self.release_wait_warned = False
        self.motion_stall_counter = 0
        self.last_ee_pos = None
        self.gripper_contact_mode = None
        self.is_special_experiment = False  # 非常规实验标志

        # 创建夹爪同步约束（左右夹爪镜像运动）
        c = self.p.createConstraint(
            self.panda,
            9,
            self.panda,
            10,
            jointType=self.p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=100)

        # 初始化关节状态
        for j in range(self.p.getNumJoints(self.panda)):
            self.p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.panda, j)
            jointType = info[2]
            if jointType == self.p.JOINT_PRISMATIC:
                self.p.resetJointState(self.panda, j, jointPositions[index])
                index += 1
            if jointType == self.p.JOINT_REVOLUTE:
                self.p.resetJointState(self.panda, j, jointPositions[index])
                index += 1

        self.t = 0.0
        self.set_gripper_contact_mode("release")

    def reset_robot_joints_to_initial(self):
        """
        将机械臂关节重置到初始位置
        """
        index = 0
        for j in range(self.p.getNumJoints(self.panda)):
            info = self.p.getJointInfo(self.panda, j)
            joint_type = info[2]
            if joint_type in (self.p.JOINT_PRISMATIC, self.p.JOINT_REVOLUTE):
                target = jointPositions[index]
                self.p.resetJointState(self.panda, j, target)
                self.p.setJointMotorControl2(
                    self.panda,
                    j,
                    self.p.POSITION_CONTROL,
                    target,
                    force=ARM_MOTOR_FORCE,
                    maxVelocity=0.8,
                )
                index += 1

    def set_gripper_contact_mode(self, mode):
        """
        设置夹爪接触模式
        
        参数:
            mode: "grasp"（抓取模式，高摩擦力）或 "release"（释放模式，低摩擦力）
        """
        if mode == self.gripper_contact_mode:
            return

        if mode == "grasp":
            # 抓取模式：高摩擦力，防止物体滑落
            friction_by_link = {
                9: (GRIPPER_GRASP_LATERAL_FRICTION, GRIPPER_GRASP_SPINNING_FRICTION),
                10: (GRIPPER_GRASP_LATERAL_FRICTION, GRIPPER_GRASP_SPINNING_FRICTION),
            }
        else:
            # 释放模式：低摩擦力，便于物体脱离
            friction_by_link = {
                9: (GRIPPER_RELEASE_LATERAL_FRICTION, GRIPPER_RELEASE_SPINNING_FRICTION),
                10: (GRIPPER_RIGHT_RELEASE_LATERAL_FRICTION, GRIPPER_RIGHT_RELEASE_SPINNING_FRICTION),
            }

        for i in [9, 10]:
            lateral_friction, spinning_friction = friction_by_link[i]
            self.p.changeDynamics(
                self.panda,
                i,
                lateralFriction=lateral_friction,
                spinningFriction=spinning_friction,
                rollingFriction=0.003,
                restitution=0.0,
            )
        self.gripper_contact_mode = mode

    def reset_gripper_open_immediately(self):
        """
        立即打开夹爪（不等待物理仿真）
        """
        self.set_gripper_contact_mode("release")
        for i in [9, 10]:
            self.p.resetJointState(self.panda, i, GRIPPER_OPEN_WIDTH)
        self.setGripper(
            GRIPPER_OPEN_WIDTH,
            force=GRIPPER_RESET_FORCE,
            max_velocity=GRIPPER_RESET_VELOCITY,
        )

    def clear_grasp_contact_state(self, settle_steps=80):
        """
        清除抓取接触状态
        
        参数:
            settle_steps: 稳定步数
        """
        self.detach_held_object()
        self.set_gripper_contact_mode("release")
        self.reset_gripper_open_immediately()
        for _ in range(settle_steps):
            self.setGripper(
                GRIPPER_OPEN_WIDTH,
                force=GRIPPER_RELEASE_FORCE,
                max_velocity=GRIPPER_RESET_VELOCITY,
            )
            self.p.stepSimulation()

    def reset_to_initial_pose(self, settle_steps=80):
        """
        重置到初始姿态
        
        参数:
            settle_steps: 稳定步数
        """
        self.clear_grasp_contact_state(settle_steps=settle_steps)
        self.reset_robot_joints_to_initial()
        self.state = 0
        self.state_t = 0
        self.cur_state = 0
        self.place_position = None
        self.current_place_char = None
        self.place_target_snapshot = None
        self.place_target_override = None
        self.release_snapped = False
        self.release_open_start_t = None
        self.place_animation = None
        self.release_wait_warned = False
        self.motion_stall_counter = 0
        self.last_ee_pos = None
        self.place_count = 0
        self.placed_objects = []
        self.stack_center = None
        self.stack_anchor = None
        self.is_special_experiment = False

    def attach_held_object(self, held_object_id):
        """
        附加被抓取物体（建立物理约束）
        
        参数:
            held_object_id: 被抓取物体的ID
        """
        if held_object_id is None:
            return

        self.active_grasp_constraint = None
        self.held_object_id = held_object_id
        self.set_gripper_contact_mode("grasp")
        self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)

    def object_is_in_gripper_capture_zone(self, target_object_id):
        """
        检查物体是否在夹爪捕获区域内
        
        参数:
            target_object_id: 目标物体ID
            
        返回:
            bool: 物体是否在捕获区域内
        """
        if target_object_id is None:
            return False

        link_state = self.p.getLinkState(self.panda, pandaEndEffectorIndex)
        ee_pos = link_state[4]
        ee_orn = link_state[5]
        obj_pos, _ = self.p.getBasePositionAndOrientation(target_object_id)
        inv_pos, inv_orn = self.p.invertTransform(ee_pos, ee_orn)
        local_pos, _ = self.p.multiplyTransforms(inv_pos, inv_orn, obj_pos, [0, 0, 0, 1])
        return (
            abs(float(local_pos[0])) <= GRASP_CAPTURE_ZONE_XY
            and abs(float(local_pos[1])) <= GRASP_CAPTURE_ZONE_XY
            and GRASP_CAPTURE_ZONE_Z_MIN <= abs(float(local_pos[2])) <= GRASP_CAPTURE_ZONE_Z_MAX
        )

    def get_held_object_bottom_z(self, held_object_id):
        """
        获取被抓取物体底部Z坐标
        
        参数:
            held_object_id: 被抓取物体ID
            
        返回:
            float: 物体底部Z坐标
        """
        if held_object_id is None:
            return 0.0
        aabb_min, _ = self.p.getAABB(held_object_id)
        return float(aabb_min[2])

    def recenter_held_object_in_gripper(self, held_object_id, reason=""):
        """
        将被抓取物体在夹爪中居中（当前未实现）
        
        参数:
            held_object_id: 被抓取物体ID
            reason: 重居中原因
        """
        return False

    def set_place_target_override(self, place_pose=None):
        """
        设置放置目标覆盖参数
        
        参数:
            place_pose: 放置位姿字典，包含x, y, z, layer_index, slot等
        """
        if not place_pose:
            self.place_target_override = None
            return

        self.place_target_override = {
            "x": float(place_pose.get("x", 0.5)),
            "y": float(place_pose.get("y", 0.0)),
            "z": float(place_pose.get("z", 0.0)),
            "layer_index": int(place_pose.get("layer_index", 0)),
            "slot": str(place_pose.get("slot", "center")),
            "place_hold_width": float(place_pose.get("place_hold_width", GRIPPER_CLOSED_WIDTH)),
            "approach_height": float(place_pose.get("approach_height", 0.4)),
            "approach_clearance": float(place_pose.get("approach_clearance", 0.08)),
            "retreat_height": float(place_pose.get("retreat_height", 0.4)),
            "retreat_lift_delta": float(place_pose.get("retreat_lift_delta", 0.0)),
        }

    def reset_motion_watchdog(self):
        """
        重置运动监控计数器
        """
        self.motion_stall_counter = 0
        self.last_ee_pos = None

    def detach_held_object(self):
        """
        分离被抓取物体（移除物理约束）
        """
        if self.active_grasp_constraint is not None:
            self.p.removeConstraint(self.active_grasp_constraint)
            self.active_grasp_constraint = None
        self.held_object_id = None

    def object_is_physically_held(self, held_object_id, require_lift=False):
        """
        检查物体是否被物理夹持
        
        参数:
            held_object_id: 被抓取物体ID
            require_lift: 是否要求物体已提升
            
        返回:
            bool: 物体是否被物理夹持
        """
        if held_object_id is None:
            return False

        left_contact, right_contact, contact_count = self.get_target_contact_summary(held_object_id)
        if contact_count == 0:
            return False

        is_cylinder = self.is_cylinder_object(held_object_id)
        has_stable_contact = (
            contact_count >= 1
            if is_cylinder
            else (left_contact and right_contact) or contact_count >= 2
        )
        if not has_stable_contact:
            return False

        ee_pos = self.get_end_effector_position()
        obj_pos, _ = self.p.getBasePositionAndOrientation(held_object_id)
        xy_error = np.linalg.norm(np.array(obj_pos[:2], dtype=float) - ee_pos[:2])
        z_error = abs(float(obj_pos[2]) - float(ee_pos[2]))
        if xy_error > PHYSICAL_GRASP_HOLD_XY_TOLERANCE or z_error > PHYSICAL_GRASP_HOLD_Z_TOLERANCE:
            return False

        if require_lift and self.get_held_object_bottom_z(held_object_id) <= PHYSICAL_GRASP_LIFT_CLEARANCE:
            return False

        return True

    def refresh_physical_hold_state(self, held_object_id, require_lift=False):
        """
        刷新物理夹持状态
        
        参数:
            held_object_id: 被抓取物体ID
            require_lift: 是否要求物体已提升
        """
        if self.object_is_physically_held(held_object_id, require_lift=require_lift):
            self.held_object_id = held_object_id
            return True

        if self.held_object_id == held_object_id:
            self.held_object_id = None
        return False

    def calcJointLocation(self, pos, orn):
        """
        计算逆运动学（IK）- 根据末端执行器目标位置计算关节角度
        
        参数:
            pos: 目标位置 [x, y, z]
            orn: 目标朝向（四元数）
            
        返回:
            list: 关节角度列表
        """
        joint_poses = self.p.calculateInverseKinematics(
            self.panda,
            pandaEndEffectorIndex,
            pos,
            orn,
            ll,
            ul,
            jr,
            rp,
            maxNumIterations=80,
            residualThreshold=1e-5,
        )
        if joint_poses is None or len(joint_poses) < pandaNumDofs:
            print(f"      [IK警告] 逆运动学计算失败或结果无效: pos={pos}")
        return joint_poses

    def get_target_contact_summary(self, target_object_id):
        """
        获取目标物体与夹爪的接触摘要
        
        参数:
            target_object_id: 目标物体ID
            
        返回:
            tuple: (左夹爪是否接触, 右夹爪是否接触, 总接触点数)
        """
        if target_object_id is None:
            return False, False, 0

        left_contacts = self.p.getContactPoints(self.panda, target_object_id, linkIndexA=9)
        right_contacts = self.p.getContactPoints(self.panda, target_object_id, linkIndexA=10)
        return bool(left_contacts), bool(right_contacts), len(left_contacts) + len(right_contacts)

    def is_cylinder_object(self, target_object_id):
        """
        检查物体是否为圆柱体
        
        参数:
            target_object_id: 目标物体ID
            
        返回:
            bool: 是否为圆柱体
        """
        visual_shape_data = self.p.getVisualShapeData(target_object_id)
        if not visual_shape_data:
            return False

        geom_type = visual_shape_data[0][2]
        if geom_type == self.p.GEOM_CYLINDER:
            return True

        mesh_name = visual_shape_data[0][4]
        if isinstance(mesh_name, bytes):
            mesh_name = mesh_name.decode("utf-8", errors="ignore")
        return bool(mesh_name and "cylinder" in mesh_name.lower())

    def get_end_effector_position(self):
        """
        获取末端执行器当前位置
        
        返回:
            np.array: 末端执行器位置 [x, y, z]
        """
        link_state = self.p.getLinkState(self.panda, pandaEndEffectorIndex)
        return np.array(link_state[4], dtype=float)

    def can_attach_grasp_target(self, target_object_id):
        """
        检查是否可以附加抓取目标（接触检测）
        
        参数:
            target_object_id: 目标物体ID
            
        返回:
            bool: 是否可以附加
        """
        if target_object_id is None:
            return False

        left_contact, right_contact, contact_count = self.get_target_contact_summary(target_object_id)
        if contact_count == 0:
            return False

        ee_pos = self.get_end_effector_position()
        obj_pos, _ = self.p.getBasePositionAndOrientation(target_object_id)
        xy_error = np.linalg.norm(np.array(obj_pos[:2], dtype=float) - ee_pos[:2])
        z_error = abs(float(obj_pos[2]) - float(ee_pos[2]))
        is_cylinder = self.is_cylinder_object(target_object_id)
        
        # 对于小物体，只要有接触就算稳定（放宽要求）
        visual_shape_data = self.p.getVisualShapeData(target_object_id)
        is_small_object = False
        if visual_shape_data:
            mesh_name = visual_shape_data[0][4]
            if isinstance(mesh_name, bytes):
                mesh_name = mesh_name.decode("utf-8", errors="ignore")
            if mesh_name and "cube" in mesh_name.lower():
                aabb_min, aabb_max = self.p.getAABB(target_object_id)
                size_z = aabb_max[2] - aabb_min[2]
                if size_z <= 0.04:  # 小正方体
                    is_small_object = True
        
        has_stable_contact = (
            contact_count >= 1
            if (is_cylinder or is_small_object)
            else (left_contact and right_contact) or contact_count >= 2
        )
        xy_tolerance = 0.035 if is_cylinder else GRASP_CONTACT_XY_TOLERANCE
        z_tolerance = 0.075 if is_cylinder else GRASP_CONTACT_Z_TOLERANCE
        
        # 调试输出
        if self.state_t > 0.3 and int(self.state_t * 10) % 5 == 0:  # 每0.5秒输出一次
            print(f"    [接触检测] 左接触={left_contact}, 右接触={right_contact}, 接触数={contact_count}, "
                  f"XY误差={xy_error:.4f}m(阈值{xy_tolerance}), Z误差={z_error:.4f}m(阈值{z_tolerance}), "
                  f"稳定接触={has_stable_contact}, 小物体={is_small_object}")
        
        return (
            has_stable_contact
            and xy_error <= xy_tolerance
            and z_error <= z_tolerance
        )

    def can_attach_grasp_target_by_pose(self, target_object_id, target_pos):
        """
        根据姿态检查是否可以附加抓取目标
        
        参数:
            target_object_id: 目标物体ID
            target_pos: 目标位置
            
        返回:
            bool: 是否可以附加
        """
        if target_object_id is None:
            return False

        ee_pos = self.get_end_effector_position()
        obj_pos, _ = self.p.getBasePositionAndOrientation(target_object_id)
        target = np.array(target_pos, dtype=float)
        obj_xy_error = np.linalg.norm(np.array(obj_pos[:2], dtype=float) - target[:2])
        ee_xy_error = np.linalg.norm(np.array(obj_pos[:2], dtype=float) - ee_pos[:2])
        ee_z_error = abs(float(ee_pos[2]) - float(target[2]))
        return (
            obj_xy_error <= GRASP_POSE_ATTACH_XY_TOLERANCE
            and ee_xy_error <= GRASP_POSE_ATTACH_XY_TOLERANCE
            and ee_z_error <= GRASP_POSE_ATTACH_Z_TOLERANCE
        )

    def get_held_object_release_height(self, held_object_id, support_top_z):
        """
        计算被抓取物体的释放高度
        
        参数:
            held_object_id: 被抓取物体ID
            support_top_z: 支撑面顶部Z坐标
            
        返回:
            float: 释放高度
        """
        base_clearance = 0.008
        if support_top_z <= 1e-6:
            base_clearance += FIRST_PLACE_EXTRA_CLEARANCE
        if held_object_id is None:
            return support_top_z + 0.09

        held_aabb = self.p.getAABB(held_object_id)
        ee_pos = self.get_end_effector_position()
        ee_to_object_bottom = ee_pos[2] - held_aabb[0][2]

        extra_clearance = 0.0
        visual_shape_data = self.p.getVisualShapeData(held_object_id)
        if visual_shape_data:
            geom_type = visual_shape_data[0][2]
            if geom_type == self.p.GEOM_CYLINDER:
                extra_clearance = 0.005
            elif geom_type == self.p.GEOM_MESH:
                mesh_name = visual_shape_data[0][4]
                if isinstance(mesh_name, bytes):
                    mesh_name = mesh_name.decode("utf-8", errors="ignore")
                if mesh_name and "cone_top" in mesh_name:
                    extra_clearance = 0.008

        min_release_offset = 0.05
        release_offset = max(ee_to_object_bottom, min_release_offset)
        return support_top_z + release_offset + base_clearance + extra_clearance

    def has_reached_position(self, target_pos, xy_tolerance=EE_XY_TOLERANCE, z_tolerance=EE_Z_TOLERANCE):
        """
        检查是否到达目标位置
        
        参数:
            target_pos: 目标位置
            xy_tolerance: XY容差
            z_tolerance: Z容差
            
        返回:
            bool: 是否到达
        """
        current_pos = self.get_end_effector_position()
        target = np.array(target_pos, dtype=float)
        delta = np.abs(current_pos - target)
        return delta[0] <= xy_tolerance and delta[1] <= xy_tolerance and delta[2] <= z_tolerance

    def has_reached_xy(self, target_pos, xy_tolerance=PLACE_FORCE_RELEASE_XY_TOLERANCE):
        """
        检查是否到达目标XY位置
        
        参数:
            target_pos: 目标位置
            xy_tolerance: XY容差
            
        返回:
            bool: 是否到达
        """
        current_pos = self.get_end_effector_position()
        target = np.array(target_pos, dtype=float)
        return np.linalg.norm(current_pos[:2] - target[:2]) <= xy_tolerance

    def get_held_object_offset_from_ee(self, held_object_id):
        """
        获取被抓取物体相对于末端执行器的偏移
        
        参数:
            held_object_id: 被抓取物体ID
            
        返回:
            np.array: 偏移向量
        """
        if held_object_id is None:
            return np.zeros(3, dtype=float)

        ee_pos = self.get_end_effector_position()
        obj_pos, _ = self.p.getBasePositionAndOrientation(held_object_id)
        return np.array(obj_pos, dtype=float) - ee_pos

    def get_object_centered_ee_target(self, target_xy, target_ee_z, held_object_id):
        """
        获取物体中心的末端执行器目标位置（考虑物体偏移补偿）
        
        参数:
            target_xy: 目标XY位置
            target_ee_z: 目标Z高度
            held_object_id: 被抓取物体ID
            
        返回:
            list: 补偿后的目标位置
        """
        offset = self.get_held_object_offset_from_ee(held_object_id)
        if np.linalg.norm(offset[:2]) > PLACE_GRIPPER_OBJECT_MAX_OFFSET:
            offset[:2] = 0.0
        elif np.linalg.norm(offset[:2]) > PLACE_OFFSET_COMPENSATION_LIMIT:
            xy_norm = np.linalg.norm(offset[:2])
            if xy_norm > 1e-6:
                offset[:2] = offset[:2] / xy_norm * PLACE_OFFSET_COMPENSATION_LIMIT

        return [
            float(target_xy[0] - offset[0]),
            float(target_xy[1] - offset[1]),
            float(target_ee_z),
        ]

    def held_object_is_near_gripper(self, held_object_id):
        """
        检查被抓取物体是否在夹爪附近
        
        参数:
            held_object_id: 被抓取物体ID
            
        返回:
            bool: 是否在夹爪附近
        """
        if held_object_id is None:
            return False

        offset = self.get_held_object_offset_from_ee(held_object_id)
        return (
            np.linalg.norm(offset[:2]) <= PLACE_GRIPPER_OBJECT_MAX_OFFSET
            and abs(float(offset[2])) <= 0.18
        )

    def held_object_is_over_stack(self, held_object_id, target_xy, xy_tolerance=PLACE_OBJECT_XY_TOLERANCE):
        """
        检查被抓取物体是否在堆叠位置上方
        
        参数:
            held_object_id: 被抓取物体ID
            target_xy: 目标XY位置
            xy_tolerance: XY容差
            
        返回:
            bool: 是否在堆叠位置上方
        """
        if held_object_id is None:
            return False

        obj_pos, _ = self.p.getBasePositionAndOrientation(held_object_id)
        return np.linalg.norm(np.array(obj_pos[:2], dtype=float) - np.array(target_xy, dtype=float)) <= xy_tolerance

    def placement_pose_is_ready(self, ee_target, held_object_id, target_xy):
        """
        检查放置姿态是否就绪（V12版本逻辑）
        
        检查三个条件：
        1. 末端执行器到达目标位置
        2. 物体在夹爪附近
        3. 物体在堆叠位置上方
        
        参数:
            ee_target: 末端执行器目标位置
            held_object_id: 被抓取物体ID
            target_xy: 目标XY位置
            
        返回:
            bool: 是否就绪
        """
        return (
            self.has_reached_position(ee_target, PLACE_XY_TOLERANCE, PLACE_Z_TOLERANCE)
            and self.held_object_is_near_gripper(held_object_id)
            and self.held_object_is_over_stack(held_object_id, target_xy)
        )

    def get_place_hold_width(self):
        """
        获取放置保持宽度
        
        返回:
            float: 放置保持宽度
        """
        requested_width = GRIPPER_PLACE_HOLD_WIDTH
        if self.place_target_override:
            requested_width = float(self.place_target_override.get("place_hold_width", GRIPPER_PLACE_HOLD_WIDTH))
        return float(np.clip(requested_width, GRIPPER_PLACE_HOLD_WIDTH, 0.012))

    def get_slow_release_width(self, elapsed=None, start_width=None):
        """
        获取慢速释放宽度
        
        参数:
            elapsed: 已过时间
            start_width: 起始宽度
            
        返回:
            float: 当前目标宽度
        """
        release_elapsed = self.state_t if elapsed is None else elapsed
        progress = np.clip(release_elapsed / PLACE_SLOW_RELEASE_DURATION, 0.0, 1.0)
        base_width = self.get_place_hold_width() if start_width is None else start_width
        return base_width + progress * (GRIPPER_OPEN_WIDTH - base_width)

    def open_gripper_for_release(self, slow=False, elapsed=None, start_width=None):
        """
        打开夹爪进行释放
        
        参数:
            slow: 是否慢速释放
            elapsed: 已过时间
            start_width: 起始宽度
        """
        self.set_gripper_contact_mode("release")
        finger_target = self.get_slow_release_width(elapsed, start_width=start_width) if slow else GRIPPER_OPEN_WIDTH
        self.setGripper(
            finger_target,
            force=GRIPPER_RELEASE_FORCE,
            max_velocity=GRIPPER_RELEASE_VELOCITY,
        )

    def force_release_held_object(self):
        """
        强制释放被抓取物体
        """
        self.held_object_id = None
        self.set_gripper_contact_mode("release")
        self.reset_gripper_open_immediately()

    def get_stack_support_top_z(self, target_xy=None, exclude_object_id=None):
        """
        获取堆叠支撑面顶部Z坐标
        
        参数:
            target_xy: 目标XY位置
            exclude_object_id: 排除的物体ID
            
        返回:
            float: 支撑面顶部Z坐标
        """
        support_top_z = 0.0
        for obj_id in self.placed_objects:
            if obj_id == exclude_object_id:
                continue
            aabb = self.p.getAABB(obj_id)
            if target_xy is not None:
                x_min = aabb[0][0] - PLACE_SUPPORT_XY_MARGIN
                x_max = aabb[1][0] + PLACE_SUPPORT_XY_MARGIN
                y_min = aabb[0][1] - PLACE_SUPPORT_XY_MARGIN
                y_max = aabb[1][1] + PLACE_SUPPORT_XY_MARGIN
                if not (x_min <= target_xy[0] <= x_max and y_min <= target_xy[1] <= y_max):
                    continue
            support_top_z = max(support_top_z, aabb[1][2])
        return support_top_z

    def get_stack_target_pose(self, held_object_id):
        """
        获取堆叠目标位姿
        
        参数:
            held_object_id: 被抓取物体ID
            
        返回:
            tuple: (目标位置, 目标朝向)
        """
        if held_object_id is None or self.place_position is None:
            return None, None

        support_top_z = self.get_stack_support_top_z(target_xy=self.place_position[:2], exclude_object_id=held_object_id)
        aabb_min, aabb_max = self.p.getAABB(held_object_id)
        object_height = max(aabb_max[2] - aabb_min[2], 0.02)
        _, obj_orn = self.p.getBasePositionAndOrientation(held_object_id)
        obj_yaw = self.p.getEulerFromQuaternion(obj_orn)[2]
        upright_orn = self.p.getQuaternionFromEuler([0, 0, obj_yaw])
        target_pos = [
            self.place_position[0],
            self.place_position[1],
            support_top_z + object_height / 2.0 + 0.003,
        ]
        return target_pos, upright_orn

    def snap_object_to_stack(self, held_object_id):
        """
        将物体吸附到堆叠位置（未实现）
        """
        return False

    def update_gripper_place_animation(self, held_object_id):
        """
        更新夹爪放置动画（未实现）
        """
        return False

    def advance_state(self):
        """
        推进到下一个状态
        """
        self.cur_state += 1
        if self.cur_state >= len(self.states):
            self.cur_state = 0
        self.state_t = 0
        self.state = self.states[self.cur_state]
        self.reset_motion_watchdog()

    def setArm(self, jointPoses, max_velocity=ARM_MAX_VELOCITY):
        """
        设置机械臂关节位置
        
        参数:
            jointPoses: 关节角度列表
            max_velocity: 最大速度
        """
        if jointPoses is None or len(jointPoses) < pandaNumDofs:
            print(f"    [setArm错误] 关节位置无效: {jointPoses}")
            return
        for i in range(pandaNumDofs):
            self.p.setJointMotorControl2(
                self.panda,
                i,
                self.p.POSITION_CONTROL,
                jointPoses[i],
                force=ARM_MOTOR_FORCE,
                maxVelocity=max_velocity,
            )

    def setGripper(self, finger_target, force=GRIPPER_GRASP_FORCE, max_velocity=GRIPPER_MAX_VELOCITY):
        """
        设置夹爪宽度
        
        参数:
            finger_target: 目标宽度
            force: 夹爪力
            max_velocity: 最大速度
        """
        for i in [9, 10]:
            self.p.setJointMotorControl2(
                self.panda,
                i,
                self.p.POSITION_CONTROL,
                finger_target,
                force=force,
                maxVelocity=max_velocity,
            )

    def is_small_grasp_target(self, gripper_w):
        """
        检查是否为小物体抓取目标
        
        参数:
            gripper_w: 夹爪宽度
            
        返回:
            bool: 是否为小物体
        """
        return gripper_w <= SMALL_OBJECT_GRASP_WIDTH_THRESHOLD

    def grasp_step(self, pos, angle, gripper_w, held_object_id=None):
        """
        抓取步骤状态机
        
        状态说明:
        - 状态0: 移动到安全位置
        - 状态1: 移动到物体上方（接近位置）
        - 状态2: 下降到抓取位置
        - 状态3: 闭合夹爪，检测接触
        - 状态4: 提升物体
        - 状态5: 撤退到安全高度
        
        参数:
            pos: 抓取位置 [x, y, z]
            angle: 抓取角度
            gripper_w: 夹爪宽度
            held_object_id: 被抓取物体ID
            
        返回:
            bool: 抓取是否完成
        """
        self.update_state()
        target_pos = list(pos)
        target_pos[2] += 0.047  # 末端执行器偏移
        orn = self.p.getQuaternionFromEuler([math.pi, 0.0, angle + math.pi / 2])

        # 状态0: 移动到安全位置
        if self.state == 0:
            safe_pos = [0.5, 0, 0.4]
            safe_orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(safe_pos, safe_orn))
            return False

        # 状态1: 移动到物体上方
        if self.state == 1:
            approach_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
            self.setArm(self.calcJointLocation(approach_pos, orn))
            self.setGripper(gripper_w)
            if self.has_reached_position(approach_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        # 状态2: 下降到抓取位置
        if self.state == 2:
            self.setArm(self.calcJointLocation(target_pos, orn))
            if self.has_reached_position(target_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        # 状态3: 闭合夹爪，检测接触
        if self.state == 3:
            squeeze_pos = list(target_pos)
            if self.state_t >= GRASP_ATTACH_DELAY + 0.12:
                squeeze_pos[2] -= GRASP_RETRY_LOWERING
            self.setArm(self.calcJointLocation(squeeze_pos, orn))
            self.set_gripper_contact_mode("grasp")
            self.setGripper(0)
            
            can_attach = self.can_attach_grasp_target(held_object_id)
            timeout_reached = self.state_t > self.state_durations[self.cur_state] + GRASP_CONTACT_TIMEOUT
            
            if int(self.state_t * 10) % 3 == 0:
                print(f"    [State3] state_t={self.state_t:.2f}s, can_attach={can_attach}, timeout={timeout_reached}, "
                      f"delay_ok={self.state_t >= GRASP_ATTACH_DELAY}")
            
            if can_attach:
                print(f"    [抓取成功] 检测到稳定接触，准备附加物体")
                self.attach_held_object(held_object_id)
                self.advance_state()
            elif timeout_reached:
                print(f"    [抓取超时] state_t={self.state_t:.2f}s, 超时阈值={self.state_durations[self.cur_state] + GRASP_CONTACT_TIMEOUT:.2f}s")
                print("抓取失败：夹爪未形成稳定接触，本次按空抓处理。")
                self.advance_state()
            return False

        # 状态4: 提升物体
        if self.state == 4:
            lift_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.05]
            self.setArm(self.calcJointLocation(lift_pos, orn))
            self.set_gripper_contact_mode("grasp")
            self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
            if self.has_reached_position(lift_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        # 状态5: 撤退到安全高度
        if self.state == 5:
            retreat_pos = [target_pos[0], target_pos[1], 0.4]
            self.setArm(self.calcJointLocation(retreat_pos, orn))
            self.set_gripper_contact_mode("grasp")
            self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
            if self.has_reached_position(retreat_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                return True
            return False

        return False

    def place_step(self, char, held_object_id=None):
        """
        放置步骤状态机（V12版本逻辑）
        
        状态说明:
        - 状态7: 移动到放置位置上方（接近位置）
        - 状态8: 下降到释放高度
        - 状态9: 保持位置稳定
        - 状态10: 慢速打开夹爪释放物体
        - 状态11: 撤退到安全高度
        
        参数:
            char: 放置目标按键 ('7', '8', '9', '0')
            held_object_id: 被抓取物体ID
            
        返回:
            bool: 放置是否完成
        """
        self.update_state()
        
        # 状态7: 移动到放置位置上方
        if self.state == 7:
            self.recenter_held_object_in_gripper(held_object_id, reason="放置上方移动前检测到偏心")
            if char not in PLACE_TARGETS:
                return False

            pos = PLACE_TARGETS[char]
            self.current_place_char = char
            # 获取放置位置（优先使用覆盖参数）
            if self.place_target_override:
                place_x = self.place_target_override["x"]
                place_y = self.place_target_override["y"]
                slot = self.place_target_override.get("slot", "center")
                layer_index = self.place_target_override.get("layer_index", 0)
                if slot in {"front", "back", "left", "right"}:
                    # 非常规实验：底层特殊槽位
                    self.is_special_experiment = True
                    self.stack_center = [place_x, place_y]
                    self.stack_anchor = None
                    self.place_position = [place_x, place_y, 0]
                    self.place_target_snapshot = [place_x, place_y]
                    target_xy = [place_x, place_y]
                    approach_pos = self.get_object_centered_ee_target(target_xy, pos[2], held_object_id)
                    orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
                    self.setArm(self.calcJointLocation(approach_pos, orn), max_velocity=ARM_CARRY_MAX_VELOCITY)
                    self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
                    reached_approach = self.placement_pose_is_ready(approach_pos, held_object_id, target_xy)
                    deterministic_ready = (
                        DETERMINISTIC_OBJECT_TRANSPORT
                        and self.state_t >= PLACE_DETERMINISTIC_APPROACH_DURATION
                    )
                    if reached_approach or deterministic_ready:
                        self.advance_state()
                    return False
                elif slot == "center" and layer_index > 0 and self.is_special_experiment:
                    # 非常规实验：上层物块需要计算正确的接近高度
                    support_top_z = 0.0
                    for obj_id in self.placed_objects:
                        if obj_id == held_object_id:
                            continue
                        aabb = self.p.getAABB(obj_id)
                        support_top_z = max(support_top_z, aabb[1][2])
                    approach_height = pos[2] + support_top_z + 0.02
                    self.stack_center = [place_x, place_y]
                    self.stack_anchor = [place_x, place_y]
                    self.place_position = [place_x, place_y, 0]
                    self.place_target_snapshot = [place_x, place_y]
                    target_xy = [place_x, place_y]
                    approach_pos = self.get_object_centered_ee_target(target_xy, approach_height, held_object_id)
                    orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
                    self.setArm(self.calcJointLocation(approach_pos, orn), max_velocity=ARM_CARRY_MAX_VELOCITY)
                    self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
                    reached_approach = self.placement_pose_is_ready(approach_pos, held_object_id, target_xy)
                    deterministic_ready = (
                        DETERMINISTIC_OBJECT_TRANSPORT
                        and self.state_t >= PLACE_DETERMINISTIC_APPROACH_DURATION
                    )
                    if reached_approach or deterministic_ready:
                        self.advance_state()
                    return False
            else:
                place_x = pos[0]
                place_y = pos[1]
            
            # 初始化堆叠中心
            if self.stack_center is None or self.place_count == 0:
                self.stack_center = [place_x, place_y]
                self.stack_anchor = [place_x, place_y]
            elif self.stack_anchor is not None:
                self.stack_center = [self.stack_anchor[0], self.stack_anchor[1]]

            self.place_position = [self.stack_center[0], self.stack_center[1], 0]
            self.place_target_snapshot = [self.stack_center[0], self.stack_center[1]]
            target_xy = [self.place_position[0], self.place_position[1]]
            
            # 使用动态补偿计算目标位置
            approach_pos = self.get_object_centered_ee_target(target_xy, pos[2], held_object_id)
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(approach_pos, orn), max_velocity=ARM_CARRY_MAX_VELOCITY)
            self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
            
            # 检查是否到达目标位置（使用完整的3条件检查）
            reached_approach = self.placement_pose_is_ready(approach_pos, held_object_id, target_xy)
            deterministic_ready = (
                DETERMINISTIC_OBJECT_TRANSPORT
                and self.state_t >= PLACE_DETERMINISTIC_APPROACH_DURATION
            )
            if reached_approach or deterministic_ready:
                self.advance_state()
            return False

        # 状态8: 下降到释放高度
        if self.state == 8 and self.place_position:
            self.recenter_held_object_in_gripper(held_object_id, reason="放置下降前检测到偏心")
            
            # 非常规实验：上层物块需要计算正确的支撑高度
            slot = self.place_target_override.get("slot", "center") if self.place_target_override else "center"
            layer_index = self.place_target_override.get("layer_index", 0) if self.place_target_override else 0
            if slot == "center" and layer_index > 0 and self.is_special_experiment:
                support_top_z = 0.0
                for obj_id in self.placed_objects:
                    if obj_id == held_object_id:
                        continue
                    aabb = self.p.getAABB(obj_id)
                    support_top_z = max(support_top_z, aabb[1][2])
            else:
                support_top_z = self.get_stack_support_top_z(target_xy=self.place_position[:2], exclude_object_id=held_object_id)
            
            # 更新放置位置
            if self.place_target_snapshot is not None:
                self.stack_center = [self.place_target_snapshot[0], self.place_target_snapshot[1]]
                self.place_position = [self.stack_center[0], self.stack_center[1], 0]
            elif self.stack_anchor is not None:
                self.stack_center = [self.stack_anchor[0], self.stack_anchor[1]]
                self.place_position = [self.stack_center[0], self.stack_center[1], 0]

            release_height = self.get_held_object_release_height(held_object_id, support_top_z)

            target_xy = [self.place_position[0], self.place_position[1]]
            # 使用动态补偿计算目标位置
            low_pos = self.get_object_centered_ee_target(target_xy, release_height, held_object_id)
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(low_pos, orn), max_velocity=ARM_DESCEND_MAX_VELOCITY)
            self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
            
            # 检查是否到达目标位置（使用完整的3条件检查）
            reached_low_pos = self.placement_pose_is_ready(low_pos, held_object_id, target_xy)
            deterministic_ready = (
                DETERMINISTIC_OBJECT_TRANSPORT
                and self.state_t >= PLACE_DETERMINISTIC_DESCENT_DURATION
            )
            if reached_low_pos or deterministic_ready:
                self.advance_state()
            return False

        # 状态9: 保持位置稳定
        if self.state == 9:
            self.recenter_held_object_in_gripper(held_object_id, reason="释放前检测到偏心")
            
            # 非常规实验：上层物块需要计算正确的支撑高度
            slot = self.place_target_override.get("slot", "center") if self.place_target_override else "center"
            layer_index = self.place_target_override.get("layer_index", 0) if self.place_target_override else 0
            if slot == "center" and layer_index > 0 and self.is_special_experiment:
                support_top_z = 0.0
                for obj_id in self.placed_objects:
                    if obj_id == held_object_id:
                        continue
                    aabb = self.p.getAABB(obj_id)
                    support_top_z = max(support_top_z, aabb[1][2])
            else:
                support_top_z = self.get_stack_support_top_z(target_xy=self.place_position[:2], exclude_object_id=held_object_id)

            release_height = self.get_held_object_release_height(held_object_id, support_top_z)
            target_xy = [self.place_position[0], self.place_position[1]]
            # 使用动态补偿计算目标位置
            hold_pos = self.get_object_centered_ee_target(target_xy, release_height, held_object_id)
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(hold_pos, orn), max_velocity=ARM_DESCEND_MAX_VELOCITY)
            self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
            
            # 检查是否到达目标位置（使用完整的3条件检查）
            reached_hold_pos = self.placement_pose_is_ready(hold_pos, held_object_id, target_xy)
            deterministic_ready = (
                DETERMINISTIC_OBJECT_TRANSPORT
                and self.state_t >= PLACE_DETERMINISTIC_HOLD_DURATION
            )
            if (reached_hold_pos and self.state_t >= PLACE_SETTLE_DURATION) or deterministic_ready:
                self.advance_state()
            return False

        # 状态10: 慢速打开夹爪释放物体
        if self.state == 10:
            slot = self.place_target_override.get("slot", "center") if self.place_target_override else "center"
            is_special_bottom = slot in {"front", "back", "left", "right"}
            
            if not self.release_snapped:
                if DETERMINISTIC_OBJECT_TRANSPORT:
                    # 确定性搬运模式：动画移动物体到堆叠位置
                    if not self.update_gripper_place_animation(held_object_id):
                        return False
                    self.snap_object_to_stack(held_object_id)
                    print("确定性搬运：夹爪与物块已同步移动到堆叠中心，随后慢速张开夹爪。")
                else:
                    # 物理释放模式：检查物体是否对准堆叠点
                    target_xy = [self.place_position[0], self.place_position[1]]
                    if not (
                        self.held_object_is_near_gripper(held_object_id)
                        and self.held_object_is_over_stack(held_object_id, target_xy)
                    ):
                        # 物体未对准，退回状态7重新对位
                        self.recenter_held_object_in_gripper(held_object_id, reason="释放检查失败")
                        if not self.release_wait_warned:
                            print("释放检查未通过：物块尚未对准堆叠点，退回放置上方重新对位。")
                            self.release_wait_warned = True
                        self.state = 7
                        self.cur_state = 7
                        self.state_t = 0
                        return False
                    # 物体已对准，执行释放
                    self.p.resetBaseVelocity(held_object_id, [0, 0, 0], [0, 0, 0])
                    print("物理释放：物块中心已对准堆叠点，快速张开夹爪并解除约束。")
                self.release_open_start_t = self.state_t
                self.release_snapped = True
                # 非常规实验：先稍微打开夹爪，再解除约束，避免物块被夹爪带着走
                # 常规实验：保持原有逻辑，立即解除约束并快速打开夹爪
                if is_special_bottom:
                    self.setGripper(GRIPPER_OPEN_WIDTH * 0.5, force=GRIPPER_RELEASE_FORCE, max_velocity=GRIPPER_RELEASE_VELOCITY)
                    self.detach_held_object()
                else:
                    self.detach_held_object()
                    self.setGripper(GRIPPER_OPEN_WIDTH, force=GRIPPER_RELEASE_FORCE, max_velocity=GRIPPER_RELEASE_VELOCITY * 2)
                return False

            # 检查是否完全释放
            _, _, contact_count = self.get_target_contact_summary(held_object_id)
            release_elapsed = max(0.0, self.state_t - (self.release_open_start_t or self.state_t))
            
            # 非常规实验：先完全打开夹爪
            if is_special_bottom:
                self.setGripper(GRIPPER_OPEN_WIDTH, force=GRIPPER_RELEASE_FORCE, max_velocity=GRIPPER_RELEASE_VELOCITY * 2)
            
            if is_special_bottom and held_object_id is not None:
                vel, ang_vel = self.p.getBaseVelocity(held_object_id)
                speed = (vel[0]**2 + vel[1]**2 + vel[2]**2) ** 0.5
                ang_speed = (ang_vel[0]**2 + ang_vel[1]**2 + ang_vel[2]**2) ** 0.5
                object_settled = speed < 0.02 and ang_speed < 0.1
                contacts_released = contact_count == 0 and release_elapsed > 0.15
                if contacts_released and object_settled:
                    self.advance_state()
                elif release_elapsed > 2.0:
                    self.advance_state()
            else:
                # 常规实验：保持原有逻辑
                contacts_released = (
                    contact_count == 0
                    and release_elapsed > 0.1
                )
                release_timed_out = release_elapsed > 0.5
                if contacts_released or release_timed_out:
                    if release_timed_out and contact_count > 0:
                        print("夹爪释放后仍检测到接触，继续等待脱离。")
                    self.advance_state()
            return False

        # 状态11: 撤退到安全高度
        if self.state == 11:
            retreat_pos = [self.place_position[0], self.place_position[1], 0.4]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(retreat_pos, orn))
            self.open_gripper_for_release()
            if not self.has_reached_position(retreat_pos, PLACE_XY_TOLERANCE, PLACE_Z_TOLERANCE) and self.state_t <= self.state_durations[self.cur_state] + 1.0:
                return False
            if held_object_id is not None:
                self.placed_objects.append(held_object_id)
            self.place_count += 1
            self.reset()
            return True

        return False

    def reset(self):
        """
        重置状态变量
        """
        self.state = 0
        self.state_t = 0
        self.cur_state = 0
        self.place_position = None
        self.current_place_char = None
        self.place_target_snapshot = None
        self.place_target_override = None
        self.release_snapped = False
        self.release_open_start_t = None
        self.place_animation = None
        self.release_wait_warned = False
        self.reset_motion_watchdog()
        if self.place_count == 0:
            self.stack_anchor = None
        self.detach_held_object()

    def start_place(self):
        """
        开始放置流程（进入状态7）
        """
        self.state = 7
        self.cur_state = 7
        self.state_t = 0
        self.release_snapped = False
        self.release_open_start_t = None
        self.place_animation = None
        self.release_wait_warned = False
        self.reset_motion_watchdog()


class PandaSimAuto(PandaSim):
    """
    Panda机械臂自动控制类
    
    继承自PandaSim，添加自动状态更新功能
    """
    
    def __init__(self, bullet_client, offset):
        """
        初始化自动控制机械臂
        
        参数:
            bullet_client: PyBullet客户端
            offset: 机械臂位置偏移
        """
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.state_durations = [0.5, 1.0, 1.5, 1.0, 1.5, 1.0, 0.2, 1.0, 2.0, 0.5, PLACE_SLOW_RELEASE_DURATION, 1.0]
        self.place_count = 0
        self.placed_objects = []
        self.stack_center = None

    def update_state(self):
        """
        更新状态（自动推进非姿态控制状态）
        """
        self.state_t += self.control_dt
        if self.states[self.cur_state] in self.pose_controlled_states:
            return
        if self.state_t > self.state_durations[self.cur_state]:
            self.advance_state()
