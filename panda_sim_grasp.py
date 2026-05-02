import numpy as np
import math
import pybullet as p

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
ul = [7] * pandaNumDofs
jr = [7] * pandaNumDofs
jointPositions = (
    0.8045609285966308,
    0.525471701354679,
    -0.02519566900946519,
    -1.3925086098003587,
    0.013443782914225877,
    1.9178323512245277,
    -0.007207024243406651,
    0.01999436579245478,
    0.019977024051412193,
)
rp = jointPositions

GRIPPER_MAX_VELOCITY = 0.25
GRIPPER_RELEASE_VELOCITY = 0.08
GRIPPER_RESET_VELOCITY = 0.5
GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0
GRIPPER_GRASP_FORCE = 90
GRIPPER_RELEASE_FORCE = 45
GRIPPER_RESET_FORCE = 120
ARM_MOTOR_FORCE = 4 * 240.0
ARM_MAX_VELOCITY = 0.8
ARM_CARRY_MAX_VELOCITY = 0.55
ARM_DESCEND_MAX_VELOCITY = 0.38
DETERMINISTIC_OBJECT_TRANSPORT = False
FREEZE_PLACED_OBJECTS = False
EE_XY_TOLERANCE = 0.003
EE_Z_TOLERANCE = 0.005
PLACE_XY_TOLERANCE = 0.008
PLACE_Z_TOLERANCE = 0.012
PLACE_OBJECT_XY_TOLERANCE = 0.012
PLACE_GRIPPER_OBJECT_MAX_OFFSET = 0.12
PLACE_REGRIP_XY_OFFSET = 0.018
PLACE_SUPPORT_XY_MARGIN = 0.026
GRASP_CENTER_OBJECT_ON_ATTACH = True
GRASP_CENTER_ATTACH_MAX_XY_CORRECTION = 0.075
GRASP_CAPTURE_ZONE_XY = 0.022
GRASP_CAPTURE_ZONE_Z_MIN = 0.015
GRASP_CAPTURE_ZONE_Z_MAX = 0.090
GRASP_OBJECT_EE_MIN_Z_OFFSET = 0.045
GRASP_OBJECT_EE_MAX_Z_OFFSET = 0.070
GRASP_OBJECT_MIN_BOTTOM_CLEARANCE = 0.006
GRASP_REGRIP_MIN_BOTTOM_Z = 0.055
GRASP_CONSTRAINT_FORCE = 520
GRASP_ATTACH_DELAY = 0.12
GRASP_CONTACT_TIMEOUT = 0.45
GRASP_RETRY_LOWERING = 0.006
GRASP_CONTACT_XY_TOLERANCE = 0.025
GRASP_CONTACT_Z_TOLERANCE = 0.06
GRASP_POSE_ATTACH_XY_TOLERANCE = 0.04
GRASP_POSE_ATTACH_Z_TOLERANCE = 0.09
PHYSICAL_GRASP_LIFT_CLEARANCE = 0.014
PHYSICAL_GRASP_HOLD_XY_TOLERANCE = 0.04
PHYSICAL_GRASP_HOLD_Z_TOLERANCE = 0.11
PLACE_SETTLE_DURATION = 0.35
PLACE_APPROACH_TIMEOUT = 1.2
PLACE_DESCENT_TIMEOUT = 1.4
PLACE_RELEASE_TIMEOUT = 1.0
PLACE_HARD_RELEASE_TIMEOUT = 3.0
PLACE_SLOW_RELEASE_DURATION = 1.2
PLACE_OBJECT_MOVE_DURATION = 0.9
PLACE_DETERMINISTIC_APPROACH_DURATION = 0.35
PLACE_DETERMINISTIC_DESCENT_DURATION = 0.35
PLACE_DETERMINISTIC_HOLD_DURATION = 0.12
PLACE_FORCE_RELEASE_XY_TOLERANCE = 0.025
PLACE_LOOSE_XY_TOLERANCE = 0.045
PLACE_TARGETS = {
    "7": [0.5, 0.0, 0.4],
    "8": [0.0, 0.5, 0.4],
    "9": [-0.5, 0.0, 0.4],
    "0": [0.5, 0.2, 0.4],
}


class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.p = bullet_client
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)

        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn = [0, 0, 0, 1]
        self.panda = self.p.loadURDF(
            "franka_panda/panda_1.urdf",
            np.array([0, 0, 0]) + self.offset,
            orn,
            useFixedBase=True,
            flags=flags,
        )

        for i in [9, 10]:
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

        index = 0
        self.state = 0
        self.control_dt = 1.0 / 240.0
        self.finger_target = 0
        self.gripper_height = 0.2
        self.pose_controlled_states = {1, 2, 4, 5, 7, 8, 9, 10, 11}
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

    def reset_robot_joints_to_initial(self):
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

    def reset_gripper_open_immediately(self):
        for i in [9, 10]:
            self.p.resetJointState(self.panda, i, GRIPPER_OPEN_WIDTH)
        self.setGripper(
            GRIPPER_OPEN_WIDTH,
            force=GRIPPER_RESET_FORCE,
            max_velocity=GRIPPER_RESET_VELOCITY,
        )

    def clear_grasp_contact_state(self, settle_steps=80):
        self.detach_held_object()
        self.reset_gripper_open_immediately()
        for _ in range(settle_steps):
            self.setGripper(
                GRIPPER_OPEN_WIDTH,
                force=GRIPPER_RELEASE_FORCE,
                max_velocity=GRIPPER_RESET_VELOCITY,
            )
            self.p.stepSimulation()

    def reset_to_initial_pose(self, settle_steps=80):
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

    def attach_held_object(self, held_object_id):
        if held_object_id is None:
            return

        self.active_grasp_constraint = None
        self.held_object_id = held_object_id
        self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)

    def object_is_in_gripper_capture_zone(self, target_object_id):
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
        if held_object_id is None:
            return 0.0
        aabb_min, _ = self.p.getAABB(held_object_id)
        return float(aabb_min[2])

    def recenter_held_object_in_gripper(self, held_object_id, reason=""):
        return False

    def set_place_target_override(self, place_pose=None):
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
        }

    def reset_motion_watchdog(self):
        self.motion_stall_counter = 0
        self.last_ee_pos = None

    def detach_held_object(self):
        if self.active_grasp_constraint is not None:
            self.p.removeConstraint(self.active_grasp_constraint)
            self.active_grasp_constraint = None
        self.held_object_id = None

    def object_is_physically_held(self, held_object_id, require_lift=False):
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
        if self.object_is_physically_held(held_object_id, require_lift=require_lift):
            self.held_object_id = held_object_id
            return True

        if self.held_object_id == held_object_id:
            self.held_object_id = None
        return False

    def calcJointLocation(self, pos, orn):
        return self.p.calculateInverseKinematics(
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

    def get_target_contact_summary(self, target_object_id):
        if target_object_id is None:
            return False, False, 0

        left_contacts = self.p.getContactPoints(self.panda, target_object_id, linkIndexA=9)
        right_contacts = self.p.getContactPoints(self.panda, target_object_id, linkIndexA=10)
        return bool(left_contacts), bool(right_contacts), len(left_contacts) + len(right_contacts)

    def is_cylinder_object(self, target_object_id):
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
        link_state = self.p.getLinkState(self.panda, pandaEndEffectorIndex)
        return np.array(link_state[4], dtype=float)

    def can_attach_grasp_target(self, target_object_id):
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
        has_stable_contact = (
            contact_count >= 1
            if is_cylinder
            else (left_contact and right_contact) or contact_count >= 2
        )
        xy_tolerance = 0.035 if is_cylinder else GRASP_CONTACT_XY_TOLERANCE
        z_tolerance = 0.075 if is_cylinder else GRASP_CONTACT_Z_TOLERANCE
        return (
            has_stable_contact
            and xy_error <= xy_tolerance
            and z_error <= z_tolerance
        )

    def can_attach_grasp_target_by_pose(self, target_object_id, target_pos):
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
        base_clearance = 0.012
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
                extra_clearance = 0.008
            elif geom_type == self.p.GEOM_MESH:
                mesh_name = visual_shape_data[0][4]
                if isinstance(mesh_name, bytes):
                    mesh_name = mesh_name.decode("utf-8", errors="ignore")
                if mesh_name and "cone_top" in mesh_name:
                    extra_clearance = 0.012

        min_release_offset = 0.06
        release_offset = max(ee_to_object_bottom, min_release_offset)
        return support_top_z + release_offset + base_clearance + extra_clearance

    def has_reached_position(self, target_pos, xy_tolerance=EE_XY_TOLERANCE, z_tolerance=EE_Z_TOLERANCE):
        current_pos = self.get_end_effector_position()
        target = np.array(target_pos, dtype=float)
        delta = np.abs(current_pos - target)
        return delta[0] <= xy_tolerance and delta[1] <= xy_tolerance and delta[2] <= z_tolerance

    def has_reached_xy(self, target_pos, xy_tolerance=PLACE_FORCE_RELEASE_XY_TOLERANCE):
        current_pos = self.get_end_effector_position()
        target = np.array(target_pos, dtype=float)
        return np.linalg.norm(current_pos[:2] - target[:2]) <= xy_tolerance

    def get_held_object_offset_from_ee(self, held_object_id):
        if held_object_id is None:
            return np.zeros(3, dtype=float)

        ee_pos = self.get_end_effector_position()
        obj_pos, _ = self.p.getBasePositionAndOrientation(held_object_id)
        return np.array(obj_pos, dtype=float) - ee_pos

    def get_object_centered_ee_target(self, target_xy, target_ee_z, held_object_id):
        offset = self.get_held_object_offset_from_ee(held_object_id)
        if np.linalg.norm(offset[:2]) > PLACE_GRIPPER_OBJECT_MAX_OFFSET:
            # If the object is this far from the gripper, it is probably lost; do not chase it.
            offset[:2] = 0.0

        return [
            float(target_xy[0] - offset[0]),
            float(target_xy[1] - offset[1]),
            float(target_ee_z),
        ]

    def held_object_is_near_gripper(self, held_object_id):
        if held_object_id is None:
            return False

        offset = self.get_held_object_offset_from_ee(held_object_id)
        return (
            np.linalg.norm(offset[:2]) <= PLACE_GRIPPER_OBJECT_MAX_OFFSET
            and abs(float(offset[2])) <= 0.18
        )

    def held_object_is_over_stack(self, held_object_id, target_xy, xy_tolerance=PLACE_OBJECT_XY_TOLERANCE):
        if held_object_id is None:
            return False

        obj_pos, _ = self.p.getBasePositionAndOrientation(held_object_id)
        return np.linalg.norm(np.array(obj_pos[:2], dtype=float) - np.array(target_xy, dtype=float)) <= xy_tolerance

    def placement_pose_is_ready(self, ee_target, held_object_id, target_xy):
        return (
            self.has_reached_position(ee_target, PLACE_XY_TOLERANCE, PLACE_Z_TOLERANCE)
            and self.held_object_is_near_gripper(held_object_id)
            and self.held_object_is_over_stack(held_object_id, target_xy)
        )

    def get_place_hold_width(self):
        if not self.place_target_override:
            return GRIPPER_CLOSED_WIDTH
        requested_width = float(self.place_target_override.get("place_hold_width", GRIPPER_CLOSED_WIDTH))
        return float(np.clip(requested_width, GRIPPER_CLOSED_WIDTH, 0.012))

    def get_slow_release_width(self, elapsed=None, start_width=None):
        release_elapsed = self.state_t if elapsed is None else elapsed
        progress = np.clip(release_elapsed / PLACE_SLOW_RELEASE_DURATION, 0.0, 1.0)
        base_width = self.get_place_hold_width() if start_width is None else start_width
        return base_width + progress * (GRIPPER_OPEN_WIDTH - base_width)

    def open_gripper_for_release(self, slow=False, elapsed=None, start_width=None):
        finger_target = self.get_slow_release_width(elapsed, start_width=start_width) if slow else GRIPPER_OPEN_WIDTH
        self.setGripper(
            finger_target,
            force=GRIPPER_RELEASE_FORCE,
            max_velocity=GRIPPER_RELEASE_VELOCITY,
        )

    def force_release_held_object(self):
        self.held_object_id = None
        self.reset_gripper_open_immediately()

    def get_stack_support_top_z(self, target_xy=None, exclude_object_id=None):
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
        return False

    def update_gripper_place_animation(self, held_object_id):
        return False

    def advance_state(self):
        self.cur_state += 1
        if self.cur_state >= len(self.states):
            self.cur_state = 0
        self.state_t = 0
        self.state = self.states[self.cur_state]
        self.reset_motion_watchdog()

    def setArm(self, jointPoses, max_velocity=ARM_MAX_VELOCITY):
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
        for i in [9, 10]:
            self.p.setJointMotorControl2(
                self.panda,
                i,
                self.p.POSITION_CONTROL,
                finger_target,
                force=force,
                maxVelocity=max_velocity,
            )

    def grasp_step(self, pos, angle, gripper_w, held_object_id=None):
        self.update_state()
        target_pos = list(pos)
        target_pos[2] += 0.047
        orn = self.p.getQuaternionFromEuler([math.pi, 0.0, angle + math.pi / 2])

        if self.state == 0:
            safe_pos = [0.5, 0, 0.4]
            safe_orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(safe_pos, safe_orn))
            return False

        if self.state == 1:
            approach_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.1]
            self.setArm(self.calcJointLocation(approach_pos, orn))
            self.setGripper(gripper_w)
            if self.has_reached_position(approach_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        if self.state == 2:
            self.setArm(self.calcJointLocation(target_pos, orn))
            if self.has_reached_position(target_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        if self.state == 3:
            squeeze_pos = list(target_pos)
            if self.state_t >= GRASP_ATTACH_DELAY + 0.12:
                squeeze_pos[2] -= GRASP_RETRY_LOWERING
            self.setArm(self.calcJointLocation(squeeze_pos, orn))
            self.setGripper(0, force=GRIPPER_GRASP_FORCE)
            can_attach_by_contact = self.can_attach_grasp_target(held_object_id)
            can_attach_by_pose = (
                self.can_attach_grasp_target_by_pose(held_object_id, target_pos)
                and self.object_is_in_gripper_capture_zone(held_object_id)
            )
            can_attach_deterministically = False
            if self.state_t >= GRASP_ATTACH_DELAY and (can_attach_by_contact or can_attach_by_pose):
                if can_attach_by_pose and not can_attach_by_contact:
                    print("夹爪已包住目标物块，使用位姿兜底建立夹持。")
                if can_attach_deterministically and not (can_attach_by_contact or can_attach_by_pose):
                    print("启用确定性搬运：不等待接触点，直接挂接目标物块。")
                self.attach_held_object(held_object_id)
                self.advance_state()
            elif self.state_t > self.state_durations[self.cur_state] + GRASP_CONTACT_TIMEOUT:
                print("抓取失败：夹爪未形成稳定接触，本次按空抓处理。")
                self.advance_state()
            return False

        if self.state == 4:
            self.recenter_held_object_in_gripper(held_object_id, reason="抓取抬升前检测到偏心")
            lift_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.05]
            self.setArm(self.calcJointLocation(lift_pos, orn))
            self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
            if self.has_reached_position(lift_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        if self.state == 5:
            self.recenter_held_object_in_gripper(held_object_id, reason="抓取搬运前检测到偏心")
            retreat_pos = [target_pos[0], target_pos[1], 0.4]
            self.setArm(self.calcJointLocation(retreat_pos, orn), max_velocity=ARM_CARRY_MAX_VELOCITY)
            self.setGripper(GRIPPER_CLOSED_WIDTH, force=GRIPPER_GRASP_FORCE)
            if self.has_reached_position(retreat_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                return True
            return False

        return False

    def place_step(self, char, held_object_id=None):
        self.update_state()
        if self.state == 7:
            self.recenter_held_object_in_gripper(held_object_id, reason="放置上方移动前检测到偏心")
            if char not in PLACE_TARGETS:
                return False

            pos = PLACE_TARGETS[char]
            self.current_place_char = char
            place_x = self.place_target_override["x"] if self.place_target_override else pos[0]
            place_y = self.place_target_override["y"] if self.place_target_override else pos[1]
            place_z = self.place_target_override["z"] if self.place_target_override else 0.0
            self.stack_center = [place_x, place_y]
            self.place_position = [place_x, place_y, place_z]
            self.place_target_snapshot = [place_x, place_y, place_z]
            target_xy = [self.place_position[0], self.place_position[1]]
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

        if self.state == 8 and self.place_position:
            self.recenter_held_object_in_gripper(held_object_id, reason="放置下降前检测到偏心")
            support_top_z = self.get_stack_support_top_z(target_xy=self.place_position[:2], exclude_object_id=held_object_id)
            if self.place_target_snapshot is not None:
                self.stack_center = [self.place_target_snapshot[0], self.place_target_snapshot[1]]
                self.place_position = [self.place_target_snapshot[0], self.place_target_snapshot[1], self.place_target_snapshot[2] if len(self.place_target_snapshot) > 2 else 0]
            elif self.stack_anchor is not None:
                self.stack_center = [self.stack_anchor[0], self.stack_anchor[1]]
                self.place_position = [self.stack_center[0], self.stack_center[1], 0]

            release_height = self.get_held_object_release_height(held_object_id, support_top_z)

            target_xy = [self.place_position[0], self.place_position[1]]
            low_pos = self.get_object_centered_ee_target(target_xy, release_height, held_object_id)
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(low_pos, orn), max_velocity=ARM_DESCEND_MAX_VELOCITY)
            self.setGripper(self.get_place_hold_width(), force=GRIPPER_GRASP_FORCE)
            reached_low_pos = self.placement_pose_is_ready(low_pos, held_object_id, target_xy)
            deterministic_ready = (
                DETERMINISTIC_OBJECT_TRANSPORT
                and self.state_t >= PLACE_DETERMINISTIC_DESCENT_DURATION
            )
            if reached_low_pos or deterministic_ready:
                self.advance_state()
            return False

        if self.state == 9:
            self.recenter_held_object_in_gripper(held_object_id, reason="释放前检测到偏心")
            support_top_z = self.get_stack_support_top_z(target_xy=self.place_position[:2], exclude_object_id=held_object_id)

            release_height = self.get_held_object_release_height(held_object_id, support_top_z)
            target_xy = [self.place_position[0], self.place_position[1]]
            hold_pos = self.get_object_centered_ee_target(target_xy, release_height, held_object_id)
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(hold_pos, orn), max_velocity=ARM_DESCEND_MAX_VELOCITY)
            self.setGripper(self.get_place_hold_width(), force=GRIPPER_GRASP_FORCE)
            reached_hold_pos = self.placement_pose_is_ready(hold_pos, held_object_id, target_xy)
            deterministic_ready = (
                DETERMINISTIC_OBJECT_TRANSPORT
                and self.state_t >= PLACE_DETERMINISTIC_HOLD_DURATION
            )
            if (reached_hold_pos and self.state_t >= PLACE_SETTLE_DURATION) or deterministic_ready:
                self.advance_state()
            return False

        if self.state == 10:
            release_start_width = self.get_place_hold_width()
            if not self.release_snapped:
                self.setGripper(release_start_width, force=GRIPPER_GRASP_FORCE)
                if DETERMINISTIC_OBJECT_TRANSPORT:
                    if not self.update_gripper_place_animation(held_object_id):
                        return False
                    self.snap_object_to_stack(held_object_id)
                    print("确定性搬运：夹爪与物块已同步移动到堆叠中心，随后慢速张开夹爪。")
                else:
                    target_xy = [self.place_position[0], self.place_position[1]]
                    if not (
                        self.held_object_is_near_gripper(held_object_id)
                        and self.held_object_is_over_stack(held_object_id, target_xy)
                    ):
                        self.recenter_held_object_in_gripper(held_object_id, reason="释放检查失败")
                        if not self.release_wait_warned:
                            print("释放检查未通过：物块尚未对准堆叠点，退回放置上方重新对位。")
                            self.release_wait_warned = True
                        self.state = 7
                        self.cur_state = 7
                        self.state_t = 0
                        return False
                    self.p.resetBaseVelocity(held_object_id, [0, 0, 0], [0, 0, 0])
                    self.detach_held_object()
                    print("物理释放：物块中心已对准堆叠点，解除夹爪约束并慢速张开夹爪。")
                self.release_open_start_t = self.state_t
                self.release_snapped = True

            release_elapsed = max(0.0, self.state_t - (self.release_open_start_t or self.state_t))
            self.open_gripper_for_release(slow=True, elapsed=release_elapsed, start_width=release_start_width)
            _, _, contact_count = self.get_target_contact_summary(held_object_id)
            contacts_released = (
                contact_count == 0
                and release_elapsed > PLACE_SLOW_RELEASE_DURATION
            )
            release_timed_out = release_elapsed > max(PLACE_RELEASE_TIMEOUT, PLACE_SLOW_RELEASE_DURATION + 0.2)
            if contacts_released or release_timed_out:
                self.held_object_id = None
                if release_timed_out and contact_count > 0:
                    print("夹爪释放后仍检测到接触，继续抬升以脱离物块。")
                self.advance_state()
            return False

        if self.state == 11:
            retreat_pos = [self.place_position[0], self.place_position[1], 0.4]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(retreat_pos, orn), max_velocity=ARM_CARRY_MAX_VELOCITY)
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
        self.state = 7
        self.cur_state = 7
        self.state_t = 0
        self.release_snapped = False
        self.release_open_start_t = None
        self.place_animation = None
        self.release_wait_warned = False
        self.reset_motion_watchdog()


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.state_durations = [0.5, 1.0, 1.5, 1.0, 1.5, 1.0, 0.2, 1.0, 2.0, 0.5, PLACE_SLOW_RELEASE_DURATION, 1.0]
        self.place_count = 0
        self.placed_objects = []
        self.stack_center = None

    def update_state(self):
        self.state_t += self.control_dt
        if self.states[self.cur_state] in self.pose_controlled_states:
            return
        if self.state_t > self.state_durations[self.cur_state]:
            self.advance_state()
