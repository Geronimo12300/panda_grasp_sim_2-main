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
ARM_MOTOR_FORCE = 4 * 240.0
EE_XY_TOLERANCE = 0.003
EE_Z_TOLERANCE = 0.005
PLACE_XY_TOLERANCE = 0.008
PLACE_Z_TOLERANCE = 0.012
GRASP_ATTACH_DELAY = 0.12
GRASP_CONTACT_TIMEOUT = 0.45
GRASP_RETRY_LOWERING = 0.006
GRASP_CONTACT_XY_TOLERANCE = 0.025
GRASP_CONTACT_Z_TOLERANCE = 0.06
PLACE_SETTLE_DURATION = 0.35


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
                lateralFriction=8.0,
                spinningFriction=0.3,
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

    def attach_held_object(self, held_object_id):
        if held_object_id is None or self.active_grasp_constraint is not None:
            return

        link_state = self.p.getLinkState(self.panda, pandaEndEffectorIndex)
        ee_pos = link_state[4]
        ee_orn = link_state[5]
        obj_pos, obj_orn = self.p.getBasePositionAndOrientation(held_object_id)

        inv_pos, inv_orn = self.p.invertTransform(ee_pos, ee_orn)
        parent_frame_pos, parent_frame_orn = self.p.multiplyTransforms(inv_pos, inv_orn, obj_pos, obj_orn)

        self.active_grasp_constraint = self.p.createConstraint(
            self.panda,
            pandaEndEffectorIndex,
            held_object_id,
            -1,
            self.p.JOINT_FIXED,
            [0, 0, 0],
            parent_frame_pos,
            [0, 0, 0],
            parentFrameOrientation=parent_frame_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        self.held_object_id = held_object_id

    def detach_held_object(self):
        if self.active_grasp_constraint is not None:
            self.p.removeConstraint(self.active_grasp_constraint)
            self.active_grasp_constraint = None
        self.held_object_id = None

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
            maxNumIterations=20,
        )

    def get_target_contact_summary(self, target_object_id):
        if target_object_id is None:
            return False, False, 0

        left_contacts = self.p.getContactPoints(self.panda, target_object_id, linkIndexA=9)
        right_contacts = self.p.getContactPoints(self.panda, target_object_id, linkIndexA=10)
        return bool(left_contacts), bool(right_contacts), len(left_contacts) + len(right_contacts)

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
        has_stable_contact = (left_contact and right_contact) or contact_count >= 2
        return (
            has_stable_contact
            and xy_error <= GRASP_CONTACT_XY_TOLERANCE
            and z_error <= GRASP_CONTACT_Z_TOLERANCE
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

    def advance_state(self):
        self.cur_state += 1
        if self.cur_state >= len(self.states):
            self.cur_state = 0
        self.state_t = 0
        self.state = self.states[self.cur_state]

    def setArm(self, jointPoses):
        for i in range(pandaNumDofs):
            self.p.setJointMotorControl2(
                self.panda,
                i,
                self.p.POSITION_CONTROL,
                jointPoses[i],
                force=ARM_MOTOR_FORCE,
                maxVelocity=0.8,
            )

    def setGripper(self, finger_target):
        for i in [9, 10]:
            self.p.setJointMotorControl2(
                self.panda,
                i,
                self.p.POSITION_CONTROL,
                finger_target,
                force=100,
                maxVelocity=GRIPPER_MAX_VELOCITY,
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
            self.setGripper(0)
            if self.state_t >= GRASP_ATTACH_DELAY and self.can_attach_grasp_target(held_object_id):
                self.attach_held_object(held_object_id)
                self.advance_state()
            elif self.state_t > self.state_durations[self.cur_state] + GRASP_CONTACT_TIMEOUT:
                print("抓取失败：夹爪未形成稳定接触，本次按空抓处理。")
                self.advance_state()
            return False

        if self.state == 4:
            lift_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.05]
            self.setArm(self.calcJointLocation(lift_pos, orn))
            if self.has_reached_position(lift_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        if self.state == 5:
            retreat_pos = [target_pos[0], target_pos[1], 0.4]
            self.setArm(self.calcJointLocation(retreat_pos, orn))
            if self.has_reached_position(retreat_pos) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                return True
            return False

        return False

    def place_step(self, char, held_object_id=None):
        self.update_state()

        if self.state == 7:
            position_map = {
                "7": [0.5, 0, 0.4],
                "8": [0, 0.5, 0.4],
                "9": [-0.5, 0, 0.4],
                "0": [0.5, 0.2, 0.4],
            }
            if char not in position_map:
                return False

            pos = position_map[char]
            self.current_place_char = char
            if self.stack_center is None or self.place_count == 0:
                self.stack_center = [pos[0], pos[1]]
                self.stack_anchor = [pos[0], pos[1]]
            elif self.stack_anchor is not None:
                self.stack_center = [self.stack_anchor[0], self.stack_anchor[1]]

            self.place_position = [self.stack_center[0], self.stack_center[1], 0]
            approach_pos = [self.place_position[0], self.place_position[1], pos[2]]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(approach_pos, orn))
            if self.has_reached_position(approach_pos, PLACE_XY_TOLERANCE, PLACE_Z_TOLERANCE) or self.state_t > self.state_durations[self.cur_state] + 1.0:
                self.advance_state()
            return False

        if self.state == 8 and self.place_position:
            support_top_z = 0.0
            for obj_id in self.placed_objects:
                aabb = self.p.getAABB(obj_id)
                support_top_z = max(support_top_z, aabb[1][2])
            if self.stack_anchor is not None:
                self.stack_center = [self.stack_anchor[0], self.stack_anchor[1]]
                self.place_position = [self.stack_center[0], self.stack_center[1], 0]

            release_height = self.get_held_object_release_height(held_object_id, support_top_z)

            low_pos = [self.place_position[0], self.place_position[1], release_height]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(low_pos, orn))
            if self.has_reached_position(low_pos, PLACE_XY_TOLERANCE, PLACE_Z_TOLERANCE) or self.state_t > self.state_durations[self.cur_state] + 1.2:
                self.advance_state()
            return False

        if self.state == 9:
            support_top_z = 0.0
            for obj_id in self.placed_objects:
                aabb = self.p.getAABB(obj_id)
                support_top_z = max(support_top_z, aabb[1][2])

            release_height = self.get_held_object_release_height(held_object_id, support_top_z)
            hold_pos = [self.place_position[0], self.place_position[1], release_height]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(hold_pos, orn))
            self.setGripper(0)
            if (self.has_reached_position(hold_pos, PLACE_XY_TOLERANCE, PLACE_Z_TOLERANCE) and self.state_t >= PLACE_SETTLE_DURATION) or self.state_t > self.state_durations[self.cur_state] + 0.8:
                self.advance_state()
            return False

        if self.state == 10:
            self.detach_held_object()
            self.setGripper(0.04)
            if self.state_t > self.state_durations[self.cur_state]:
                self.advance_state()
            return False

        if self.state == 11:
            retreat_pos = [self.place_position[0], self.place_position[1], 0.4]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(retreat_pos, orn))
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
        if self.place_count == 0:
            self.stack_anchor = None
        self.detach_held_object()

    def start_place(self):
        self.state = 7
        self.cur_state = 7
        self.state_t = 0


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.state_durations = [0.5, 1.0, 1.5, 1.0, 1.5, 1.0, 0.2, 1.0, 2.0, 0.5, 0.3, 1.0]
        self.place_count = 0
        self.placed_objects = []
        self.stack_center = None

    def update_state(self):
        self.state_t += self.control_dt
        if self.states[self.cur_state] in self.pose_controlled_states:
            return
        if self.state_t > self.state_durations[self.cur_state]:
            self.advance_state()
