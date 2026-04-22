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
        self.pose_controlled_states = {1, 2}
        self.current_place_char = None

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

    def get_end_effector_position(self):
        link_state = self.p.getLinkState(self.panda, pandaEndEffectorIndex)
        return np.array(link_state[4], dtype=float)

    def has_reached_position(self, target_pos):
        current_pos = self.get_end_effector_position()
        target = np.array(target_pos, dtype=float)
        delta = np.abs(current_pos - target)
        return delta[0] <= EE_XY_TOLERANCE and delta[1] <= EE_XY_TOLERANCE and delta[2] <= EE_Z_TOLERANCE

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

    def grasp_step(self, pos, angle, gripper_w):
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
            if self.has_reached_position(approach_pos):
                self.advance_state()
            return False

        if self.state == 2:
            self.setArm(self.calcJointLocation(target_pos, orn))
            if self.has_reached_position(target_pos):
                self.advance_state()
            return False

        if self.state == 3:
            self.setGripper(0)
            return False

        if self.state == 4:
            lift_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.05]
            self.setArm(self.calcJointLocation(lift_pos, orn))
            return False

        if self.state == 5:
            retreat_pos = [target_pos[0], target_pos[1], 0.4]
            self.setArm(self.calcJointLocation(retreat_pos, orn))
            return True

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
            if self.stack_center is None or char == "7":
                self.stack_center = [pos[0], pos[1]]

            self.place_position = [self.stack_center[0], self.stack_center[1], 0]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(pos, orn))
            return False

        if self.state == 8 and self.place_position:
            accumulated_height = 0.0
            for obj_id in self.placed_objects:
                aabb = self.p.getAABB(obj_id)
                accumulated_height += aabb[1][2] - aabb[0][2]

            current_cube_height = 0.04
            if held_object_id is not None:
                aabb = self.p.getAABB(held_object_id)
                current_cube_height = aabb[1][2] - aabb[0][2]

            release_height = accumulated_height + current_cube_height + 0.05

            if self.place_count > 0 and self.placed_objects and self.current_place_char != "7":
                last_obj_id = self.placed_objects[-1]
                obj_pos, _ = self.p.getBasePositionAndOrientation(last_obj_id)
                self.stack_center = [obj_pos[0], obj_pos[1]]
                self.place_position = [self.stack_center[0], self.stack_center[1], 0]

            low_pos = [self.place_position[0], self.place_position[1], release_height]
            orn = self.p.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2])
            self.setArm(self.calcJointLocation(low_pos, orn))
            return False

        if self.state == 9:
            self.setGripper(0.04)
            return False

        if self.state == 10:
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

    def start_place(self):
        self.state = 7
        self.cur_state = 7
        self.state_t = 0


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.state_durations = [0.5, 1.0, 1.5, 1.0, 1.5, 1.0, 0.2, 1.0, 2.0, 0.5, 0.5]
        self.place_count = 0
        self.placed_objects = []
        self.stack_center = None

    def update_state(self):
        self.state_t += self.control_dt
        if self.states[self.cur_state] in self.pose_controlled_states:
            return
        if self.state_t > self.state_durations[self.cur_state]:
            self.advance_state()
