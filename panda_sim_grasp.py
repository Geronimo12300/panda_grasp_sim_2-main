import time
import numpy as np
import math
import pybullet as p

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
# restposes for null space
jointPositions=(0.8045609285966308, 0.525471701354679, -0.02519566900946519, -1.3925086098003587, 0.013443782914225877, 1.9178323512245277, -0.007207024243406651, 0.01999436579245478, 0.019977024051412193)
            # [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.p = bullet_client
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn=[0, 0, 0, 1]
        self.panda = self.p.loadURDF("franka_panda/panda_1.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)

        # 设置夹爪的物理属性
        for i in [9, 10]:  # 夹爪的关节索引
            self.p.changeDynamics(self.panda, i, mass=0.2)
            self.p.changeDynamics(self.panda, i, lateralFriction=5.0)
        # 启用夹爪的碰撞检测
        for i in [9, 10]:  # 夹爪的关节索引
            self.p.changeDynamics(self.panda, i, collisionMargin=0.001)  # 设置碰撞边界


        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        #create a constraint to keep the fingers centered
        c = self.p.createConstraint(self.panda,
                          9,
                          self.panda,
                          10,
                          jointType=self.p.JOINT_GEAR,
                          jointAxis=[1, 0, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=100)
    
        for j in range(self.p.getNumJoints(self.panda)):
            self.p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.panda, j)
            #print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.p.JOINT_PRISMATIC):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1

            if (jointType == self.p.JOINT_REVOLUTE):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
        self.t = 0.

    def calcJointLocation(self, pos, orn):
        """
        根据 pos 和 orn 计算机械臂的关节位置 
        """
        jointPoses = self.p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul, jr, rp, maxNumIterations=20)
        return jointPoses
    
    
    def loadObjInURDF(self, idx, num, render_n):
    # ... 其他代码 ...

    # 加载物体
        urdf_id = self.p.loadURDF(self.urdfs_filename[0], basePosition, baseOrientation) # type: ignore

    # 设置物体的物理属性
        self.p.changeDynamics(urdf_id, -1, mass=0.1)  # 设置物体的质量
        self.p.changeDynamics(urdf_id, -1, lateralFriction=1.0)  # 设置物体的摩擦力


    def setArm(self, jointPoses):
        """
        设置机械臂位置
        """
        for i in range(pandaNumDofs):
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
    
    def setGripper(self, finger_target):
        finger_target = finger_target * 1
        for i in [9,10]:
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, finger_target, force=100, maxVelocity=0.5)


    def grasp_step(self, pos, angle, gripper_w):
        """
        pos: [x, y, z]
        angle: 弧度
        gripper_w: 抓取器张开宽度
        """
        
        # 测试用
        # pos = [0.5, 0, 0.3] # 机械手位置
        # orn = self.p.getQuaternionFromEuler([math.pi, 0., math.pi / 2])   # 机械手方向
        # jointPoses = self.calcJointLocation(pos, orn)
        # print('jointPoses = ', jointPoses)
        # self.setArm(jointPoses)
        # return False



        # 更新状态
        self.update_state()
        pos[2] += 0.047

        
        if self.state == 0:
            # print('恢复初始状态')
            pos = [0.5, 0, 0.4] # 机械手位置
            orn = self.p.getQuaternionFromEuler([math.pi,0., math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        elif self.state == 1:
            # print('物体上方，张开抓取器')
            pos[2] += 0.1
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            self.setGripper(gripper_w)
            return False

        elif self.state == 2:
            # print('抓取位置')
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        elif self.state == 3:
            # print('闭合抓取器')
            self.setGripper(0)
            return False
        
        elif self.state == 4:
            # print('物体上方')
            pos[2] += 0.05
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False
        
        elif self.state == 5:
            # print('物体上方(预抓取位置)')
            pos[2] = 0.4
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        # ========================= 晃动抓取器，测试抓取稳定性 =========================
        elif self.state == 6:
            # print('x正方向晃动')
            pos[2] = 0.4
            pos[0] -= 0.05
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False
        elif self.state == 7:
            # print('x负方向晃动')
            pos[2] = 0.4
            pos[0] += 0.05
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return True
        # =========================  =========================

    def place_step(self, char, held_object_id=None):
        self.update_state()

        if self.state == 7:
            position_map = {
                '7': [0.5, 0, 0.4],
                '8': [0, 0.5, 0.4],
                '9': [-0.5, 0, 0.4],
                '0': [0.5, 0.2, 0.4]
            }

            if char not in position_map:
                return False

            pos = position_map[char]
            if self.stack_center is None:
                self.stack_center = [pos[0], pos[1]]
            
            self.place_position = [self.stack_center[0], self.stack_center[1], 0]
            orn = self.p.getQuaternionFromEuler([math.pi, 0., math.pi / 2])
            joint_poses = self.calcJointLocation(pos, orn)
            self.setArm(joint_poses)

        elif self.state == 8:
            if self.place_position:
                accumulated_height = 0.0
                for obj_id in self.placed_objects:
                    aabb = self.p.getAABB(obj_id)
                    obj_height = aabb[1][2] - aabb[0][2]
                    accumulated_height += obj_height
                
                current_cube_height = 0.04
                if held_object_id is not None:
                    aabb = self.p.getAABB(held_object_id)
                    current_cube_height = aabb[1][2] - aabb[0][2]
                
                release_height = accumulated_height + current_cube_height + 0.05
                
                if self.place_count > 0 and len(self.placed_objects) > 0:
                    last_obj_id = self.placed_objects[-1]
                    obj_pos, _ = self.p.getBasePositionAndOrientation(last_obj_id)
                    self.stack_center = [obj_pos[0], obj_pos[1]]
                    self.place_position = [self.stack_center[0], self.stack_center[1], 0]
                
                low_pos = [self.place_position[0], self.place_position[1], release_height]
                orn = self.p.getQuaternionFromEuler([math.pi, 0., math.pi / 2])
                joint_poses = self.calcJointLocation(low_pos, orn)
                self.setArm(joint_poses)

        elif self.state == 9:
            self.setGripper(0.04)

        elif self.state == 10:
            if held_object_id is not None:
                self.placed_objects.append(held_object_id)
            self.place_count += 1
            self.reset()
            return True


    def reset(self):
        self.state = 0
        self.state_t = 0
        self.cur_state = 0
        self.place_position = None


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.state_durations=[0.5, 1.0, 1.5, 1.0, 1.5, 1.0, 0.2, 1.0, 2.0, 0.5, 0.5]
        self.place_position = None
        self.place_count = 0
        self.placed_objects = []
        self.stack_center = None
    
    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state=self.states[self.cur_state]
