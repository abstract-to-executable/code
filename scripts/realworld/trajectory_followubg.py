import xarm
import numpy as np

Z_LOW = 0.175
Z_LOW_PANDA = 0.11
Z_SCALING = 5.08 / 4
Z_LIMIT = 0.5
GRIPPER_LIMIT = 850

arm = xarm.XArmAPI("192.168.1.240", is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
arm.set_gripper_enable(True)
"""
STEPSSSSSS

"""



def main(filename):
    traj = np.load(filename, allow_pickle=True)
    states = traj["states"]
    print(f"Trajectory length: {len(states)}")
    min_z = 1
    ground_place = True
    states = np.array(states)
    ms=states[-1:, -1, 18+2]
    # print(states[:, -1, 18:21])
    if ms > 0.03:
        ground_place = False
    block_height = 0
    block_height = np.round((ms - 0.02) / 0.04, )
    print(ms)
    print(f"===ground_place = {ground_place} - bh = {block_height}===")
    for i in range(8, len(states)):
        robot_state = states[i][-1][:-4]
        ee_pos = robot_state[25:28]
        gripper_pos = robot_state[7:9]
        min_z = min(min_z, ee_pos[2])
        move_to_pos(ee_pos, gripper_pos, i, ground_place=ground_place, block_height=block_height)
    print(min_z)


def move_to_pos(ee_pos, gripper_pos, i, ground_place=True, block_height=0):
    # Move the xy position
    # panda2xarm_offset = np.array([0.425, 0.12, 0])
    ee_pos[0] *= Z_SCALING
    ee_pos[1] *= Z_SCALING

    # calibrate this offset before starting!
    panda2xarm_offset = np.array([0.425, 0.16, 0])
    # panda2xarm_offset = np.array([0.382, 0.115, 0])
    ee_xarm = ee_pos + panda2xarm_offset

    # Scale and move the z position
    scaled_z = (ee_xarm[2] - Z_LOW_PANDA) * Z_SCALING + Z_LOW
    # scaled_x = (ee_xarm[1] - X_LOW_PANDA) * Z_SCALING + X_LOW

    ee_xarm[2] = np.clip(scaled_z, Z_LOW, Z_LIMIT)

    gripper_xarm = np.mean(gripper_pos) / 0.04 * GRIPPER_LIMIT * Z_SCALING * 1
    
    print("##G", gripper_pos, gripper_xarm)
    print(f"step {i}, pos: {ee_xarm}")
    input("wait\n")
    x, y, z = ee_xarm * 1000  # m -> cm
    if gripper_xarm < 550:
        gripper_xarm * 0.95
        gripper_xarm -= 50
    if i > 27:
        # settings for tower builds, 
        if ground_place:
            z -= 100 # for ground blocks due to bounciness
        else:
            # address sim2real problem where in simulator we have perfect drops but in real
            # the blocks we use are a little sticky and the gripper is sticky, causing some imprecisions
            # so we force the gripper to drop from lower heights with this in addition to creating abstract
            # trajectories that are lower in height than usual
            z -= 100 
            z = max(z, (Z_LOW + 1e-2*block_height + 0.05*block_height) * 1000)
        # the above two is equivalent to a simple discretization of position control, although we do not make it that easy
        z = max(z, (Z_LOW+1e-2) * 1000)
        print(z)
        arm.set_position(x=x, y=y, z=z, roll=np.pi, pitch=0, yaw=0, speed=100, wait=True)
        arm.set_gripper_position(gripper_xarm, wait=False, wait_motion=True)
    else:
        if gripper_xarm > 650:
            gripper_xarm += 60 # discretize the gripper position a little to handle discrepency between panda and xarm grippers
        
        # setting for pyramid build, must pick up from higher
        # z += 20
        arm.set_gripper_position(gripper_xarm, wait=False, wait_motion=True)
        arm.set_position(x=x, y=y, z=z, roll=np.pi, pitch=0, yaw=0, speed=100, wait=True)
    

# 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
if __name__ == '__main__':
    # main("./realtower-6-1.pkl")
    # main("./realtower-5-1.pkl")
    # main("./realpyramid-3-1.pkl")
    # main("./realtower2-3-1.pkl")
    # main("./realtower2-3-1.pkl")
    main("./realcustom_mc_scene_4-28.pkl")
# traj['states']  # is a list of robot observations, doesn't include abstract trajectory, attention masks etc.
# traj['actions']  # is a list of robot
# stacked_state = traj['states'][0]  # each state is of shape (stack_size, 36), stack_size=5 here
#
# state = stacked_state[-1][
#         :-4]  # is the observed state in the end. we strip out last 4 dimensions as that's for the actions
# state.shape
#
# # 9 dim qpos, 9 dim qvel, 7 dim block pose, 3 dim ee xyz, 4 dim ee quat
# state[7:9]  # should grip end effect positions
# state[25:28]  # should EE xyz
