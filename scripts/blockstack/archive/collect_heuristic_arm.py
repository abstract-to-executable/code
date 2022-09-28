from skilltranslation.envs.blockstacking import BlockStackFloatPandaEnv
import gym
import numpy as np
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper

def animate(imgs, filename="animation.mp4", _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs = imgs["image"]
    print(f"animating {filename}")
    from moviepy.editor import ImageSequenceClip

    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video

        return Video(filename, embed=True)



def collect_teacher_traj(env: BlockStackFloatPandaEnv, render=False, mode='human'):
    assert env.goal_coords is not None
    float_traj = dict(

        observations=[],
    )
    vids = []
    if render and mode=='human':
        env.render()
        env.get_viewer().paused = True


    i_step = 0

    def goto(xyz, gripper_action):
        nonlocal i_step, block
        target = np.array(xyz)
        while True:
            ee_pos = env.unwrapped._agent.hand_link.get_pose().p
            delta_xyz = target - ee_pos
            robot_to_target_dist = np.linalg.norm(delta_xyz)
            vel = np.linalg.norm(env.unwrapped._agent._robot.get_qvel()[:3])

            grasp_flag = env.unwrapped._agent.check_grasp(block)
            gripper_flag = grasp_flag != (gripper_action > 0)

            if robot_to_target_dist < 0.01 and vel<0.05 and gripper_flag:
                break
            action = np.zeros(4)
            action[:3] = delta_xyz.copy()
            action[-1] = gripper_action
            # print(action)
            if 'NormalizeActionWrapper' in str(env):
                action[:3] = action[:3] / env.env.action_space.high[0]
                action[-1] = 1 if gripper_action > 0 else -1
            # print(action)
            print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
                i_step, robot_to_target_dist, action, ee_pos, xyz
            ))
            print(env.unwrapped._agent.hand_link.get_pose())
            env.step(action)

            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step += 1


    for i in range(len(env.blocks)):
        block = env.blocks[i]
        block_pos = env.blocks[i].pose.p
        above_pos = [block_pos[0], block_pos[1], 0.25]
        grasp_pos = [block_pos[0], block_pos[1], 0.113]

        goto(xyz=above_pos, gripper_action=0.04)
        goto(xyz=grasp_pos, gripper_action=0.04)
        goto(xyz=grasp_pos, gripper_action=0)
        goto(xyz=above_pos, gripper_action=0)

        goal_pos = env.goal_coords[i]
        goal_above_pos = [goal_pos[0], goal_pos[1], 0.25]
        release_pos = [goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        goto(xyz=goal_above_pos, gripper_action=0)
        goto(xyz=release_pos, gripper_action=0)
        goto(xyz=release_pos, gripper_action=0.04)
        goto(xyz=goal_above_pos, gripper_action=0.04)

    if render and mode == 'rgb_array':
        return vids
    return None

def generate_dataset():
    import os.path as osp
    import os
    from tqdm import tqdm
    parent_path=osp.dirname(__file__)
    dataset_path = osp.join(parent_path, 'dataset')
    os.makedirs(dataset_path,exist_ok=True)
    j=0
    while osp.isdir(osp.join(dataset_path,'floating_teacher-' + str(j))):
        j+=1
    path=osp.join(dataset_path,'floating_teacher-' + str(j))
    print("dataset dir created at: {}".format(path))
    os.mkdir(path)

    dataset = dict(
        student=dict(),
        teacher=dict(),
    )

    env_name = 'BlockStackFloat-v0'
    env:BlockStackFloatPandaEnv = gym.make(
        env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_train',
    )
    env.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    for i in tqdm(range(20)):
        print(i)
        env.reset(seed=i)
        floating_teacher_traj = collect_teacher_traj(env)
        teacher_observation =np.array(floating_teacher_traj['observations'])
        student_observation = np.zeros(1000)
        student_action = np.zeros(999)
        #print(teacher_observation.shape)
        dataset['teacher'][str(i)] = dict(
            observations = teacher_observation
        )
        dataset['student'][str(i)] = dict(
            observations = student_observation,
            actions = student_action,
        )
    dataset_file=open(osp.join(path, "data.pkl"), "wb")
    import pickle
    pickle.dump(dataset, dataset_file)
    dataset_file.close()
    print("done")

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)

    # generate_dataset()
    # exit()
    #test_seed()
    env_name = 'BlockStackArm-v0'
    env:BlockStackFloatPandaEnv = gym.make(
        env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='tower_7',
    )
    env = ManiSkillActionWrapper(env)
    env = NormalizeActionWrapper(env)
    print(env.action_space)

    for i in range(20):
        env.reset(seed=i)
        # vids = collect_teacher_traj(env, render=True, mode='human')
<<<<<<< HEAD
        vids = collect_teacher_traj(env, render=True, mode='rgb_array')
        animate(vids, 'videos/teacher_{:d}.mp4'.format(i), fps=24)
=======
        vids = collect_heuristic_student_traj(env, render=True, mode='human')
        animate(vids, 'student-videos/teacher_{:d}.mp4'.format(i), fps=24)
>>>>>>> a150dd73bc4a0215fbbac37cc10a5c1a14a0ac44
