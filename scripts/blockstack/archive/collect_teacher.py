from skilltranslation.envs.blockstacking import BlockStackFloatPandaEnv
import gym
import numpy as np

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

def test_actions():
    env_name = 'BlockStackFloat-v0'
    env: BlockStackFloatPandaEnv = gym.make(env_name,
                          reward_mode='sparse',
                          obs_mode='state_dict',
                          )

    env.seed(42)
    np.random.seed(42)
    env.action_space[env.control_mode].seed(42)
    env.reset(seed=99)
    vids = []
    while True:
        robot_pos = env.get_articulations()[0].get_root_pose().p + env.get_obs()['agent']['qpos'][:3]
        goal_pos = np.array([0.3,0,0.3])
        robot_to_goal_pos = goal_pos - robot_pos
        robot_to_goal_dist = np.linalg.norm(robot_to_goal_pos)
        if robot_to_goal_dist < 3e-2:
            break
        action = np.hstack([robot_to_goal_pos, np.zeros(1)])
        env.step(action)
        vids.append(env.render('rgb_array'))

    robot_pos=env.get_articulations()[0].get_root_pose().p + env.get_obs()['agent']['qpos'][:3]
    print(robot_pos)
    animate(vids, 'test_teacher.mp4', fps=20)
def test_seed():
    env_name = 'BlockStackFloat-v0'
    env:BlockStackFloatPandaEnv = gym.make(
        env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=2,
        goal='pick_and_place',
    )

    env.reset(0)
    print(env.goal_coords)
    for i in range(2):
        print('block',i,':',env.blocks[i].pose.p)
    env.reset(0)
    print(env.goal_coords)
    for i in range(2):
        print('block',i,':',env.blocks[i].pose.p)
def collect_teacher_traj(env: BlockStackFloatPandaEnv, render=False, mode='human'):
    assert env.goal_coords is not None
    float_traj = dict(

        observations=[],
    )
    vids = []
    if render and mode=='human':
        env.render()
        env.get_viewer().paused = True
    # find terminal for 6 phases
    lms = []
    for i in range(len(env.blocks)):
        block_pos = env.blocks[i].pose.p
        lm1 = np.array([block_pos[0], block_pos[1], 0.25, 0.04])
        lm2 = np.array([block_pos[0], block_pos[1], 0.113, 0.04])
        lm3 = np.array([block_pos[0], block_pos[1], 0.25, 0])
        goal_pos = env.goal_coords[i]
        goal_lm1 = np.array([goal_pos[0], goal_pos[1], 0.25, 0])
        goal_lm2 = np.array([goal_pos[0], goal_pos[1], goal_pos[2] + 0.096,0])
        goal_lm3 = np.array([goal_pos[0], goal_pos[1], 0.25, 0.04]) # testing, 0.4 is not a typo
        cur_lms = [lm1, lm2, lm3, goal_lm1, goal_lm2, goal_lm3]
        lms += cur_lms
    total_steps = 0
    
    for j in range(len(lms)):
        phase = j % 6
        steps_in_phase = 0
        print('========== Phase {:d} ==========='.format(j))
        while True:
            robot_qpos = env.get_articulations()[0].get_qpos()
            target_qpos = np.hstack([lms[j], lms[j][-1]]) # dup last dim
            delta_qpos = target_qpos - robot_qpos
            robot_to_target_dist = np.linalg.norm(delta_qpos[:3])
            # if target_qpos[-1] == 0.04:
            #     check_finger_reach_target = robot_qpos[-1] > 0.035
            # else:
            #     check_finger_reach_target = robot_qpos[-1] < 0.026

            gripper_flag = True
            if phase == 2:
                gripper_flag = env.unwrapped._agent.check_grasp(env.blocks[int(j / 6)])
                print('@@', gripper_flag)

            # if robot_to_target_dist < 0.015 and check_finger_reach_target and steps_in_phase > 15:
            #     break
            if robot_to_target_dist < 0.015 and gripper_flag:
                break


            obs_dict = env.get_obs()
            obs = np.zeros(24) # [5] [5] [7] [7]
            obs[:5] = obs_dict['agent']['qpos']
            obs[5:10] = obs_dict['agent']['qvel']
            obs[10:] = obs_dict['extra']['obs_blocks']['absolute']
            float_traj['observations'].append(obs)
            action = delta_qpos.copy()
            action[-2] = lms[j][-1] # position controller for finger
            env.step(np.stack(action[:4]))

            print("{}: dist {:.4f}, act {}, cur qpos {}, target qpos {}".format(
                total_steps, robot_to_target_dist, action[:4], robot_qpos, target_qpos
            ))


            #env.step(action[:4])
            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)

            steps_in_phase += 1
            total_steps += 1
    obs_dict=env.get_obs()
    obs=np.zeros(24)  # [5] [5] [7] [7]
    obs[:5]=obs_dict['agent']['qpos']
    obs[5:10]=obs_dict['agent']['qvel']
    obs[10:]=obs_dict['extra']['obs_blocks']['absolute']
    float_traj['observations'].append(obs)
    if render and mode == 'rgb_array':
        return vids
    return float_traj

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
        num_blocks=2,
        goal='pick_and_place',
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
    env_name = 'BlockStackFloat-v0'
    env:BlockStackFloatPandaEnv = gym.make(
        env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=2,
        goal='pick_and_place',
    )
    env.reset(seed=6)
    # print(env.get_obs()['agent']['qpos'])
    # print(env.action_space)

    vids = collect_teacher_traj(env, render=True, mode='human')
    # vids = collect_teacher_traj(env, render=True, mode='rgb_array')
    animate(vids, 'videos/teacher_example.mp4', fps=24)
