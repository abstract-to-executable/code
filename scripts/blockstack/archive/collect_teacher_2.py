from skilltranslation.envs.blockstacking import BlockStackFloatPandaEnv, BlockStackMagicPandaEnv
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

def collect_magic_teacher_traj(env: BlockStackMagicPandaEnv, render=False, mode='human'):
    assert env.connected == False

    assert env.goal_coords is not None

    magic_traj = dict(
        observations=[],
    )
    vids = []
    if render and mode == 'human':
        env.render()
        env.get_viewer().paused = True
    i_step = 0
    def goto(xyz, gripper_action, attn=0):
        assert gripper_action is None
        nonlocal i_step, block
        target = np.array(xyz)
        observations = [] # during this goto()
        while True:
            robot_qpos = env.get_articulations()[0].get_qpos()
            delta_xyz = target - robot_qpos[:3]
            robot_to_target_dist = np.linalg.norm(delta_xyz)
            vel = np.linalg.norm(env.unwrapped._agent._robot.get_qvel()[:3])
            if robot_to_target_dist < 0.01 and vel<0.05:
                break
            action=np.zeros(4)
            action[:3]=delta_xyz.copy()
            #action[-1]=gripper_action
            # print(action)
            if 'NormalizeActionWrapper' in str(env):
                action[:3]=action[:3] / env.env.action_space.high[0]
                # gripper always +1
                action[-1] = +1
            # collect observations
            obs_dict=env.get_obs()
            dense_obs=np.zeros(17)
            dense_obs[:5]=obs_dict['agent']['qpos']
            dense_obs[5:10]=obs_dict['agent']['qvel']
            dense_obs[10:]=obs_dict['extra']['obs_blocks']['absolute'][attn * 7: attn * 7 + 7]
            observations.append(dense_obs)
            # attentions.append(attn)
            print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
                i_step, robot_to_target_dist, action, robot_qpos[:3], xyz
            ))
            assert action [-1] == +1
            env.step(action)

            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step += 1
        magic_traj['observations']+=observations
    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p
        # above_pos=[block_pos[0], block_pos[1], 0.25]
        # grasp_pos=[block_pos[0], block_pos[1], 0.04 + 0.112]
        above_pos = [block_pos[0], block_pos[1], 0.25]
        grasp_pos = [block_pos[0], block_pos[1], 0.113]


        goto(xyz=above_pos, gripper_action=None, attn=i)
        goto(xyz=grasp_pos, gripper_action=None, attn=i)
        env.magic_grasp(block)
        goto(xyz=above_pos, gripper_action=None, attn=i)

        goal_pos=env.goal_coords[i]
        # goal_above_pos=[goal_pos[0], goal_pos[1], 0.25]
        # release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.132]
        goal_above_pos = [goal_pos[0], goal_pos[1], 0.25]
        release_pos = [goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        goto(xyz=goal_above_pos, gripper_action=None, attn=i)
        goto(xyz=release_pos, gripper_action=None, attn=i)
        env.magic_release()
        goto(xyz=goal_above_pos, gripper_action=None, attn=i)
        # last observation and attention:
    obs_dict=env.get_obs()
    dense_obs=np.zeros(17)
    dense_obs[:5]=obs_dict['agent']['qpos']
    dense_obs[5:10]=obs_dict['agent']['qvel']
    dense_obs[10:]=obs_dict['extra']['obs_blocks']['absolute'][i * 7:i * 7 + 7]
    # float_traj['attentions'].append(i) ## here 'i' is the last block
    magic_traj['observations'].append(dense_obs)

    # assert len(float_traj['attentions']) == len(float_traj['observations'])
    if render and mode == 'rgb_array':
        return vids

    if not render:
        return magic_traj
    return None


def collect_teacher_traj(env: BlockStackFloatPandaEnv, render=False, mode='human'):
    assert env.goal_coords is not None
    float_traj = dict(

        observations=[],
        #attentions=[], # ground truth attention
    )
    vids = []
    if render and mode=='human':
        env.render()
        env.get_viewer().paused = True


    i_step = 0

    def goto(xyz, gripper_action,attn=0):
        nonlocal i_step, block
        target = np.array(xyz)
        observations = [] ## obs during this goto()
        #attentions = [] ## attentions during this goto()
        while True:
            robot_qpos = env.get_articulations()[0].get_qpos()
            delta_xyz = target - robot_qpos[:3]
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

            # collect observations
            obs_dict = env.get_obs()
            dense_obs = np.zeros(17)
            dense_obs[:5] = obs_dict['agent']['qpos']
            dense_obs[5:10] = obs_dict['agent']['qvel']
            dense_obs[10:] = obs_dict['extra']['obs_blocks']['absolute'][attn*7: attn*7+7]
            observations.append(dense_obs)
            #attentions.append(attn)
            print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
                i_step, robot_to_target_dist, action, robot_qpos[:3], xyz
            ))
            print(env.unwrapped._agent.hand_link.get_pose())
            env.step(action)


            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step += 1
        float_traj['observations'] += observations
        #float_traj['attentions'] += attentions

    for i in range(len(env.blocks)):
        block = env.blocks[i]
        block_pos = env.blocks[i].pose.p
        above_pos = [block_pos[0], block_pos[1], 0.25]
        grasp_pos = [block_pos[0], block_pos[1], 0.113]

        goto(xyz=above_pos, gripper_action=0.04, attn=i)
        goto(xyz=grasp_pos, gripper_action=0.04, attn=i)
        goto(xyz=grasp_pos, gripper_action=0, attn=i)
        goto(xyz=above_pos, gripper_action=0, attn=i)

        goal_pos = env.goal_coords[i]
        goal_above_pos = [goal_pos[0], goal_pos[1], 0.25]
        release_pos = [goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        goto(xyz=goal_above_pos, gripper_action=0, attn=i)
        goto(xyz=release_pos, gripper_action=0, attn=i)
        goto(xyz=release_pos, gripper_action=0.04, attn=i)
        goto(xyz=goal_above_pos, gripper_action=0.04, attn=i)

    # last observation and attention:
    obs_dict=env.get_obs()
    dense_obs=np.zeros(17)
    dense_obs[:5]=obs_dict['agent']['qpos']
    dense_obs[5:10]=obs_dict['agent']['qvel']
    dense_obs[10:]=obs_dict['extra']['obs_blocks']['absolute'][i*7:i*7+7]
    #float_traj['attentions'].append(i) ## here 'i' is the last block
    float_traj['observations'].append(dense_obs)

   # assert len(float_traj['attentions']) == len(float_traj['observations'])
    if render and mode == 'rgb_array':
        return vids

    if not render:
        return float_traj
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
    env = ManiSkillActionWrapper(env)
    env = NormalizeActionWrapper(env)
    #print(env.action_space)

    for i in tqdm(range(40)):
        #print(i)
        env.reset(seed=i)
        floating_teacher_traj = collect_teacher_traj(env)
        teacher_observation =np.array(floating_teacher_traj['observations'])
        #teacher_attention = np.array(floating_teacher_traj['attentions'])
        #print(teacher_observation.shape)
        student_observation = np.zeros(100)
        student_action = np.zeros(99)
        dataset['teacher'][str(i)] = dict(
            observations = teacher_observation,
            #attentions = teacher_attention
        )
        dataset['student'][str(i)] = dict(
            observations = student_observation,
            actions = student_action,
        )
    dataset_file=open(osp.join(path, "data.pkl"), "wb")
    import pickle
    pickle.dump(dataset, dataset_file)
    dataset_file.close()

    sanity_check_demo = dataset['teacher']['0']['observations']
    for i in range(len(sanity_check_demo)):
        data = sanity_check_demo[i]
        print('qpos',data[:5],'block',data[-7:])

    print("done")

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)

    # generate_dataset()
    # exit()
    #test_seed()
    env_name = 'BlockStackMagic-v0'
    env:BlockStackFloatPandaEnv = gym.make(
        env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1,
        goal='pick_and_place_train',
    )
    env = ManiSkillActionWrapper(env)
    env = NormalizeActionWrapper(env)
    print(env.action_space)
    # import pdb; pdb.set_trace()

    for i in range(1262,1264):#range(20):
        env.reset(seed=i)
        # print(env.grasp_site.pose.p)
        # print(env.get_articulations()[0].get_qpos()[:3])
        # vids = collect_teacher_traj(env, render=True, mode='human')
        vids = collect_magic_teacher_traj(env, render=True, mode='rgb_array')
        animate(vids, 'videos/teacher_{:d}.mp4'.format(i), fps=24)
