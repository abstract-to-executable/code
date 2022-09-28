import pickle
from omegaconf import OmegaConf
from skilltranslation.envs.blockstacking import BlockStackFloatPandaEnv
import gym
import numpy as np
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper
import os.path as osp
from skilltranslation.envs.blockstacking.env import BlockStackMagicPandaEnv
from skilltranslation.utils.sampling import resample_teacher_trajectory

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


def collect_magic_teacher_traj(env: BlockStackMagicPandaEnv, render=False, mode='human', change_in_height=0, change_in_goal_height=0):
    assert env.connected == False

    assert env.goal_coords is not None
    init_xyz = env.get_articulations()[0].get_qpos()[:3]
    magic_traj=dict(
        observations=[],
        attns=[],
    )
    vids=[]
    if render and mode == 'human':
        env.render()
        env.get_viewer().paused=True
    i_step=0

    def goto(xyz, gripper_action, attn=0):
        assert gripper_action is None
        nonlocal i_step, block
        target=np.array(xyz)
        observations=[]  # during this goto()
        attns = []
        while True:
            robot_qpos=env.get_articulations()[0].get_qpos()
            delta_xyz=target - robot_qpos[:3]
            robot_to_target_dist=np.linalg.norm(delta_xyz)
            vel=np.linalg.norm(env.unwrapped._agent._robot.get_qvel()[:3])
            if robot_to_target_dist < 0.01 and vel < 0.05:
                break
            action=np.zeros(4)
            action[:3]=delta_xyz.copy()
            # action[-1]=gripper_action
            # print(action)
            if 'NormalizeActionWrapper' in str(env):
                action[:3]=action[:3] / env.env.action_space.high[0]
                # gripper always +1
                action[-1]=+1
            # collect observations
            obs_dict=env.get_obs()
            dense_obs=np.zeros(12)
            dense_obs[:5]=obs_dict['agent']['qpos']
            # dense_obs[5]=obs_dict['agent']['qvel']
            dense_obs[5:]=obs_dict['extra']['obs_blocks']['absolute'][attn * 7: attn * 7 + 7]
            observations.append(dense_obs)
            attns.append(attn)
            # attentions.append(attn)
            # print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
            #     i_step, robot_to_target_dist, action, robot_qpos[:3], xyz
            # ))
            assert action[-1] == +1
            env.step(action)

            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step+=1
        magic_traj['observations']+=observations
        magic_traj["attns"] += attns

    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p
        # above_pos=[block_pos[0], block_pos[1], 0.25]
        # grasp_pos=[block_pos[0], block_pos[1], 0.04 + 0.112]
        above_pos=[block_pos[0], block_pos[1], 0.25+change_in_height]
        grasp_pos=[block_pos[0], block_pos[1], 0.113]

        goto(xyz=above_pos, gripper_action=None, attn=i)
        goto(xyz=grasp_pos, gripper_action=None, attn=i)
        env.magic_grasp(block)
        goto(xyz=above_pos, gripper_action=None, attn=i)

        goal_pos=env.goal_coords[i]
        # goal_above_pos=[goal_pos[0], goal_pos[1], 0.25]
        # release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.132]
        goal_above_pos=[goal_pos[0], goal_pos[1], goal_pos[2]+0.25+ change_in_goal_height]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        goto(xyz=goal_above_pos, gripper_action=None, attn=i)
        goto(xyz=release_pos, gripper_action=None, attn=i)
        env.magic_release()
        goto(xyz=goal_above_pos, gripper_action=None, attn=i)

        # back to init
        print("go back to init")
        goto(xyz = init_xyz, gripper_action=None, attn=i)
        print("finished init")
        # last observation and attention:
    obs_dict=env.get_obs()
    dense_obs=np.zeros(12)
    dense_obs[:5]=obs_dict['agent']['qpos']
    dense_obs[5:]=obs_dict['extra']['obs_blocks']['absolute'][i * 7:i * 7 + 7]
    magic_traj['observations'].append(dense_obs)

    magic_traj['attns'] = np.array(magic_traj['attns'], dtype=int)
    magic_traj['observations'] = np.array(magic_traj['observations'], dtype=np.float32)
    # assert len(float_traj['attentions']) == len(float_traj['observations'])
    if render and mode == 'rgb_array':
        return magic_traj, vids

    if not render:
        return magic_traj
    return magic_traj


def collect_teacher_traj(env: BlockStackFloatPandaEnv, render=False, mode='human', fixed_gripper=True):
    assert env.goal_coords is not None
    float_traj = dict(
        observations=[],
        attns=[],
    )
    vids = []
    init_xyz = env.get_articulations()[0].get_qpos()[:3]
    if render and mode=='human':
        env.render()
        env.get_viewer().paused = True


    i_step = 0

    def goto(xyz, gripper_action, attn=0):
        nonlocal i_step, block
        target = np.array(xyz)
        observations = []
        attns = []
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
            obs_dict=env.get_obs()
            dense_obs=np.zeros(12)
            dense_obs[:5]=obs_dict['agent']['qpos']
            if fixed_gripper:
                dense_obs[3:5] = np.array([0.04, 0.04])
            # dense_obs[5:10]=obs_dict['agent']['qvel']
            dense_obs[5:]=obs_dict['extra']['obs_blocks']['absolute'][attn * 7: attn * 7 + 7]
            observations.append(dense_obs)
            attns.append(attn)
            # print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
            #     i_step, robot_to_target_dist, action, ee_pos, xyz
            # ))
            # print(env.unwrapped._agent.hand_link.get_pose())
            env.step(action)

            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step += 1
        float_traj["observations"] += observations
        float_traj["attns"] += attns


    for i in range(len(env.blocks)):
        block = env.blocks[i]
        block_pos = env.blocks[i].pose.p
        goal_pos = env.goal_coords[i]
        transport_height = goal_pos[2] + 0.12

        above_pos = [block_pos[0], block_pos[1], 0.25]
        grasp_pos = [block_pos[0], block_pos[1], 0.113]
        start_transport_pos = [block_pos[0], block_pos[1], transport_height]
        end_transport_pos = [goal_pos[0] - 0.04, goal_pos[1], transport_height]

        goto(xyz=above_pos, gripper_action=0.04, attn=i)
        goto(xyz=grasp_pos, gripper_action=0.04, attn=i)
        goto(xyz=grasp_pos, gripper_action=0, attn=i)
        goto(xyz=start_transport_pos, gripper_action=0, attn=i)
        goto(xyz=end_transport_pos, gripper_action=0, attn=i)

        goal_above_pos = [goal_pos[0], goal_pos[1], transport_height]
        release_pos = [goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        goto(xyz=goal_above_pos, gripper_action=0, attn=i)
        goto(xyz=release_pos, gripper_action=0, attn=i)
        goto(xyz=release_pos, gripper_action=0.04, attn=i)
        goto(xyz=goal_above_pos, gripper_action=0.04, attn=i)
        goto(xyz=end_transport_pos, gripper_action=0.04, attn=i)
        goto(xyz = init_xyz, gripper_action=0.04, attn=i)

    # if render and mode == 'rgb_array':
    #     return vids
    float_traj['attns'] = np.array(float_traj['attns'], dtype=int)
    float_traj['observations'] = np.array(float_traj['observations'], dtype=np.float32)
    return float_traj


if __name__ == '__main__':
    cfg = OmegaConf.from_cli()
    from tqdm import tqdm
    np.set_printoptions(suppress=True, precision=3)
    save_path = cfg.save_path
    # size = cfg.size
    goal = cfg.goal
    N = 2
    if "n" in cfg:
        N = cfg.n
    env_name = 'BlockStackMagic-v0'
    env = gym.make(
        env_name,
        reward_mode='sparse',
        obs_mode='state_dict',
        num_blocks=1 if "train" in goal else -1,
        goal=goal,
    )
    env = ManiSkillActionWrapper(env)
    env = NormalizeActionWrapper(env)
    dataset = dict(teacher=dict())
    for i in tqdm(range(N)):
        env.reset(seed=i)
        # change_height = np.random.uniform(0,0.36)
        # change_goal_height = np.random.uniform(0,0.15)
        traj = collect_magic_teacher_traj(env, render=True, mode='human', change_in_goal_height=0.1, change_in_height=0.18)
        max_block_id = traj["attns"].max()
        all_obs = []
        attns = []
        for b_id in sorted(np.unique(traj["attns"])):
            obs = traj["observations"][np.where(traj["attns"] == b_id)]
            obs = resample_teacher_trajectory(obs, max_dist=4e-2)
            dataset["teacher"][f"{i}-{b_id}"] = dict(observations=obs)
            attns.append(np.ones(len(obs), dtype=int)*b_id)
            all_obs.append(obs)
        all_obs = np.concatenate(all_obs)
        attns = np.concatenate(attns)
        dataset["teacher"][f"{i}"] = dict(observations=all_obs, attns=attns)
        # do even resampling of teacher trajectory.
        # traj = resample_teacher_trajectory(traj["observations"])
        # dataset["traj"]
        # import pdb;pdb.set_trace()
        # vids = collect_teacher_traj(env, render=True, mode='rgb_array')
        # animate(vids, 'videos/teacher_{:d}.mp4'.format(i), fps=24)
    with open(osp.join(save_path), "wb") as f:
        pickle.dump(dataset, f)