from skilltranslation.envs.blockstacking import BlockStackFloatPandaEnv
import gym
import numpy as np
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper
from skilltranslation.envs.blockstacking import BlockStackFloatPandaEnv, BlockStackMagicPandaEnv
import gym
import numpy as np
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper


def animate(imgs, filename="animation.mp4", _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs=imgs["image"]
    print(f"animating {filename}")
    from moviepy.editor import ImageSequenceClip

    imgs=ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video

        return Video(filename, embed=True)


def collect_heuristic_student_traj(env: BlockStackFloatPandaEnv, render=False, mode='human'):
    assert env.goal_coords is not None
    robot_arm_traj=dict(

        observations=[],
        actions=[],
    )
    vids=[]
    if render and mode == 'human':
        env.render()
        env.get_viewer().paused=True

    i_step=0

    def goto(xyz, gripper_action):
        nonlocal i_step, block
        target=np.array(xyz)
        observations=[]  ## during this goto()
        actions=[]
        while True:
            ee_pos=env.unwrapped._agent.hand_link.get_pose().p
            delta_xyz=target - ee_pos
            robot_to_target_dist=np.linalg.norm(delta_xyz)
            vel=np.linalg.norm(env.unwrapped._agent._robot.get_qvel()[:3])

            grasp_flag=env.unwrapped._agent.check_grasp(block)
            gripper_flag=grasp_flag != (gripper_action > 0)

            if robot_to_target_dist < 0.01 and vel < 0.05 and gripper_flag:
                break
            action=np.zeros(4)
            action[:3]=delta_xyz.copy()
            action[-1]=gripper_action
            # print(action)
            if 'NormalizeActionWrapper' in str(env):
                action[:3]=action[:3] / env.env.action_space.high[0]
                action[-1]=1 if gripper_action > 0 else -1
            # print(action)
            # print("{}: dist {:.4f}, act {}, cur xyz {}, target xyz {}".format(
            #     i_step, robot_to_target_dist, action, ee_pos, xyz
            # ))
            # print(env.unwrapped._agent.hand_link.get_pose())
            obs_dict=env.get_obs()
            dense_obs=np.zeros(38)
            dense_obs[:9]=obs_dict['agent']['qpos']
            dense_obs[9:18]=obs_dict['agent']['qvel']
            # only one block -- so
            dense_obs[18:25]=obs_dict['extra']['obs_blocks']['absolute']
            panda_hand=None
            for link in env.get_articulations()[0].get_links():
                if link.name == 'panda_hand':
                    panda_hand=link
            dense_obs[25:28]=panda_hand.pose.p
            dense_obs[28:32]=panda_hand.pose.q
            block_velocity=env.blocks[0].get_velocity()
            block_angular_velocity=env.blocks[0].get_angular_velocity()
            dense_obs[32:35]=block_velocity
            dense_obs[35:38]=block_angular_velocity
            observations.append(dense_obs)
            actions.append(action)

            env.step(action)

            if render:
                if mode == 'rgb_array':
                    vids.append(env.render(mode))
                else:
                    env.render(mode)
            i_step+=1
        robot_arm_traj['observations']+=observations
        robot_arm_traj['actions']+=actions

    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p
        above_pos=[block_pos[0], block_pos[1], 0.25]
        grasp_pos=[block_pos[0], block_pos[1], 0.113]

        goto(xyz=above_pos, gripper_action=0.04)
        goto(xyz=grasp_pos, gripper_action=0.04)
        goto(xyz=grasp_pos, gripper_action=0)
        goto(xyz=above_pos, gripper_action=0)

        goal_pos=env.goal_coords[i]
        goal_above_pos=[goal_pos[0], goal_pos[1], 0.25]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

        goto(xyz=goal_above_pos, gripper_action=0)
        goto(xyz=release_pos, gripper_action=0)
        goto(xyz=release_pos, gripper_action=0.04)
        goto(xyz=goal_above_pos, gripper_action=0.04)
    # last observation
    obs_dict=env.get_obs()
    dense_obs=np.zeros(38)
    dense_obs[:9]=obs_dict['agent']['qpos']
    dense_obs[9:18]=obs_dict['agent']['qvel']
    # only one block -- so
    dense_obs[18:25]=obs_dict['extra']['obs_blocks']['absolute']
    panda_hand=None
    for link in env.get_articulations()[0].get_links():
        if link.name == 'panda_hand':
            panda_hand=link
    dense_obs[25:28]=panda_hand.pose.p
    dense_obs[28:32]=panda_hand.pose.q
    block_velocity=env.blocks[0].get_velocity()
    block_angular_velocity=env.blocks[0].get_angular_velocity()
    dense_obs[32:35]=block_velocity
    dense_obs[35:38]=block_angular_velocity
    robot_arm_traj['observations'].append(dense_obs)

    if render and mode == 'rgb_array':
        return vids
    if not render:
        return robot_arm_traj
    return None


def collect_magic_teacher_traj(env: BlockStackMagicPandaEnv, render=False, mode='human'):
    assert env.connected == False

    assert env.goal_coords is not None

    magic_traj=dict(
        observations=[],
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
        visuals = []
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
            dense_obs=np.zeros(17)
            dense_obs[:5]=obs_dict['agent']['qpos']
            dense_obs[5:10]=obs_dict['agent']['qvel']
            dense_obs[10:]=obs_dict['extra']['obs_blocks']['absolute'][attn * 7: attn * 7 + 7]
            observations.append(dense_obs)
            visuals.append(obs_dict['cameras'])
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
            if i_step > 100:
                break
        magic_traj['observations']+=observations

    for i in range(len(env.blocks)):
        block=env.blocks[i]
        block_pos=env.blocks[i].pose.p
        # above_pos=[block_pos[0], block_pos[1], 0.25]
        # grasp_pos=[block_pos[0], block_pos[1], 0.04 + 0.112]
        above_pos=[block_pos[0], block_pos[1], 0.25]
        grasp_pos=[block_pos[0], block_pos[1], 0.113]

        goto(xyz=above_pos, gripper_action=None, attn=i)
        goto(xyz=grasp_pos, gripper_action=None, attn=i)
        env.magic_grasp(block)
        goto(xyz=above_pos, gripper_action=None, attn=i)

        goal_pos=env.goal_coords[i]
        # goal_above_pos=[goal_pos[0], goal_pos[1], 0.25]
        # release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.132]
        goal_above_pos=[goal_pos[0], goal_pos[1], 0.25]
        release_pos=[goal_pos[0], goal_pos[1], goal_pos[2] + 0.096]

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


def generate_dataset():
    import os.path as osp
    import os
    from tqdm import tqdm
    parent_path=osp.dirname(__file__)
    dataset_path=osp.join(parent_path, 'dataset')
    os.makedirs(dataset_path, exist_ok=True)
    j=0
    while osp.isdir(osp.join(dataset_path, 'heuristic-' + str(j))):
        j+=1
    path=osp.join(dataset_path, 'heuristic-' + str(j))
    print("dataset dir created at: {}".format(path))
    os.mkdir(path)

    dataset=dict(
        student=dict(),
        teacher=dict(),
    )

    teacher_env_name='BlockStackMagic-v0'
    teacher_env: BlockStackMagicPandaEnv=gym.make(
        teacher_env_name,
        reward_mode='sparse',
        obs_mode='rgbd',
        num_blocks=1,
        goal='pick_and_place_train',
    )
    teacher_env=ManiSkillActionWrapper(teacher_env)
    teacher_env=NormalizeActionWrapper(teacher_env)

    student_env_name='BlockStackArm-v0'
    student_env: BlockStackFloatPandaEnv=gym.make(
        student_env_name,
        reward_mode='sparse',
        obs_mode='rgbd',
        num_blocks=1,
        goal='pick_and_place_train',
    )
    student_env=ManiSkillActionWrapper(student_env)
    student_env=NormalizeActionWrapper(student_env)

    for i in tqdm(range(20)):
        student_env.reset(seed=i)
        teacher_env.reset(seed=i)
        assert np.all(student_env.blocks[0].pose.p == teacher_env.blocks[0].pose.p)
        assert np.all(student_env.goal_coords.flatten() == student_env.goal_coords.flatten())

        student_trajectory=collect_heuristic_student_traj(student_env)
        teacher_trajectory=collect_magic_teacher_traj(teacher_env)

        # print('student success', student_env.check_success())
        # print('teacher success', teacher_env.check_success())

        teacher_observation=np.array(teacher_trajectory['observations'])
        student_observation=np.array(student_trajectory['observations'])
        student_actions=np.array(student_trajectory['actions'])
        assert len(student_actions) + 1 == len(student_observation)
        dataset['teacher'][str(i)]=dict(
            observations=teacher_observation
        )
        dataset['student'][str(i)]=dict(
            observations=student_observation,
            actions=student_actions
        )
        # print(
        #     f"teacher observation shape {teacher_observation.shape}, "
        #     f"student observation shape {student_observation.shape}, "
        #     f"student action shape {student_actions.shape} ")

    dataset_file=open(osp.join(path, "data.pkl"), "wb")
    import pickle
    pickle.dump(dataset, dataset_file)
    dataset_file.close()

    sanity_check_demo=dataset['teacher']['0']['observations']
    for i in range(len(sanity_check_demo)):
        data=sanity_check_demo[i]
        print('qpos', data[:5], 'block', data[-7:])
    sanity_check_stduent=dataset['student']['0']['observations']
    for i in range(len(sanity_check_stduent)):
        data=sanity_check_stduent[i]
        print(i, 'grippe', data[7:9], 'block', data[18:18 + 3], 'hand', data[25:28])
    print("done")


if __name__ == '__main__':
    generate_dataset()
