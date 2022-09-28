import gym
import numpy as np
from skilltranslation.envs.blockstack import MultipleBlockStack
import os
from tqdm import tqdm
import os.path as osp
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

def pick_and_place_magic_gripper(env:MultipleBlockStack):

    assert env.goal_coords is not None
    goal_coords = env.goal_coords
    magic_landmarks = []
    magic_traj = dict(
        actions=[],
        observations = [],
        attentions = [],
    )
    for i in env.get_targeted_blocks_idx():
        #print("working on block",i)
        offset = env.get_blocks()[i].pose.p - env.get_articulations().get_root_pose().p
        lm1 = np.array([offset[0], offset[1], 0.3, 0,0,0,0,0])
        lm2 = np.array([offset[0], offset[1], 0.04 + 0.095, 0,0,0,0,0])
        lm3 = lm1.copy()
        goal_coord = goal_coords[i]
        goal_lm1 = np.array([goal_coord[0], goal_coord[1], 0.3,0,0,0,0,0])
        goal_lm2 = np.array([goal_coord[0], goal_coord[1], goal_coord[2]+0.095,0,0,0,0,0])
        goal_lm3 = goal_lm1.copy()
        magic_landmarks.append(lm1) # goto block i
        magic_landmarks.append(lm2) # go down and grab
        magic_landmarks.append(lm3) # go up
        magic_landmarks.append(goal_lm1) # goto above target
        magic_landmarks.append(goal_lm2) # leave at target
        magic_landmarks.append(goal_lm3) # go up
    imgs = []
    for j in range(len(magic_landmarks)):
        internal_clock = j % 6
        temp_clock = 0
        while True:
            cur_qpos = env.get_articulations().get_qpos()
            xx = np.zeros(8)
            xx[:3] = cur_qpos[:3]
            cur_qpos = xx
            delta_qpos = magic_landmarks[j] - cur_qpos
            dist = np.linalg.norm(delta_qpos)
            action=delta_qpos[:7].copy()
            if dist < 0.03:
                if internal_clock == 1:
                    #print(env.get_obs())

                    env.create_magic_drive_with(env.get_blocks()[env.get_targeted_blocks_idx()[int(j // 6)]])
                    action[6] = -0.01
                    #env.connect_nearest_block()
                elif internal_clock == 4:
                    env.stop_magic_drive()
                    action[6] = 0.01
                #print(f"step {internal_clock} takes {temp_clock} steps")
                break
            magic_traj['observations'].append(env.get_obs())
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs,_,_,_ = env.step(action)
            magic_traj['actions'].append(action)
            magic_traj['attentions'].append(env.get_targeted_blocks_idx()[int(j//6)])


            temp_clock+=1

    return magic_traj


def pick_and_place_floating_gripper(env:MultipleBlockStack, render=False):
    assert env.goal_coords is not None
    goal_coords = env.goal_coords
    float_landmarks = []
    float_traj = dict(
        actions=[],
        observations = [],
        attentions = [],
    )

    for i in env.get_targeted_blocks_idx():## TODO: hardcoded number
        offset =  env.get_blocks()[i].pose.p - env.get_articulations().get_root_pose().p
        lm1 = np.array([offset[0], offset[1], 0.3, 0, 0, 0,0,0])
        lm2 = np.array([offset[0], offset[1], 0.11, 0,0,0 ,0,0])
        lm3 = np.array([offset[0], offset[1], 0.4,0,0,0,0,0])
        goal_coord = goal_coords[i]
        goal_lm1 = np.array([goal_coord[0], goal_coord[1], 0.3, 0,0,0,0,0])
        goal_lm2 = np.array([goal_coord[0], goal_coord[1], goal_coord[2]+0.09,0,0,0,0,0])
        float_landmarks.append(lm1) # goto block i
        float_landmarks.append(lm2) # go down and grab
        float_landmarks.append(lm3) # go up
        float_landmarks.append(goal_lm1) # goto above target
        float_landmarks.append(goal_lm2) # leave at target
        float_landmarks.append(goal_lm1) # go up
    #print(env.get_obs())
    imgs = []
    for j in range(len(float_landmarks)):


        internal_clock = j % 6
        temp_clock = 0

        while True:

            cur_qpos = env.get_articulations().get_qpos()
            xx = np.zeros(8)
            xx[:3] = cur_qpos[:3]
            cur_qpos = xx
            delta_qpos = float_landmarks[j] - cur_qpos

            dist = np.linalg.norm(delta_qpos)

            if dist < 0.02:
                break
            if internal_clock in [0,1,5]:
                if internal_clock == 5 and temp_clock < 10:
                    delta_qpos[:3] = [0,0,0]
                delta_qpos[6] = 0.001
            else:
                delta_qpos[6] = -0.002
                if internal_clock == 2 and temp_clock < 30:
                    delta_qpos[:3] = [0,0,0]
            obs = env.get_obs()
            float_traj['observations'].append(obs)
            action = delta_qpos[:7].copy()

            action = np.clip(action, env.action_space.low, env.action_space.high)
            # print(f"heading to {float_landmarks[j]}")
            # print(f"action is {action}")
            # print(f"cur qpos {cur_qpos}")
            float_traj['actions'].append(action)
            float_traj['attentions'].append(env.get_targeted_blocks_idx()[int(j//6)])
            env.step(action)
            #
            if render:
                img = env.render('rgb_array')
                imgs.append(img)
            temp_clock += 1
    float_traj['observations'].append(env.get_obs())
    #print(float_traj['attentions'])
    # return imgs
    if render:
        return imgs
    return float_traj


def generate_dataset():
    '''this dataset will include a attention tag'''
    parent_path=osp.dirname(__file__)
    dataset_path = osp.join(parent_path, 'dataset')
    os.makedirs(dataset_path,exist_ok=True)
    j=0
    while osp.isdir(osp.join(dataset_path,'random_simple_tower-' + str(j))):
        j+=1

    path=osp.join(dataset_path,'random_simple_tower-' + str(j))
    print("dataset dir created at: {}".format(path))
    os.mkdir(path)
    dataset = dict(
        student=dict(),
        teacher=dict(),
    )
    env: MultipleBlockStack
    env_kwargs = dict(
        obs_mode = 'state',
        magic_control = False,
        fix_rotation = True,
        num_blocks = 9,
        goal = 'random_simple_tower',
    )
    env = gym.make('MultipleBlockStack-v1', **env_kwargs)


    vids = []
    for i in tqdm(range(1200)):
        env.reset(seed=i)
        magic_env_goal = env.goal_coords.flatten()
        mag_traj = pick_and_place_magic_gripper(env)
        if env.get_done() is False: print(f"magic trajectory {i} failed")
        env.reset(seed=i)
        float_env_goal = env.goal_coords.flatten()
        assert np.all (float_env_goal == magic_env_goal)
        float_traj = pick_and_place_floating_gripper(env)
        if env.get_done() is False: print(f"float trajectory {i} failed")

        teacher_observation = np.array(mag_traj['observations'])
        teacher_attention = np.array(mag_traj['attentions'])
        student_observation =np.array(float_traj['observations'])
        student_action =np.array(float_traj['actions'])
        student_attention = np.array(float_traj['attentions'])
        # print(
        #     f"teacher observation shape {teacher_observation.shape}, "
        #     f"teacher attention shape {teacher_attention.shape}, "
        #     f"student observation shape {student_observation.shape}, "
        #     f"student action shape {student_action.shape}, "
        #     f"student attention shape {student_attention.shape}")

        dataset['student'][str(i)] = dict(
            observations=student_observation,
            actions=student_action,
            attentions=student_attention,
        )
        dataset['teacher'][str(i)] = dict(
            observations=teacher_observation,
            attentions=teacher_attention,
        )
    #print(dataset['student'][str(0)]['observations'][0])
    dataset_file=open(osp.join(path, "data.pkl"), "wb")
    import pickle
    pickle.dump(dataset, dataset_file)
    dataset_file.close()
    env.close()
    print("done")
    #animate(vids, 'dataset_prev.mp4', fps=40)

if __name__ == '__main__':



    generate_dataset()
    exit(0)
    env: MultipleBlockStack
    env_kwargs = dict(
        obs_mode = 'state_dict',
        magic_control = False,
        fix_rotation = True,
        num_blocks = 9,
        goal = 'random_simple_tower',
    )
    env = gym.make('MultipleBlockStack-v1', **env_kwargs)
    env.seed(0)
    env.reset(seed=1)
    print(env.get_obs()['extra']['obs_goal']['absolute'].reshape(9,3))
    print(env.get_obs()['extra']['obs_blocks']['absolute'].reshape(9,7)[:,:3])
    temp_img = []
    for i in range(100):
        temp_img.append(env.render('rgb_array'))
    animate(temp_img, "checker.mp4", fps=10)

    exit(0)
    vids = []
    for i in range(3):
        # env.reset(seed=i, reconfigure=True)
        # imgs = pick_and_place_magic_gripper(env)
        # vids += imgs

        env.reset(seed=i, reconfigure=True)

        imgs = pick_and_place_floating_gripper(env)
        vids += imgs
        print(env.get_done())
    animate(vids, 'random_simple_tower.mp4', fps=50)
    #
    # # for i in range(4):
    # #     env.reset(seed=i,reconfigure=True)
    # #     imgs = pick_and_place_magic_gripper(env)
    # #     vids += imgs
    # # animate(vids,'pp.mp4',fps=20)