import numpy as np
from skilltranslation import DATASET_DIR
import pickle


def extract_block_stack_traj(filename):
    '''
    supporting dataset with 'attention' manually added, final obs only included one blocks info
    '''
    longest_student = 0
    longest_teacher = 0

    with open(filename, 'rb') as file:
        trajectory = pickle.load(file)
    trajectory = trajectory.copy()
    # test with 0
    student_trajectories = trajectory['student']
    teacher_trajectories = trajectory['teacher']
    #assert len(student_trajectories) == 100
    print(len(student_trajectories))
    for i in range(len(student_trajectories)):
        if len(student_trajectories[str(i)]['observations']) > longest_student:
            longest_student = len(student_trajectories[str(i)]['observations'])
        if len(teacher_trajectories[str(i)]['observations']) > longest_teacher:
            longest_teacher = len(teacher_trajectories[str(i)]['observations'])
        #print(student_trajectories[str(i)])
        #student_observation = student_trajectories[str(i)]['observations']
        teacher_observation = teacher_trajectories[str(i)]['observations']
        #student_block_to_move = student_trajectories[str(i)]['attentions']
        teacher_block_to_move = teacher_trajectories[str(i)]['attentions']
        #student_block_to_move = np.append(student_block_to_move,student_trajectories[str(i)]['attentions'][-1])
        #teacher_block_to_move = np.append(teacher_block_to_move,teacher_trajectories[str(i)]['attentions'][-1])
        #student_block_to_move_index = student_block_to_move * 7 + 19
        teacher_block_to_move_index = teacher_block_to_move * 7 + 10


        #student_agent_info = student_observation[:, :16] ## qpos and qvel for student
        teacher_agent_info = teacher_observation[:, :10] ## only xyz of teacher
        #student_block_to_move_info = []
        teacher_block_to_move_info = []
        # for j in range(len(student_observation)):
        #     to_move_index = student_block_to_move_index[j]
        #     block_info = student_observation[j][to_move_index: to_move_index+7]
        #     student_block_to_move_info.append(block_info)
        # student_block_to_move_info = np.array(student_block_to_move_info)

        for j in range(len(teacher_observation)):
            to_move_index = teacher_block_to_move_index[j]
            block_info = teacher_observation[j][to_move_index: to_move_index+7]
            teacher_block_to_move_info.append(block_info)
        teacher_block_to_move_info = np.array(teacher_block_to_move_info)

        #student_obs = np.hstack([student_agent_info, student_block_to_move_info])
        teacher_obs = np.hstack([teacher_agent_info, teacher_block_to_move_info])
        #assert student_obs.shape == (len(student_observation), 23)
        assert teacher_obs.shape == (len(teacher_observation), 17)
        #student_trajectories[str(i)]['observations'] = student_obs
        teacher_trajectories[str(i)]['observations'] = teacher_obs

    trajectory['student'] = None
    trajectory['teacher'] = teacher_trajectories
    # print(trajectory['student']['1']['observations'][0])
    # print(trajectory['teacher']['1']['observation'][0])
    # f_name = DATASET_DIR / "simple_tower/simple_tower.pkl"
    f_name = 'datasets/blockstack/simple_tower.pkl'
    f_name = DATASET_DIR / "simple_tower/fix_seed_processed.pkl"
    f_name = DATASET_DIR / "random_simple_tower/random_simple_tower_processed.pkl"
    f_name = DATASET_DIR / "tower/tower_100_processed.pkl"
    f_name = DATASET_DIR / 'float_teacher/pick_and_place_w_attn.pkl'
    with open(f_name, "wb") as f:
        pickle.dump(trajectory, f)
    print(f'longest student: {longest_student}')
    print(f'longest teacher: {longest_teacher}')


def extract_block_stack_traj_all_blocks(filename):
    '''
    truncating all auxiliary info but leave all blocks poses
    '''
    longest_student = 0
    longest_teacher = 0

    with open(filename, 'rb') as file:
        trajectory = pickle.load(file)
    trajectory = trajectory.copy()
    # test with 0
    student_trajectories = trajectory['student']
    teacher_trajectories = trajectory['teacher']
    assert len(student_trajectories) == 800
    for i in range(len(student_trajectories)):
        if len(student_trajectories[str(i)]['observations']) > longest_student:
            longest_student = len(student_trajectories[str(i)]['observations'])
        if len(teacher_trajectories[str(i)]['observations']) > longest_teacher:
            longest_teacher = len(teacher_trajectories[str(i)]['observations'])
        #print(student_trajectories[str(i)])
        student_observation = student_trajectories[str(i)]['observations']
        teacher_observation = teacher_trajectories[str(i)]['observations']


        student_agent_info = student_observation[:, :16] ## qpos and qvel for student
        teacher_agent_info = teacher_observation[:, :3] ## only xyz of teacher
        student_block_info = student_observation[:, 19: 19 + 7*9]
        teacher_block_info = teacher_observation[:, 19: 19 + 7*9]

        student_obs = np.hstack([student_agent_info, student_block_info])
        teacher_obs = np.hstack([teacher_agent_info, teacher_block_info])

        assert student_obs.shape == (len(student_observation), 16 + 63)
        assert teacher_obs.shape == (len(teacher_observation), 3 + 63)
        student_trajectories[str(i)]['observations'] = student_obs
        teacher_trajectories[str(i)]['observations'] = teacher_obs

    trajectory['student'] = student_trajectories
    trajectory['teacher'] = teacher_trajectories
    # print(trajectory['student']['1']['observations'][0])
    # print(trajectory['teacher']['1']['observation'][0])
    f_name = DATASET_DIR / "random_simple_tower/random_simple_tower_all_blocks.pkl"
    with open(f_name, "wb") as f:
        pickle.dump(trajectory, f)
    print(f'longest student: {longest_student}')
    print(f'longest teacher: {longest_teacher}')


def combine_dict(dict_names=[]):
    from typing import Dict
    agg_dict = dict(
        students=dict(),
        teachers=dict(),
    )
    for dict_name in dict_names:
        with open(dict_name, 'rb') as f:
            dict:Dict = pickle.load(f)

        for key in dict['students'].keys():
            agg_dict['students'][key] = dict['students'][key]

        for key in dict['teachers'].keys():
            agg_dict['teachers'][key] = dict['teachers'][key]








if __name__ == '__main__':
    filename = DATASET_DIR / "float_teacher/float_teacher_data.pkl"
    extract_block_stack_traj(filename)
    #extract_block_stack_traj_all_blocks(filename)