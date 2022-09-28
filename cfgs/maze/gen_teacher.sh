### Generate abstract trajectory datasets

# Maze v1, used in paper 
python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=2 walk_dist_range="(12, 20)" N=2400 sparsity=2.5e-2

python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=3 walk_dist_range="(12, 20)" N=2400 sparsity=2.5e-2
python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=3 walk_dist_range="(20, 24)" N=2400 sparsity=2.5e-2

python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=4 walk_dist_range="(12, 20)" N=2400 sparsity=2.5e-2
python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=4 walk_dist_range="(20, 24)" N=2400 sparsity=2.5e-2


python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=5 walk_dist_range="(20, 24)" N=2400 sparsity=2.5e-2

python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=6 walk_dist_range="(20, 24)" N=2400 sparsity=2.5e-2


# Maze v2, longer, harder
# world size = 200 here
python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=2 walk_dist_range="(12,24)" N=600 render=False sparsity=1e-2 path=maze_v2
python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=3 walk_dist_range="(12,24)" N=600 render=False sparsity=1e-2 path=maze_v2

# average around 86
python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=5 walk_dist_range="(12,24)" N=1200 render=False sparsity=4e-2 path=maze_v2


# Average of around 120 frames
python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=6 walk_dist_range="(12,30)" N=2400 render=False sparsity=1e-2

python 'skilltranslation/envs/maze/couch/generate_teacher.py' max_walks=6 walk_dist_range="(30, 60)" N=2400 render=False sparsity=1e-2