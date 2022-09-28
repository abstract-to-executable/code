# generate teacher trajectories
python scripts/maniskill/collect_dataset.py n=480 mode=train cpu=16 save_path=datasets/maniskill/dataset.pkl
python scripts/maniskill/collect_dataset.py n=10 mode=test cpu=10 save_path=datasets/maniskill/dataset_test.pkl
