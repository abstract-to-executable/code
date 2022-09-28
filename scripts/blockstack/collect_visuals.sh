for i in $(seq $1 $2)
do  
    echo $i
    python scripts/blockstack/collect_visual_dataset.py \
        cfg=cfgs/blockstacking/watch_translation.yml save_dir=datasets/blockstack_visual_512/ traj=datasets/blockstack_v4/dataset.pkl \
        traj_id=$i > /dev/null
done