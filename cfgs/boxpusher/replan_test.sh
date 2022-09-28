a=(200 100 80 60 40 20)
for i in ${a[@]}
do  
    python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml \
        env_cfg.planner_cfg.min_student_execute_length=${i} \
        model=$1
done

# results/online/boxpusher_v2_translation_final/transformer/2586/models/best_train_EpRet.pt