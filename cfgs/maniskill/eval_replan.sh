
a=(2)
base=$1
for i in ${a[@]}
do  
    python scripts/eval_translation.py cfg=cfgs/maniskill/eval_translation_plan_opentwo.yml \
        model=$2 \
        env_cfg.max_plans=${i} env_cfg.mode=2 save_results_path=results/test/maniskill/replan.${base} \
        env_cfg.sub_goals=True
done
