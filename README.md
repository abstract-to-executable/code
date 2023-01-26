# Abstract-to-Executable Trajectory Translation for One-Shot Task Generalization


## Development/Running experiments

First setup conda environment with

```
conda env create -f environment.yml
```

Setup docker if you wish
```
docker build -t abstracttoexecutable .
```

You will also need to intall [SAPIEN](https://sapien.ucsd.edu)

## Requirements

All experiments were run and tested on a RTX 2080 TI with >= 20 cpus and 32GB of RAM.

## Organization

skilltranslation/envs - all environment code

skilltranslation/models - all models / policies

skilltranslation/paper_rl - all policy optimization code

skilltranslation/scripts - various scripts to use and run, including some abstract trajectory generation scripts 



## Reproducing Results

To reproduce training, follow the hyperparameter settings and use scripts/train_translation_online.py cfg=path/to/cfg.yml 

To run evaluation, run scripts/eval_translation.py

## anonymization

Commit using

```
git -c user.name='anon' -c user.email='my@email.org' commit -m 'test'
```
