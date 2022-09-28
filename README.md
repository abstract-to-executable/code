# Abstract-to-Executable Trajectory Translation for One-Shot Task Generalization

This the anonymized source code for our submission to ICLR 2023

## Development/Running experiments

First setup conda environment with

```
conda env create -f environment.yml
```

Setup docker if you wish

```
docker build -t abstracttoexecutable .
```

## Organization

skilltranslation/envs - all envs

skilltranslation/models - all models / policies

skilltranslation/scripts - various scripts to use and run, including some abstract trajectory generation scripts 



## Reproducing Results

To reproduce training, follow the hyperparameter settings and use scripts/train_translation_online.py cfg=path/to/cfg.yml 



## anonymization

Commit using

```
git -c user.name='anon' -c user.email='my@email.org' commit -m 'test'
```
