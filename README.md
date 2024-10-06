## DUSDi: Disentangled Unsupervised Skill Discovery for Efficient Hierarchical Reinforcement Learning (NeurIPS 2024)
This codebase was modified based on [URLB](https://github.com/rll-research/url_benchmark).


## Requirements
We assume you have access to a GPU that can run CUDA 11.7. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the installation ends, you can activate your environment with
```sh
conda activate dusdi
```
After the environment is activated, set up the PettingZoom repo:
```sh
git clone https://github.com/JiahengHu/Pettingzoo-skill.git
cd Pettingzoo-skill
pip install -e .
```
Install pytorch with cuda support:
```sh
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
```

## Instructions

### Pre-training
To run pre-training use the `pretrain.py` script. For example:
```sh
python pretrain.py agent=dusdi_diayn domain=particle agent.skill_dim=5 env.particle.N=10 exp_nm="test"
```
The snapshots will be stored under the following directory:
```sh
./models/<obs_type>/<domain>/<agent>/
```

### Downstream Hierarchical Learning
Once you have pre-trained your method, you can use the saved snapshots to learn downstream task. For example:
```sh
python train.py domain=particle ds_task=poison_l low_path="seed:2 particle dusdi_diayn test"
```
Checkout finetune.py for baselines that don't learn skills.


### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in a form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```

