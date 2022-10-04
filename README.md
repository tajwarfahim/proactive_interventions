# Proactive Interventions in Autonomous Reinforcement Learning
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

This is the official PyTorch implementation of our NeurIPS 2022 paper "When to Ask for Help: Proactive Interventions in Autonomous Reinforcement Learning" by [Annie Xie*](https://anxie.github.io/), [Fahim Tajwar*](https://tajwarfahim.github.io/), [Archit Sharma*](https://architsharma97.github.io/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/). Please see the [project website](https://sites.google.com/view/proactive-interventions) for example results. For any questions/concerns related to the codebase, please reach out to [Fahim Tajwar](mailto:tajwarfahim932@gmail.com).

## Citation

If you use this repo in your research, please consider citing our paper:

```
@inproceedings{xie2022paint,
 author = {Xie, Annie and Tajwar, Fahim and Sharma, Archit and Finn, Chelsea},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {When to Ask for Help: Proactive Interventions in Autonomous Reinforcement Learning},
 volume = {35},
 year = {2022}
}
```

## Installation
Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

**Install the following libraries:**

```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

**Add mujuco/code files to bashrc:**
Add the following lines to ~/.bashrc:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export PYTHONPATH=$PYTHONPATH:~/proactive_interventions/
export PYTHONPATH=$PYTHONPATH:~/proactive_interventions/envs/
export PYTHONPATH=$PYTHONPATH:~/proactive_interventions/envs/sticky_wall_env
```

Make sure the paths you enter matches the actual path in your device instead of the ones in the template above. Then source the bashrc file:

```
source ~/.bashrc
```

**Install dependencies:**

```
conda env create -f paint/conda_env.yml
conda activate paint
```


## Running experiments

First, make sure you are in the "paint" directory.

**Train an episodic PAINT agent (maze):**

```
bash run_scripts/maze.sh
```

**Train a PAINT agent in a continuing task setting (cheetah):**

```
bash run_scripts/cheetah.sh
```

**Train a non-episodic (forward-backward) PAINT agent:**

(Tabletop manipulation)

```
bash run_scripts/tabletop.sh
```

(Peg insertion)

```
bash run_scripts/peg.sh
```

**Monitor results:**

```
tensorboard --logdir paint/exp_local
```

## Acknowledgements

The codebase for the algorithm is built on top of the PyTorch implementation of [DrQ-v2](https://arxiv.org/abs/2107.09645), original codebase linked [here](https://github.com/facebookresearch/drqv2). The codebase for our environments with irreversibility is built on top of the codebase for [EARL Benchmark](https://arxiv.org/abs/2112.09605), original codebase linked [here](https://github.com/architsharma97/earl_benchmark). We thank the authors for providing us with easy-to-work-with codebases.
