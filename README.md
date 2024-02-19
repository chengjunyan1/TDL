# Bridging Neural and Symbolic Representations with Transitional Dictionary Learning

 

## Installation

1. First clone the directory. 

```code
git submodule init; git submodule update
```
(If showing error of no permission, need to first [add a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).)

2. Install dependencies.

Create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html), with Python >= 3.10.6 Install [PyTorch](https://pytorch.org/) (version >= 2.0.0). The repo is tested with PyTorch version of 1.10.1 and there is no guarentee that other version works. Then install other dependencies via:
```code
pip install -r requirements.txt
```

## Dataset
The dataset files for LineWorld and LW-G can be generated using the BabyARC engine with the `datasets/BabyARC` submodule, the OmniGlot dataset found [here](https://github.com/brendenlake/omniglot), and ShapeNet can be downloaded [here](https://shapenet.org/download/shapenetcore), the code for preprocessing the datasets are under `dataset/` or directly downloaded via [this link](https://drive.google.com/file/d/15S1RsDy_5MdNq_iUsZCPsRHd9qMqdTBy/view?usp=share_link). If download from the above link, put the downloaded data under the `./datasets/files/` folder.


Download from this anonymous link: https://drive.google.com/file/d/15S1RsDy_5MdNq_iUsZCPsRHd9qMqdTBy/view?usp=share_link 


## Structure
Here we detail the repo's structure:
* The [datasets](https://github.com/TDL/datasets) contains the codes for loading and processing the datasets.
* The [TAE.py](https://github.com/TDL/TAE.py) is the implementation of model
* The [SDE.py](https://github.com/TDL/SDE.py) is the implementation of the diffusion model
* The [attention.py](https://github.com/TDL/attention.py) is the implementation of the attention layers
* The [sampler.py](https://github.com/TDL/sampler.py) is a larger diffusion model based on DDPM
* The [pltrain.py](https://github.com/TDL/pltrain.py) is a basic training script using pytorch lightning
* All files wil 3D are the 3D version of the file for handling 3D voxels.


## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
cheng2024bridging,
title={Bridging Neural and Symbolic Representations with Transitional Dictionary Learning},
author={Junyan Cheng and Peter Chin},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=uqxBTcWRnj}
}
```
