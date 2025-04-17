# 环境配置

参考：[GitHub - MarkFzp/act-plus-plus:](https://github.com/MarkFzp/act-plus-plus.git)
```shell
conda create -n jk python=3.10
conda activate jk
# 安装ACT依赖包
cd act_plus_plus/detr && pip install -e . && cd ..
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython

# 安装以下内容以运行act-plus中的 Diffusion Policy 但是安装后 numpy 版本冲突 暂不安装
# git clone https://githubfast.com/ARISE-Initiative/robomimic --recurse-submodules
# git checkout r2d2
# pip install -e .
```
**测试**
```shell
python record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/test --num_episodes 1 --onscreen_render
```

## 训练

```bash
python src/act_plus_plus/act_plus_plus/imitate_episodes.py --task_name demo --ckpt_dir training --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --lr 1e-5 --seed 0 --num_steps 2000

```

