# 环境配置

## 安装 librealsense 2.54.2

安装包：[librealsense/archive/refs/tags/v2.54.2.tar.gz](https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.54.2.tar.gz)
安装说明，跳过内核修补：[IntelRealSense/librealsense · GitHub](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)

**测试**
```shell
realsense-viewer
```

## 安装 realsense-ros 4.54.1

安装包：[realsense-ros/archive/refs/tags/4.54.1.tar.gz](https://github.com/IntelRealSense/realsense-ros/archive/refs/tags/4.54.1.tar.gz)

```shell
#创建一个 ROS2 工作空间
mkdir -p ~/realsence_warpper/src
cd ~/realsence_warpper/src/  #解压到此
cd ~/realsence_warpper
# 安装依赖项
sudo apt-get install python3-rosdep -y
sudo rosdep init 
rosdep update 
rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
colcon build
# 加入环境
source ~/code/realsence_warpper/install/local_setup.bash
```
**测试**
```
ros2 launch realsense2_camera rs_launch.py 
```

## 安装 json-3.11.3

安装包： https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz

```bash
wget https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
tar -xf v3.11.3.tar.gz 
cd json-3.11.3
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j12
sudo make install
```
 
## 安装其他包

```shell
pip3 install lark
pip3 install numpy==1.24.0
pip3 install minimalmodbus
pip3 install pyserial
```

# 数据采集

## 操作说明

```shell
#系统python环境编译
conda deactivate
rm -rf build/ install/ log/
colcon build
source install/setup.bash

#一键启动脚本
# ros2 launch teleop_twist_joy teleop-launch.py
bash scripts/startrobot.sh start/stop
bash scripts/startcamera.sh start/stop

# 数据采集
# 默认配置文件：config_data_sampler_default.yaml
ros2 run data_sampler data_sampler 

# 数据可视化 参数为epoch
# 默认配置文件：config_visualize_epoch.yaml
python3 scripts/visualize_epoch.py 0

```

操作方式：
- Y 开始收集
- A 长按关闭夹爪
- B 执行预定义动作 
- X 丢弃数据
- upload 结束收集
- 左摇杆 水平移动   ⬆️Y+ ⬇️Y- ➡️X+ ⬅️X-
- 右摇杆 垂直移动   ⬆️Z+ ⬇️Z-  🔄Z轴旋转

## 数据格式

```bash
└── data
    └── sample
        └── test
            ├── 0
            └── 1
                ├── camera0
                ├── camera1
                ├── camera2
                │   ├── 0.png
                │   ├── 1.png
                │   └── 2.png
                └── state.json
```

