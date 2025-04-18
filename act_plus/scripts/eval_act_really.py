
# ros2 run your_package eval_act_really --ckpt_dir /path/to/model/checkpoint --ckpt_name policy_best.ckpt --save_trajectory

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from jk_robot_msgs.action import StrAction
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import copy
import pathlib

# 导入ACT相关模块
from constants import FPS, PUPPET_GRIPPER_JOINT_OPEN
from utils import set_seed, compute_dict_mean, detach_dict, calibrate_linear_vel, postprocess_base_action
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from detr.models.latent_model import Latent_Model_Transformer
from einops import rearrange
from torchvision import transforms

# ROS2相关
from data_sampler.robot_client import RobotClient, Mode
from data_sampler.camera_subscriber import CameraSubscriber
from data_sampler.pyRobotiqGripper import RobotiqGripper

class ActEvaluator(Node):
    def __init__(self):
        super().__init__("act_evaluator")
        
        # 解析命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--ckpt_dir', type=str, required=True, help='模型检查点路径')
        parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt', help='模型文件名')
        parser.add_argument('--max_timesteps', type=int, default=400, help='最大时间步')
        parser.add_argument('--camera_topic_prefix', type=str, default='/', help='相机话题前缀')
        parser.add_argument('--serial_port', type=str, default='/dev/ttyUSB0', help='夹爪串口')
        parser.add_argument('--save_trajectory', action='store_true', help='是否保存轨迹')
        
        # 使用sys.argv[1:]来获取ROS2传递的参数而不是节点名称
        import sys
        self.args = parser.parse_args(sys.argv[1:])
        
        # 设置回调组
        self.sensors_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.actions_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.hardware_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        
        # 初始化机器人客户端
        self.robot_client = RobotClient(dt=1/FPS)
        self.robot_client.create_robot_client(self, "jk_robot_server", self.hardware_callback_group)
        self.get_logger().info("机器人客户端初始化完成")
        
        # 初始化夹爪
        self.gripper = RobotiqGripper(self.args.serial_port)
        self.gripper.activate()
        self.gripper.calibrate(0, 40)
        self.get_logger().info("夹爪初始化完成")
        
        # 加载ACT模型和配置
        self.load_model()
        
        # 初始化相机
        self.camera_subscribers = []
        for cam_name in self.camera_names:
            self.camera_subscribers.append(CameraSubscriber(cam_name))
            self.camera_subscribers[-1].create_rgb_subscriber(
                self, 
                f"{self.args.camera_topic_prefix}{cam_name}/color/image_raw", 
                5, 
                self.sensors_callback_group
            )
            self.get_logger().info(f"相机 {cam_name} 初始化完成")
        
        # 初始化动作客户端
        self.action_client = ActionClient(self, StrAction, 'robot_policy_action')
        self.get_logger().info("动作客户端初始化完成")
        
        # 创建保存路径
        if self.args.save_trajectory:
            now = time.strftime("%Y%m%d-%H%M%S")
            self.save_dir = pathlib.Path(self.args.ckpt_dir) / f"real_eval_{now}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"轨迹将保存到: {self.save_dir}")
        
        # 等待动作服务可用
        self.action_client.wait_for_server()
        self.get_logger().info("动作服务已连接，准备开始评估")

    def load_model(self):
        """加载模型和配置"""
        self.get_logger().info(f"正在加载模型: {self.args.ckpt_dir}/{self.args.ckpt_name}")
        
        # 加载配置
        config_path = os.path.join(self.args.ckpt_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
        
        # 设置模型参数
        self.camera_names = self.config['camera_names']
        self.policy_class = self.config['policy_class']
        self.policy_config = self.config['policy_config']
        self.state_dim = self.config['state_dim']
        
        # 加载统计信息
        stats_path = os.path.join(self.args.ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # 设置预处理和后处理函数
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        if self.policy_class == 'Diffusion':
            self.post_process = lambda a: ((a + 1) / 2) * (self.stats['action_max'] - self.stats['action_min']) + self.stats['action_min']
        else:
            self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        
        # 创建并加载模型
        if self.policy_class == 'ACT':
            self.policy = ACTPolicy(self.policy_config)
        elif self.policy_class == 'CNNMLP':
            self.policy = CNNMLPPolicy(self.policy_config)
        elif self.policy_class == 'Diffusion':
            self.policy = DiffusionPolicy(self.policy_config)
        else:
            raise NotImplementedError(f"不支持的策略类型: {self.policy_class}")
        
        ckpt_path = os.path.join(self.args.ckpt_dir, self.args.ckpt_name)
        loading_status = self.policy.deserialize(torch.load(ckpt_path))
        self.get_logger().info(f"模型加载状态: {loading_status}")
        
        self.policy.cuda()
        self.policy.eval()
        
        # 加载VQ模型(如果使用)
        self.vq = self.config['policy_config']['vq']
        if self.vq:
            vq_dim = self.config['policy_config']['vq_dim']
            vq_class = self.config['policy_config']['vq_class']
            self.latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_ckpt_path = os.path.join(self.args.ckpt_dir, 'latent_model_last.ckpt')
            self.latent_model.deserialize(torch.load(latent_model_ckpt_path))
            self.latent_model.eval()
            self.latent_model.cuda()
            self.get_logger().info(f"加载VQ模型: {latent_model_ckpt_path}")
        
        # 设置查询频率
        self.query_frequency = self.policy_config['num_queries']
        if self.config.get('temporal_agg', False):
            self.query_frequency = 1
            self.num_queries = self.policy_config['num_queries']
        
        # 真实机器人需要延迟
        BASE_DELAY = 13
        self.query_frequency -= BASE_DELAY
        
        # 预热模型
        self.get_logger().info("预热模型...")
        dummy_qpos = torch.zeros((1, self.state_dim)).cuda()
        dummy_image = torch.zeros((1, len(self.camera_names), 3, 480, 640)).cuda()
        for _ in range(10):
            with torch.no_grad():
                self.policy(dummy_qpos, dummy_image)
        self.get_logger().info("模型预热完成")

    def get_image(self, rand_crop_resize=False):
        """从相机获取图像并处理"""
        curr_images = []
        for camera_sub in self.camera_subscribers:
            img = camera_sub.get_img()
            if img is None:
                self.get_logger().warn(f"相机 {camera_sub.camera_name} 没有返回图像")
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            curr_image = rearrange(img, 'h w c -> c h w')
            curr_images.append(curr_image)
        
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        if rand_crop_resize:
            original_size = curr_image.shape[-2:]
            ratio = 0.95
            curr_image = curr_image[..., 
                        int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                        int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
            curr_image = curr_image.squeeze(0)
            resize_transform = transforms.Resize(original_size, antialias=True)
            curr_image = resize_transform(curr_image)
            curr_image = curr_image.unsqueeze(0)
        
        return curr_image

    def run_evaluation(self):
        """运行评估"""
        # 准备记录
        qpos_history_raw = np.zeros((self.args.max_timesteps, self.state_dim))
        target_qpos_list = []
        base_action_list = []
        image_list = []
        
        # 准备机器人
        self.robot_client.set_init_pos()
        self.robot_client.set_mode(Mode.ENDVEL.value)
        time.sleep(0.05)
        self.robot_client.set_switch_flag4(1)
        time.sleep(0.05)
        self.gripper.open()
        
        self.get_logger().info("开始运行模型...")
        
        # 设置时间步
        DT = 1 / FPS
        culmulated_delay = 0
        
        # 如果使用时间聚合
        if self.config.get('temporal_agg', False):
            all_time_actions = torch.zeros([self.args.max_timesteps, self.args.max_timesteps+self.num_queries, self.state_dim+2]).cuda()
        
        # 主要评估循环
        with torch.inference_mode():
            time0 = time.time()
            for t in range(self.args.max_timesteps):
                time1 = time.time()
                
                # 获取当前机器人状态
                qpos_numpy = np.array(self.robot_client.get_end_pos())
                qpos_numpy[3:6] *= np.pi / 180  # 角度转弧度
                qpos_history_raw[t] = qpos_numpy
                
                # 预处理状态数据
                qpos = self.pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                # 获取图像数据
                if t % self.query_frequency == 0:
                    curr_image = self.get_image(rand_crop_resize=(self.policy_class == 'Diffusion'))
                    # 保存图像用于可视化
                    image_data = {}
                    for i, cam_name in enumerate(self.camera_names):
                        img = self.camera_subscribers[i].get_img()
                        if img is not None:
                            image_data[cam_name] = img
                    image_list.append(image_data)
                
                # 使用模型预测动作
                if self.policy_class == "ACT":
                    if t % self.query_frequency == 0:
                        if self.vq:
                            vq_sample = self.latent_model.generate(1, temperature=1, x=None)
                            all_actions = self.policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            all_actions = self.policy(qpos, curr_image)
                    
                    if self.config.get('temporal_agg', False):
                        all_time_actions[[t], t:t+self.num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % self.query_frequency]
                
                elif self.policy_class == "Diffusion":
                    if t % self.query_frequency == 0:
                        all_actions = self.policy(qpos, curr_image)
                    raw_action = all_actions[:, t % self.query_frequency]
                
                elif self.policy_class == "CNNMLP":
                    raw_action = self.policy(qpos, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                
                # 后处理动作
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                target_qpos = action[:-2]
                base_action = action[-2:]
                
                # 记录动作
                target_qpos_list.append(target_qpos)
                base_action_list.append(base_action)
                
                # 执行动作
                self.robot_client.set_end_vel(target_qpos)
                
                # 控制夹爪
                if raw_action[-1] > 0.5:  # 假设最后一个值控制夹爪，>0.5表示关闭
                    self.gripper.close()
                else:
                    self.gripper.open()
                
                # 控制时间步
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    self.get_logger().warn(f'警告: 步骤用时 {duration:.3f} s 在步骤 {t} 超过了 DT: {DT} s, 累计延迟: {culmulated_delay:.3f} s')
                
                # 每50步打印进度
                if t % 50 == 0:
                    self.get_logger().info(f"完成步骤: {t}/{self.args.max_timesteps}")
            
            # 保存数据
            if self.args.save_trajectory:
                self.save_trajectory(qpos_history_raw, target_qpos_list, base_action_list, image_list)
            
            # 评估完成
            self.robot_client.set_switch_flag4(0)
            self.robot_client.set_end_vel(np.zeros(self.state_dim))
            self.gripper.open()
            
            fps = self.args.max_timesteps / (time.time() - time0)
            self.get_logger().info(f"评估完成，平均FPS: {fps:.2f}")
    
    def save_trajectory(self, qpos_history, target_qpos_list, base_action_list, image_list):
        """保存轨迹数据"""
        # 保存机器人状态和动作
        np.save(os.path.join(self.save_dir, "qpos_history.npy"), qpos_history)
        np.save(os.path.join(self.save_dir, "target_qpos.npy"), np.array(target_qpos_list))
        np.save(os.path.join(self.save_dir, "base_action.npy"), np.array(base_action_list))
        
        # 保存图像
        for t, images in enumerate(image_list):
            for cam_name, img in images.items():
                cam_dir = os.path.join(self.save_dir, cam_name)
                os.makedirs(cam_dir, exist_ok=True)
                plt.imsave(os.path.join(cam_dir, f"{t:04d}.png"), img)
        
        self.get_logger().info(f"轨迹已保存到: {self.save_dir}")
        
        # 绘制轨迹图
        plt.figure(figsize=(10, 20))
        for i in range(self.state_dim):
            plt.subplot(self.state_dim, 1, i+1)
            plt.plot(qpos_history[:, i], label='Actual')
            plt.plot([q[i] for q in target_qpos_list], label='Target')
            if i == 0:
                plt.legend()
            if i != self.state_dim - 1:
                plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "trajectory_comparison.png"))
        plt.close()

def main():
    rclpy.init()
    
    evaluator = ActEvaluator()
    
    # 使用单独的线程运行评估
    import threading
    eval_thread = threading.Thread(target=evaluator.run_evaluation)
    eval_thread.start()
    
    # 启动ROS2执行器
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(evaluator)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        evaluator.get_logger().info("收到键盘中断，停止评估")
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()
        if eval_thread.is_alive():
            eval_thread.join()

if __name__ == "__main__":
    main()
