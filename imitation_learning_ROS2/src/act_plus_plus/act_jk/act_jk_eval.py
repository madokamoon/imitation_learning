import pathlib
import sys
import rclpy
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.action import ActionServer
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy

import json
import time
import cv2
import copy
import numpy as np
import time, datetime
import yaml
import shutil

from jk_robot_msgs.action import StrAction

from data_sampler.robot_client import RobotClient, Mode
from data_sampler.xbox_controller import XBOXController
from data_sampler.camera_subscriber import CameraSubscriber
from data_sampler.pyRobotiqGripper import RobotiqGripper

class RobotPolicy(Node):
    def __init__(self, node_name:str, config_file:str):
        super().__init__(node_name)
        # 读取配置文件
        
        file_yaml = 'config/' + config_file
        print("使用配置文件："+ file_yaml )
        rf = open(file=file_yaml, mode='r', encoding='utf-8')
        crf = rf.read()
        rf.close()  # 关闭文件
        yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
        # 采样设置
        self.sample_frequency = yaml_data["sample_frequency"]
        self.img_show = yaml_data["image_show"]
        self.go_on_path = yaml_data["go_on_path"]
        self.sample_prefix_path = yaml_data["sample_prefix_path"]
        self.elapsed_time_show = yaml_data["elapsed_time_show"]
        self.robot_init_pos = np.array(yaml_data["robot_init_pos"])
        self.robot_end_pos = np.array(yaml_data["robot_end_pos"])
        self.end_delay = yaml_data["end_delay"]
        self.camera_names = yaml_data["camera_names"] 
        self.pos_target_d = yaml_data["pos_target_d"]
        # 设置回调组
        self.sensors_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.actions_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.hardware_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        # 动作服务初始化
        self.action_server = ActionServer(
            self,
            StrAction,
            'robot_policy_action',
            self.robot_policy_callback)
        self.get_logger().info("Policy Action 初始化完成")
        # 机器人客户端初始化
        robot_client_config = yaml_data["robot_client"]
        self.robot_client = RobotClient(**robot_client_config)
        self.robot_client.create_robot_client(self, "jk_robot_server", self.hardware_callback_group)

        self.get_logger().info("机器人客户端 初始化完成")

        # 夹爪初始化参数
        self.gripper = RobotiqGripper("/dev/ttyUSB0")
        self.gripper.activate()
        self.gripper.calibrate(0, 40)

        self.get_logger().info("夹爪 初始化完成")
        # 相机初始化
        self.camera_subscirbers = []
        if self.camera_names != []:
            for cameraname in self.camera_names:
                self.camera_subscirbers.append(CameraSubscriber(cameraname))
                self.camera_subscirbers[-1].create_rgb_subscriber(self, "/" + cameraname + "/color/image_raw", 5, self.sensors_callback_group)
                self.get_logger().info(f"""{cameraname} 初始化完成""")
        # XBOX初始化
        xbox_controller_config = yaml_data["xbox_controller"]
        self.xbox_controller = XBOXController(**xbox_controller_config)
        self.xbox_controller.create_joy_subscriber(self, "/joy", 5, self.hardware_callback_group)
        self.xbox_controller.create_control_action_client(self, 'robot_policy_action', self.actions_callback_group)
        self.get_logger().info("XBOX 初始化完成")

        # ---------act----------
        # dp_config = yaml_data["diffusion_policy"]
        # self.dp = DiffusionPolicy(dp_config)
        # self.get_logger().info("DP 初始化完成")
        # ---------act----------

        # 数据采样
        self.data_id = 0
        self.epoch = 0
        self.statedatas = {}
        self.imgdatas = {}

    def robot_policy_callback(self, goal_handle) -> StrAction.Result:
        # 初始化运行周期
        dt = 1 / self.sample_frequency
        last_command = None
        # 初始化录制路径（相机、机器人状态都适配）
        now_time = datetime.datetime.now()
        str_time = now_time.strftime("%Y-%m-%d-%H-%M-%S")
        record_path = pathlib.Path(self.sample_prefix_path)
        if self.go_on_path == "":
            record_path = record_path.joinpath(str_time)
            if record_path.exists():
                self.get_logger().error("给定路径已存在： %s" % str(record_path))
                goal_handle.failed()
                return StrAction.Result(result=goal_handle.request.goal)
            self.go_on_path = str_time
        else:
            record_path = record_path.joinpath(self.go_on_path)
        # 机器人操作
        print("机器人复原...")
        self.robot_client.set_init_pos()
        self.robot_client.set_mode(Mode.ENDVEL.value)
        time.sleep(0.05)
        self.robot_client.set_switch_flag4(1)
        time.sleep(0.05)
        self.gripper.open()

        # ---------act----------
        # self.dp.diffusion_policy_init()
        # ---------act----------

        print("-----开始录制-----")
        # 初始化机器人状态数据录制
        self.start_sample(record_path)
        while 1:
            data = {}
            imgdata = {}
            start_time = time.time()
            # 获取状态（机器人状态、图片）、按钮信息
            pressed_buttons_id, control_command = self.xbox_controller.get_all_infos()
            # X键退出动作服务并删除录制数据
            if self.xbox_controller.X_ID in pressed_buttons_id:
                print("终止动作，删除已录制数据...")
                self.delete_sampled_datas()
                print("-----删除完成-----")
                # 机器人暂停
                self.robot_client.set_switch_flag4(0)
                break

            # 上传键上传录制
            if self.xbox_controller.UPLOAD_ID in pressed_buttons_id:
                # 数据操作
                print("上传采样...")
                self.stop_sample()
                print("-----保存完成-----")
                break

            data["timestamp"] = time.time()
            end_pos = copy.deepcopy(self.robot_client.get_end_pos())
            gripper_pos = self.gripper.getPosition() / 255
            data["grasp_state"] = [gripper_pos]
            end_pos[3:6] *= np.pi / 180  # 单位：° -> rad
            data["robot_state"] = end_pos.tolist() 
            for camera_subscirber in self.camera_subscirbers:
                imgdata[camera_subscirber.camera_name] = camera_subscirber.get_img()
            action = np.zeros(6)

            # ---------act----------
            # act_control_command = self.dp.get_actions(imgdata, data["robot_state"])
            # print("推断耗时: {0:.3f}s".format(time.time() - data["timestamp"]))
            # for i in range(self.dp.steps_per_inference):
            #     action = actions[i]
            #     action[3:6] *= 180 / np.pi


            act_control_command = None

            if not (act_control_command is None):

                # self.robot_client.set_end_vel(act_control_command[0:6])
                # print("control_command: ", [f"{v:.4f}" for v in act_control_command[0:3]])
  
                # self.robot_client.jk_robot_tcp.setEndPos(action)
                # vel_command = (action - self.robot_client.get_end_pos()) / self.dp.dt
                # self.robot_client.set_end_vel(vel_command)
                # if action[6] > 0.5:
                #     self.robot_client.close_gripper()
                # else:
                #     self.robot_client.open_gripper()
                #
                # start_time = time.time()
                # while (time.time() - start_time) < (self.dp.dt):
                #     pass

            if not (control_command is None):
                self.robot_client.set_end_vel(control_command[0:6])
                # print("control_command: ", [f"{v:.4f}" for v in control_command[0:3]])
                latent_time = time.time() - start_time
                if control_command[6] == 1:
                    self.gripper.close()
                    data["grasp_action"] = [1]
                else:
                    self.gripper.open()
                    data["grasp_action"] = [0]
                control_command[0:3] /= 100  # 单位：cm/s -> m/s
                control_command[3:6] *= np.pi / 180  # 单位：°/s -> rad/s
                action[0:6] = control_command[0:6] * (dt - latent_time) + end_pos
                if last_command is not None:
                    action[0:6] += last_command[0:6] * latent_time
                last_command = copy.deepcopy(control_command)
                
                data["robot_action"] = action.tolist()
                data["robot_vel_command"] = control_command[0:6].tolist() 
                self.put_data(data, imgdata)

            # 处理图片显示
            if self.img_show:
                for camera_subscirber in self.camera_subscirbers:
                    img = imgdata[camera_subscirber.camera_name]
                    if not (img is None):
                        cv2.imshow(str(camera_subscirber.camera_name), img )
                cv2.waitKey(1)

            # 处理用时显示
            if self.elapsed_time_show:
                print("用时: {0}s".format(time.time() - start_time))
            # 控制周期
            while (time.time() - start_time) < dt:
                pass
            # self.robot_client.if_thread_close = False
        goal_handle.succeed()
        return StrAction.Result(result=goal_handle.request.goal)

    def destroy_node(self):
        for camera_subscirber in self.camera_subscirbers:
            camera_subscirber.destroy_recorder()
        super().destroy_node()


    def start_sample(self, record_path_prefix: str):
        # 处理文件路径

        record_path = pathlib.Path(record_path_prefix)

        self.record_path = str(record_path)
        print("record_path: {}".format(self.record_path))

        # epoch文件夹
        epoch = 0
        if record_path.exists():
            epoch = len(list(record_path.glob("*")))
        epoch_folder = record_path.joinpath(str(epoch))
        epoch_folder.mkdir(parents=True, exist_ok=True)
        self.epoch_folder = str(epoch_folder)
        print("epoch_folder: {}".format(self.epoch_folder))

        # state.json
        state_json_path = epoch_folder.joinpath("state.json")
        state_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_json_path = str(state_json_path)
        #  camera_name 文件夹
        for cam_name in self.camera_names:
            camera_folder = epoch_folder.joinpath(cam_name)
            camera_folder.mkdir(parents=True, exist_ok=True)

        for cam_name in self.camera_names:
            self.imgdatas[cam_name] = []

    def put_data(self, data: dict, imgdata: dict):
        self.statedatas[str(self.data_id)] = copy.deepcopy(data)
        for camera_subscirber in self.camera_subscirbers:
            self.imgdatas[camera_subscirber.camera_name].append(imgdata[camera_subscirber.camera_name])
        self.data_id += 1

    def stop_sample(self):
        # 保存状态数据
        with open(self.state_json_path, 'w') as json_file:
            json.dump(self.statedatas, json_file, indent=4)
        # 保存图片数据
        for cam_name in self.camera_names:
            camera_folder = pathlib.Path(self.epoch_folder).joinpath(cam_name)
            imgs = self.imgdatas.get(cam_name, [])
            for idx, img in enumerate(imgs):
                if img is not None:
                    img_path = camera_folder.joinpath(f"{idx}.png")
                    cv2.imwrite(str(img_path), img)
        print("数据保存完成:"+self.epoch_folder)
        # 清空数据
        self.statedatas.clear()
        self.imgdatas.clear()
        self.data_id = 0

    def delete_sampled_datas(self):
        self.statedatas.clear()
        self.imgdatas.clear()

        record_path = pathlib.Path(self.epoch_folder)
        if record_path.exists() and record_path.is_dir():
            shutil.rmtree(record_path)
            print(f"Folder {record_path} has been removed.")
        self.data_id = 0


def main():
    rclpy.init()
    robot_policy = RobotPolicy("robot_policy_node",'config_act_eval.yaml')
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot_policy)

    executor.spin()

    robot_policy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

