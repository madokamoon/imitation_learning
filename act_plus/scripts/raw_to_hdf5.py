import os
import json
import h5py
import numpy as np
from PIL import Image
import glob
import pathlib
from tqdm import tqdm  # 添加tqdm库导入

class RawToHDF5Converter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.camera_names = []
        self.datas = {}
        self.record_path = ""
        
    def convert(self):
        folders = [f for f in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, f))]
        if not folders:
            print("输入路径下没有文件夹")
            return
        
        print(f"找到以下文件夹: {folders}")
        print(f"共计 {len(folders)} 个文件夹需要处理")
        
        # 处理每个文件夹
        for folder_idx, folder_name in enumerate(folders):
            print(f"\n---处理文件夹 {folder_idx+1}/{len(folders)}: {folder_name} ---")
            folder_path = os.path.join(self.input_path, folder_name)
            
            # 重置数据结构
            self.camera_names = []
            self.datas = {}
            
            # 检测摄像头文件夹
            self.camera_names = [f for f in os.listdir(folder_path) if f.startswith('camera')]
            print(f"找到以下摄像头: {self.camera_names}")
            print(f"摄像头数量: {len(self.camera_names)}")
            
            # 初始化数据字典
            self.datas = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            
            # 为每个摄像头创建数据项
            for cam_name in self.camera_names:
                self.datas[f'/observations/images/{cam_name}'] = []
            
            # 读取状态文件
            json_path = os.path.join(folder_path, 'state.json')
            if not os.path.exists(json_path):
                print(f"找不到状态文件: {json_path}")
                continue
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # 检查帧数量是否匹配
            frame_count = len(json_data)
            if not self.camera_names:
                print("警告: 没有找到摄像头文件夹，跳过此文件夹")
                continue
                
            camera_frame_count = len(glob.glob(os.path.join(folder_path, self.camera_names[0], '*.png')))
            
            if frame_count != camera_frame_count:
                print(f"警告: JSON文件中的帧数 ({frame_count}) 与摄像头文件夹中的帧数 ({camera_frame_count}) 不匹配")
                continue
            
            print(f"处理 {frame_count} 帧数据...")
            
            # 处理每一帧数据
            for frame_idx_str in tqdm(sorted(json_data.keys(), key=lambda x: int(x)), 
                                     desc=f"处理文件夹 {folder_name}", 
                                     total=frame_count):
                jsondata = json_data[frame_idx_str]
                frame_idx = int(frame_idx_str)
                
                # 添加机器人状态和抓取状态
                self.datas['/observations/qpos'].append(jsondata["robot_state"] + [jsondata['grasp_state'][0]])
                self.datas['/observations/qvel'].append(jsondata['robot_vel_command'] + [jsondata['grasp_action'][0]])
                self.datas['/action'].append(jsondata['robot_action'] + [jsondata['grasp_action'][0]])
                
                # 处理每个摄像头的图像
                for cam_name in self.camera_names:
                    img_path = os.path.join(folder_path, cam_name, f"{frame_idx}.png")
                    if not os.path.exists(img_path):
                        print(f"找不到图像: {img_path}")
                        continue
                    
                    # 读取并处理图像
                    img = Image.open(img_path)
                    
                    # 检查图像尺寸，如果不是480x640就调整
                    if img.size != (640, 480):
                        # print(f"调整图像尺寸从 {img.size} 到 (640, 480)",end='\r')
                        img = img.resize((640, 480))
                    
                    # 确保图像是RGB格式
                    # if img.mode != 'RGB':
                        # print(f"图像模式不正确，转换为RGB: {img.mode}",end='\r')
                        # img = img.convert('RGB')

                    # 转换为numpy数组
                    img_array = np.array(img)
                    self.datas[f'/observations/images/{cam_name}'].append(img_array)
            
            # 设置输出文件路径
            self.record_path = os.path.join(self.output_path, f"episode_{folder_name}.hdf5")
            
            # 保存为HDF5文件
            print(f"正在保存数据到HDF5文件...")
            self.save_to_hdf5()
            
            print(f"已保存到: {self.record_path}")
    
    def save_to_hdf5(self):
        # 获取时间步长
        max_timesteps = len(self.datas['/action'])
        
        with h5py.File(self.record_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            # 设置全局参数
            root.attrs['sim'] = False
            
            # 创建观测组
            obs = root.create_group('observations')
            
            # 创建图像组
            image = obs.create_group('images')
            for cam_name in self.camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), 
                                        dtype='uint8', chunks=(1, 480, 640, 3))
            
            # 创建其他数据集
            qpos = obs.create_dataset('qpos', (max_timesteps, 7))
            qvel = obs.create_dataset('qvel', (max_timesteps, 7))
            action = root.create_dataset('action', (max_timesteps, 7))
            
            # 存入数据
            for name, array in self.datas.items():
                root[name][...] = np.array(array)

def main():

    # 设置输入和输出路径
    rootpath = pathlib.Path(__file__).parent.parent.parent
    input_path = rootpath.joinpath("imitation_learning_ROS2/data/sample/test") 
    output_path = rootpath.joinpath("act_plus/data/sample/test")
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入路径不存在: {input_path}")
        return
    # 检查输出路径是否存在
    if os.path.exists(output_path):
        user_input = input("输出路径已经存在，是否覆盖？(y/n): ")
        if user_input.lower() == 'y':
            import shutil
            shutil.rmtree(output_path)
            print(f"已删除现有目录: {output_path}")
        elif user_input.lower() == 'n':
            print("操作已取消")
            return
        else:
            print("无效输入，操作已取消")
            return
    
    output_path.mkdir(parents=True, exist_ok=True)
    print("----------开始转换---------")
    print("根目录：", rootpath)
    print("输入路径：", input_path)
    print("输出路径：", output_path)


    converter = RawToHDF5Converter(input_path, output_path)
    converter.convert()

    print("----------转换完成----------")

if __name__ == "__main__":
    main()



