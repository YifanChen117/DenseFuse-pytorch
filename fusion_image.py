# -*- coding: utf-8 -*-
import os
import re
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import time
from network import Dense_Encoder, CNN_Decoder
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

gray = True # True为灰度图像，False为RGB图像
if gray:
    IR_image_path = './test_IR/'
    VIS_image_path = './test_VIS/'
    result_path = './result/test/'
    weight_path = './weight/0406_19-31_Gray_epoch=6.pt'  # 默认预训练权重文件，也可使用自己训练的.pt文件
    print("gray")
else:
    IR_image_path = 'please/input/your/ir image_path/'
    VIS_image_path = 'please/input/your/vis image path/'
    result_path = './result/your/result path/'
    weight_path = './weight/your_weight.pt'
    print("RGB")

# 确保输出目录存在
os.makedirs(result_path, exist_ok=True)

# 初始化时间记录文件
time_file = os.path.join(result_path, "fusion_time.txt")
with open(time_file, 'w') as f:
    f.write("Fusion Time Records\n")
    f.write("===================\n")

print('获取测试设备...')
print("测试设备为：{}...".format(torch.cuda.get_device_name(0)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('开始构建网络...')
in_channel = 1 if gray else 3
out_channel = 1 if gray else 3
Encoder = Dense_Encoder(input_nc=in_channel).to(device)
Decoder = CNN_Decoder(output_nc=out_channel).to(device)

print('开始载入权重...')
checkpoint = torch.load(weight_path)
Encoder.load_state_dict(checkpoint['encoder_state_dict'])
Decoder.load_state_dict(checkpoint['decoder_state_dict'])
print('载入完成！！！')

print('设置网络为评估模式...')
Encoder.eval()
Decoder.eval()

print('载入数据...')

def safe_extract_number(filename):
    """安全地从文件名中提取数字，如果没有数字则返回0"""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

IR_image_list = sorted(os.listdir(IR_image_path), key=safe_extract_number)
VIS_image_list = sorted(os.listdir(VIS_image_path), key=safe_extract_number)

# 过滤掉非图像文件
def is_image_file(filename):
    """检查文件是否为图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)

IR_image_list = [f for f in IR_image_list if is_image_file(f)]
VIS_image_list = [f for f in VIS_image_list if is_image_file(f)]

print(f"找到 {len(IR_image_list)} 个IR图像文件")
print(f"找到 {len(VIS_image_list)} 个VIS图像文件")

if len(IR_image_list) == 0 or len(VIS_image_list) == 0:
    print("错误：未找到有效的图像文件！")
    exit(1)

if len(IR_image_list) != len(VIS_image_list):
    print(f"警告：IR图像数量({len(IR_image_list)})与VIS图像数量({len(VIS_image_list)})不匹配！")
    min_count = min(len(IR_image_list), len(VIS_image_list))
    IR_image_list = IR_image_list[:min_count]
    VIS_image_list = VIS_image_list[:min_count]
    print(f"将处理前 {min_count} 对图像")

tf_list = transforms.Compose([transforms.ToTensor()])

print('开始融合...')
time_list = []  # 初始化时间记录列表

num = 0
for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
    try:
        with torch.no_grad():
            time_start = time.time()
            num += 1
            
            # 读取并处理图像
            IR_image = Image.open(os.path.join(IR_image_path, IR_image_name)).convert("L" if gray else "RGB")
            VIS_image = Image.open(os.path.join(VIS_image_path, VIS_image_name)).convert("L" if gray else "RGB")
            
            # 转换为张量
            IR_tensor = tf_list(IR_image).unsqueeze(0).to(device)
            VIS_tensor = tf_list(VIS_image).unsqueeze(0).to(device)
        
            # 编码融合
            fusion_feature = (Encoder(IR_tensor) + Encoder(VIS_tensor)) / 2
            
            # 解码生成
            Fusion_image = Decoder(fusion_feature)
            
            # CUDA同步（确保时间准确）
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            # 计算耗时
            time_cost = time.time() - time_start
            time_list.append(time_cost)
            
            # 保存结果
            save_image(Fusion_image.cpu().squeeze(), os.path.join(result_path, f"{num}.png"))
            
            # 记录时间到文件
            with open(time_file, 'a') as f:
                f.write(f"{IR_image_name}: {time_cost:.4f} s\n")
            print(f"图像 {IR_image_name} 融合耗时: {time_cost:.4f} 秒")
    
    except Exception as e:
        print(f"处理图像 {IR_image_name} 和 {VIS_image_name} 时出错: {str(e)}")
        # 记录错误到文件
        with open(time_file, 'a') as f:
            f.write(f"{IR_image_name}: ERROR - {str(e)}\n")
        continue

# 计算并记录平均时间
if time_list:
    avg_time = sum(time_list) / len(time_list)
    with open(time_file, 'a') as f:
        f.write(f"\n平均融合时间: {avg_time:.4f} s/图像 (共 {len(time_list)} 张)")
    print(f"\n平均融合时间: {avg_time:.4f} 秒/图像")