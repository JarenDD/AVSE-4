import os
import argparse
from pathlib import Path

def generate_scp(input_dir, output_file, target_dir=None):
    """
    生成WAV文件的SCP列表
    
    参数:
    input_dir (str): 输入WAV文件目录
    output_file (str): 输出SCP文件路径
    target_dir (str, optional): 目标文件目录，默认为None(使用输入目录)
    """
    # 获取所有WAV文件
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                wav_files.append((file, full_path))
    
    # 生成目标路径
    scp_lines = []
    for filename, filepath in wav_files:
        # 生成目标路径
        if target_dir:
            # 使用目标目录替换输入目录的基路径
            rel_path = os.path.relpath(filepath, input_dir)
            target_path = os.path.join(target_dir, rel_path)
        else:
            # 如果未指定目标目录，使用输入目录
            target_path = filepath
        
        # 构建SCP行: 文件名 源路径 目标路径
        scp_line = f"{os.path.splitext(filename)[0]} {filepath} {target_path}"
        scp_lines.append(scp_line)
    
    # 写入SCP文件
    with open(output_file, 'w') as f:
        f.write('\n'.join(scp_lines))
    
    print(f"成功生成SCP文件: {output_file}")
    print(f"共处理 {len(wav_files)} 个WAV文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成WAV文件的SCP列表')
    parser.add_argument('--input_dir', type=str, default='/public/home/qinxy/jarend/1', help='输入WAV文件目录')
    parser.add_argument('--output_file', type=str, default = '/public/home/qinxy/jarend/SkipConvNet/kaiduan1.scp', help='输出SCP文件路径')
    parser.add_argument('--target_dir', type=str, default='/public/home/qinxy/jarend/SkipConvNet/enhance_kaiduan1', help='目标文件目录(可选)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成SCP文件
    generate_scp(args.input_dir, args.output_file, args.target_dir)