import os
import re

def rename_images(root_folder, offset=1799):
    # 遍历指定文件夹下的所有子文件夹和文件
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            # 判断文件名是否为数字.jpg的形式
            if re.match(r'^\d+\.jpg$', file):
                old_number = int(file.split('.')[0])
                new_number = old_number + offset
                new_name = f"{new_number}.jpg"
                
                old_file_path = os.path.join(subdir, file)
                new_file_path = os.path.join(subdir, new_name)
                
                # 输出重命名信息（可选）
                print(f"重命名: {old_file_path} --> {new_file_path}")
                os.rename(old_file_path, new_file_path)

if __name__ == "__main__":
    # 用户输入要操作的文件夹路径
    folder = "/home/liu.9756/Drone_video/30_60_sec/"
    rename_images(folder)
