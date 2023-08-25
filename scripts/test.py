# 将EMODB语料库文件汇总为csv，有三列，分别为 'file', 'emotion', 'text'
import os
import csv

# 设置数据集目录和输出csv文件路径
csv_file_name = '../dataset/EMODB/wav/'
dataset_dir = r'E:\speech\dataset\EMODB\wav'
output_csv = 'EMODB.csv'

# 创建一个空列表来存储数据
data = []

# 遍历数据集目录中的所有文件
for file in os.listdir(dataset_dir):
    # 获取文件名（不包括扩展名）
    filename = os.path.splitext(file)[0]
    # 获取倒数第二个字母
    letter = filename[-2]
    # 根据倒数第二个字母设置emotion列的值
    if letter in ['B', 'N', 'L']:
        emotion = 'e0'
    elif letter == 'F':
        emotion = 'e1'
    elif letter in ['D', 'W', 'E']:
        emotion = 'e2'
    elif letter in ['A', 'T']:
        emotion = 'e3'
    else:
        emotion = ''
    # 将数据添加到列表中
    data.append([os.path.join(csv_file_name, file), emotion, 'CAN I HELP YOU'])

# 将数据写入csv文件
with open(output_csv, 'w', newline='\n') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'emotion', 'text'])
    writer.writerows(data)