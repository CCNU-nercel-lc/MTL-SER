# 从imocap3的train.csv文件中根据每个说话人随机选择50条数据，重写到test.csv文件中
import os

import pandas as pd

# 读取csv文件
df = pd.read_csv('../iemocap3/emotion_speaker_text.train.csv')

if len(df) != 5531:
    print("not full data")
    exit()

# 加入新列speaker
df['speaker'] = ''

# 提取file列中的01M并赋值给speaker列
filename = df['file'].apply(lambda x: os.path.basename(x))
i = 0
for item in filename:
    df['speaker'][i] = item[3:6]
    i = i + 1

# 按照“speaker”列进行分组，并从每个组中随机选择50条数据
selected_data = df.groupby('speaker').apply(lambda x: x.sample(50))

# 获取要删除的数据的索引
index_to_drop = selected_data.index.get_level_values(1)

# 从原始DataFrame中删除选定的数据
df = df.drop(index_to_drop)
# 按照“file”列进行排序
df = df.sort_values(by='file')
selected_data = selected_data.sort_values(by='file')

# 保存修改后的test.csv文件
selected_data.to_csv('iemocap3/emotion_speaker.test.csv', index=False)
# 保存修改后的train.csv文件
df.to_csv('iemocap3/emotion_speaker.train.csv', index=False)
