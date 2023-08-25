# 将全数据的csv文件进行划分，根据说话人信息，从训练集中按1:9的比例随机选择数据放入测试集中
import os

import pandas as pd
import random

# 读取原始CSV文件
df = pd.read_csv('../iemocap3_2/emotion_speaker_text.train.csv')

# 防止重复执行该脚本
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

# 创建一个空的DataFrame用于保存抽样结果
new_df = pd.DataFrame(columns=df.columns)

# 创建一个空的DataFrame用于保存剩余的数据
remaining_df = pd.DataFrame(columns=df.columns)

# 遍历每种speaker类型
for speaker in df['speaker'].unique():
    # 根据speaker列的值筛选出对应的数据
    subset = df[df['speaker'] == speaker].copy()

    # 计算需要抽样的数量
    sample_size = len(subset) // 10

    print("{}对应的总数据为{}, 抽取{}条数据".format(speaker, len(subset), sample_size))

    # 随机抽样
    sample = subset.sample(n=sample_size, random_state=42)

    # 将抽样结果添加到新的DataFrame中
    new_df = pd.concat([new_df, sample])

    # 将剩余的数据添加到remaining_df中
    remaining = subset.drop(sample.index)
    remaining_df = pd.concat([remaining_df, remaining])

# 保存测试集
print("共抽取{}条数据".format(len(new_df)))
new_df.to_csv('../iemocap3_2/emotion_speaker_text.test.csv', index=False)

# 将剩余的数据保存回原始文件
remaining_df.to_csv('../iemocap3_2/emotion_speaker_text.train.csv', index=False)