# 将csv文件加入speaker列
import os

import pandas as pd

# 读取csv文件
df = pd.read_csv('../iemocap3_2/emotion_speaker_text.train.csv')
# df = pd.read_csv('iemocap3/emotion_speaker.test.csv')
# 加入新列speaker
df['speaker'] = ''

# 提取file列中的01M并赋值给speaker列
filename = df['file'].apply(lambda x: os.path.basename(x))
i = 0
for item in filename:
    df['speaker'][i] = item[3:6]
    i = i + 1

# 保存修改后的csv文件
df.to_csv('../iemocap3_2/emotion_speaker_text.train.csv', index=False)
# df.to_csv('iemocap3/emotion_speaker.test.csv', index=False)
