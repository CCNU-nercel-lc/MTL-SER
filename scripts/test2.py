# 将imocap的csv文件的emotion列改为speaker
import os

import pandas as pd

# 读取csv文件
df = pd.read_csv('../iemocap2/iemocap_01F.train.csv')

# 将emotion列重命名为speaker
df = df.rename(columns={'emotion': 'speaker'})

# 提取file列中的01M并赋值给speaker列
filename = df['file'].apply(lambda x: os.path.basename(x))
i = 0
for item in filename:
    df['speaker'][i] = item[3:6]
    i = i + 1

# 保存修改后的csv文件
df.to_csv('iemocap2/emotion_speaker.train.csv', index=False)
