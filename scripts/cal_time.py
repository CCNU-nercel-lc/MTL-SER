import os
import librosa

folder_path = r"E:\speech\dataset\IEMOCAP\wav"  # 替换为包含音频文件的文件夹路径

max_duration = 0
max_duration_file = ""

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        audio, sr = librosa.load(file_path)
        duration = librosa.get_duration(audio, sr=sr)
        if duration > max_duration:
            max_duration = duration
            max_duration_file = filename

print("最长音频文件：", max_duration_file)
print("时长：", max_duration, "秒")
