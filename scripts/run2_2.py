import subprocess

# 执行run4_2.sh，并将标准输出重定向到emo_spc_text.out3_2文件中
with open("./emo_spc2_2.out", "w") as f:
    subprocess.run(['sh', './run2.sh'], stdout=f, bufsize=1)
