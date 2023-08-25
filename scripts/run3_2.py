import subprocess

# 执行run3.sh，并将标准输出重定向到emo_spc_text.out3_2文件中
with open("./final_log2/alpha0.1beta0.txt", "w") as f:
    subprocess.run(['sh', './run3.sh'], stdout=f, bufsize=1)
