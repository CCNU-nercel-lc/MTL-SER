# 在run4的基础上加上了tensorboard，重新跑一遍
import subprocess

# 执行run4_2.sh，并将标准输出重定向到run4_2.out文件中
with open("./emo_spc.out4_3_6", "w") as f:
    subprocess.run(['sh', './run4_3.sh'], stdout=f, bufsize=1)
