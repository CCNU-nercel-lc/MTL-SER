import subprocess

# 执行run4_2.sh，并将标准输出重定向到run4_2.out文件中
with open("./run4_2.out", "w") as f:
    subprocess.run(['sh', './run4_2.sh'], stdout=f, bufsize=1)
