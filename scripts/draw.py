import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(6, 6))
csvPath = r"E:\speech\专利\csv\alpha0_csv\alpha0_acc_csv"
savePath = r'E:\speech\专利\图片\alpha0\final\alpha0_acc.png'
title = "acc vs steps"
x_label = "steps"
y_label = "acc"

# Read the first CSV file
df1 = pd.read_csv(csvPath + r"\alpha0beta0.csv")

# Read the second CSV file
df2 = pd.read_csv(csvPath + r"\alpha0beta0.01.csv")

# Read the first CSV file
df3 = pd.read_csv(csvPath + r"\alpha0beta0.1.csv")

# Read the second CSV file
df4 = pd.read_csv(csvPath + r"\alpha0beta1.csv")

# Create a line plot for the first CSV file
sns.lineplot(x="Step", y="Value", data=df1, linestyle="solid", marker="s", label="α=0,β=0")

# Create a line plot for the second CSV file
sns.lineplot(x="Step", y="Value", data=df2, linestyle="dashed", label="α=0,β=0.01")

# Create a line plot for the first CSV file
sns.lineplot(x="Step", y="Value", data=df3, linestyle="dashed", marker="o", label="α=0,β=0.1")

# Create a line plot for the second CSV file
sns.lineplot(x="Step", y="Value", data=df4, marker="^", label="α=0,β=1")

# Add a title to the plot
plt.title(title)
plt.xlabel(x_label)
plt.ylabel(y_label)
# Add a legend to the plot
plt.legend()

plt.savefig(savePath)
# Show the plot
plt.show()
