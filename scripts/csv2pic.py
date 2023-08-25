import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(6, 6))
csvPath = r"E:\speech\专利\csv\alpha0.1beta0.15_csv\wer.csv"
savePath = r'E:\speech\专利\图片\alpha0.1beta0.15_wer.png'
title = "wer vs steps"
x_label = "steps"
y_label = "wer"

# Read the first CSV file
df = pd.read_csv(csvPath)

# Create a line plot for the first CSV file
sns.lineplot(x="Step", y="Value", data=df, linestyle="solid", label="α=0.1,β=0.15")

# Add a title to the plot
plt.title(title)
plt.xlabel(x_label)
plt.ylabel(y_label)
# Add a legend to the plot
plt.legend(loc='lower right', bbox_to_anchor=(1, 0.2))

plt.savefig(savePath)
# Show the plot
plt.show()
