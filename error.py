import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("data.csv")

# 计算每个变量相对于第一步的差值
variables = ['helicity_c']
diff = data.copy()
for v in variables:
    diff[v] = data[v] - data[v].iloc[0]

# 绘图
plt.figure(figsize=(10,6))
for v in variables:
    plt.plot(data['t'], diff[v], label=v)

plt.xlabel('t')
plt.ylabel('Δ Value (relative to first step)')
plt.title('Evolution of variables relative to initial step')
plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.savefig("error-plot.png", dpi=750, bbox_inches = 'tight')
plt.show()
