# Energy decay law ~ 1/t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读数据
df = pd.read_csv("output/data.csv")
t = df["t"]
energy = df["energy"]

plt.figure()
plt.loglog(t, energy, 'o-', label="data")

# 参考斜率 t^{-2/3}
ref = t**(-1)

# 随便选一个常数，把参考线移到数据附近
C = energy.iloc[len(energy)//2] / ref.iloc[len(ref)//2]

plt.loglog(t, C * ref, '--', label=r'ref: $t^{-1}$')

plt.xlabel("t")
plt.ylabel("energy")
plt.legend()
plt.grid(True, which="both")
plt.title("Energy decay with $t^{-1}$ reference slope")
plt.show()
