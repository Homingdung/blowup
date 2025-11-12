import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data.csv")

#plt.plot(df["t"], df["ens_rate"], label="ens rate")
#plt.plot(df["t"], df["ReconRate"], label="Reconnection rate")
#plt.plot(df["t"], df["ens_max"], label="ens_max")
plt.plot(df["t"], np.log(df["ens_total"]), label=r"$log(\|w\|_{L^{\infty}} + \|j\|_{L^{\infty}})$")
plt.plot(df["t"], np.log(df["w_max"]), label=r"$log(\|w\|_{L^{\infty}})$")
plt.plot(df["t"], np.log(df["j_max"]), label=r"$log(\|j\|_{L^{\infty}})$")

plt.xlabel("Time")
plt.ylabel("Value")
#plt.title("XXX vs Time")
plt.legend()
#plt.grid(True)

plt.savefig("result.png", dpi=300)
plt.show()
