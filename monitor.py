import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

#plt.plot(df["time"], df["beta"], label="beta")
#plt.plot(df["time"], df["ens_product"], label="ens production")
#plt.plot(df["time"], df["ReconRate"], label="Reconnection rate")
plt.plot(df["time"], df["j_max"], label="jmax")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("XXX vs Time")
plt.legend()
#plt.grid(True)

plt.savefig("name.png", dpi=300)
plt.show()
