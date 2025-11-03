import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

#plt.plot(df["time"], df["ens_rate"], label="ens rate")
plt.plot(df["time"], df["ReconRate"], label="Reconnection rate")
#plt.plot(df["time"], df["ens_max"], label="ens_max")

plt.xlabel("Time")
plt.ylabel("Value")
#plt.title("XXX vs Time")
plt.legend()
#plt.grid(True)

plt.savefig("result.png", dpi=300)
plt.show()
