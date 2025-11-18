import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data.csv")

plt.plot(df["t"], df["energy"], label="energy")
#plt.plot(df["t"], df["helicity_m"], label="magneticHelicity")
#plt.plot(df["t"], df["helicity_c"], label="crossHelicity")

plt.xlabel("Time")
plt.ylabel("Value")
#plt.title("XXX vs Time")
plt.legend()
#plt.grid(True)

plt.savefig("result.png", dpi=300)
plt.show()
