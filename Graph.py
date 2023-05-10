import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataframe = pd.read_csv("./error_result_2.csv")
nods = np.array(dataframe["nods"])
errors = np.array(dataframe["error"])

fig = plt.figure(figsize = (14, 7))
ax_lin = plt.subplot(121)
ax_log = plt.subplot(122)

ax_lin.set_xlim((nods.min(), nods.max()))
ax_lin.set_ylim((0, errors.max()))
ax_lin.plot(nods, errors)
ax_lin.set_title("Linear scale, error(number of nods)")
ax_lin.set_xlabel("Number of nods")
ax_lin.set_ylabel("Absolute error")
ax_lin.grid()

log_nods, log_errors = np.log(nods), np.log(errors)
ax_log.set_xlim((log_nods.min(), log_nods.max()))
ax_log.set_ylim((log_errors.min(), log_errors.max()))
ax_log.scatter(log_nods, log_errors, label="True")
a, b = np.polyfit(log_nods, log_errors, 1)
ax_log.plot(log_nods, a*log_nods + b, label=f"y = kx + b, k = {a:.2f}, b = {b:.2f}")
ax_log.set_title("Log scale, error(number of nods)")
ax_log.set_xlabel("ln(number of nods)")
ax_log.set_ylabel("ln(error)")
ax_log.legend()
ax_log.grid()

print(a, b)
plt.show()