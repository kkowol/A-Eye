import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [15.00, 10.0]
plt.rcParams["figure.autolayout"] = True
columns = ['timer', 'throttle_sem', 'brake_sem', 'throttle_safety', 'brake_safety']
df = pd.read_csv("way/to/data/pedal_tracking.csv", usecols=columns)
plt.plot(df.timer, df.throttle_sem)
plt.plot(df.timer, df.brake_safety)
plt.show()