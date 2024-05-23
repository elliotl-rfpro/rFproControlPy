import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

# Init and loads
DATA_PATH = Path(r'C:\Users\ElliotLondon\OneDrive - Anthony Best Dynamics Limited\Documents\Python\rFproControl\data')
current_data_path = DATA_PATH / '20240419_123628'
ego_index = 2

# Find out which frame is the first after init is finished
start_point = 3

# Load dfs incrementally, add to master df
master_df = pd.DataFrame
data = []

pathlist = Path(current_data_path).glob('**/*.csv')
for path in pathlist:
    path_in_str = str(path)
    df = pd.read_csv(path_in_str)
    data.append(df)

# Get each data from element, selecting ego, removing nans
master_df = pd.concat(data)
t = master_df.where(master_df['ID'] == ego_index)['Time(s)'].dropna()[start_point:]
x = master_df.where(master_df['ID'] == 2)['X(m)'].dropna()[start_point:]
y = master_df.where(master_df['ID'] == 2)['Y(m)'].dropna()[start_point:]
z = master_df.where(master_df['ID'] == 2)['Z(m)'].dropna()[start_point:]
speed = master_df.where(master_df['ID'] == ego_index)['Speed(m/s)'].dropna()[start_point:]

plt.plot(t, x, label='X')
plt.plot(t, y, label='Y')
plt.plot(t, z, label='Z')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.show()

plt.plot(t, speed, label='Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()
