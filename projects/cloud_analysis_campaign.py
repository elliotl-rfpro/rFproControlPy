"""
Master info
Lat/Long (2kflat/Cardiff): 51.5 N, 3.2W, Solstice zenith: 2008-07-21::14:14, Peak solar luminance: 1.6e9 - 1.9e9
"""
import datetime
import re
import os
import clr
import time
import itertools
import json
import shutil
from typing import List
import sun_investigation
from core.global_variables import sl_settings, autogen_fname, comments

# Load the rFpro.Controller DLL/Assembly
clr.AddReference("C:/rFpro/2023b/API/rFproControllerExamples/Controller/rFpro.Controller")

# Import rFpro.controller and some other helpful .NET objects
from rFpro import Controller
from System import DateTime, Decimal

# Load in the DATA_PATH
DATA_PATH = sun_investigation.DATA_PATH


def find_and_replace(data: List[str], values: dict) -> None:
    # For each value in the dict, go line by line through the config file until there's a string match that isn't a
    # comment. Then, replace the float on this line with the value in the dict. Janky but fast and works well.
    non_float = re.compile(r'[^\d.]+')
    for key in values:
        name = key.replace('_', '-')
        i = 0
        for line in data:
            if line.__contains__(name) and not line.startswith('#'):
                value = non_float.sub('', line)
                data[i] = data[i].replace(value, str(values[key]))
            i += 1


# Simulation settings for the current campaign. Load from configs/sl_settings_default.json
with open(f'../configs/{sl_settings}.json', 'r') as file:
    jdict = json.load(file)
save = jdict['general']['save']
times = jdict['general']['times']
cloudiness = jdict['weather']['cloudiness']
rain = jdict['weather']['rain']
fog = jdict['weather']['fog']

# Check for sweeps
turbidities = jdict['sl_settings']['default_turbidity']
if not isinstance(jdict['sl_settings']['default_turbidity'], list):
    turbidities = [jdict['sl_settings']['default_turbidity']]

# Create an instance of the rFpro.Controller
rFpro = Controller.DeserializeFromFile(f'../configs/autogen/{autogen_fname}.json')

# Static settings
rFpro.DynamicWeatherEnabled = True
rFpro.Camera = 'Static001'
rFpro.ParkedTrafficDensity = Decimal(0.5)
rFpro.Vehicle = 'Hatchback_AWD_Red'
# rFpro.Location = '2kFlat'
rFpro.VehiclePlugin = 'RemoteModelPlugin'

values_dict = {}
# Load the sl config file, set up correct sl settings, then save it
for turbidity in turbidities:
    values_dict.update(jdict['sl_settings'])
    values_dict['default_turbidity'] = turbidity
    with open(r"C:/rFpro/2023b/rFpro/GameData/SharedDX11/slresources/SilverLining.config", 'r') as f:
        sl_data = f.readlines()
    find_and_replace(sl_data, values_dict)
    with open(r"C:/rFpro/2023b/rFpro/GameData/SharedDX11/slresources/SilverLining.config", 'w') as f:
        f.writelines(sl_data)

    # Outer loop for iterating date/time
    for t in times:
        rFpro.StartTime = DateTime.Parse(t)
        # Outer loop for each combination of simulation settings
        for rFpro.Cloudiness, rFpro.Rain, rFpro.Fog in itertools.product(cloudiness, rain, fog):
            # Open the TrainingData.ini file and correctly adjust the saving folder (with date/time)
            with open(r"C:/rFpro/2023b/rFpro/Plugins/WarpBlend/TrainingData.ini", 'r') as f:
                training_data = f.readlines()
            now = datetime.datetime.now()
            folder_time = f'{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}{now.second:02d}'
            save_loc = str(DATA_PATH / f'{folder_time}')
            print(f'Saving at: {save_loc}')
            training_data[1] = 'OutputDir=' + save_loc + '\n'
            with open(r"C:/rFpro/2023b/rFpro/Plugins/WarpBlend/TrainingData.ini", 'w') as f:
                f.writelines(training_data)

            # Write any other comments about the simulation here
            jdict['general']['comments'] = comments + f' time={folder_time}'
            with open(f'../configs/{sl_settings}.json', 'w') as out_file:
                json.dump(jdict, out_file)

            # Connect
            while rFpro.NodeStatus.NumAlive < rFpro.NodeStatus.NumListeners:
                print(f'{rFpro.NodeStatus.NumAlive} of {rFpro.NodeStatus.NumListeners} Listeners connected.')
                time.sleep(1)
            print(f'{rFpro.NodeStatus.NumAlive} of {rFpro.NodeStatus.NumListeners} Listeners connected.')

            rFpro.StartSession()
            # rFpro.ToggleAi()

            t1 = time.time()
            while True:
                t2 = time.time()
                if (t2 - t1) >= 100.0:
                    rFpro.StopSession()
                    break

            # Place a copy of the sl_settings_default.json file into the folder, to track simulation settings
            if not os.path.exists(rf"../configs/{sl_settings}.json"):
                os.makedirs(rf"../configs/{sl_settings}.json")
            shutil.copyfile(rf"../configs/{sl_settings}.json", rf"{save_loc}/{sl_settings}.json", follow_symlinks=True)

            # Now analyse the luminosity of the campaigns
            sun_investigation.analyse_luminosity(folder_time)

print("Simulation campaign complete!\n")
print("\nScript executed successfully. Exiting...")
