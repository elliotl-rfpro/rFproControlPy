from pathlib import Path

# Define global paths. DATA_PATH will be the name of the folder where everything is saved this simulation campaign.
# BASE_PATH = Path(r'C:\Users\ElliotLondon\Documents\PythonLocal\rFproControlPy\data\simperftests\sun\fog_anisotropy')
BASE_PATH = Path(r'C:\Users\ElliotLondon\Documents\PythonLocal\rFproControlPy\data\simperftests\sky\fog_anisotropy')
data_name = 'fog_raytrace_2214_hg000_alb07_l18'
DATA_PATH = BASE_PATH / data_name

# Any additional comments to append to the descriptor file
comments = f'raytraced, lighting test'

# Which autogen file to use? i.e. which scene to analyse?
# autogen_fname = '2kFlatTest'
# autogen_fname = 'SimPerfTests'
autogen_fname = 'SimPerfTests_Plugin'

# Which image to analyse within the folder?
img_num = '0003'

# Image type: LDR is rasteriser, HDR is raytracer
# img_type = 'LDR'
img_type = 'HDR'

# Should the plots be saved automatically?
save = False

# Which sl_settings.json?
# sl_settings = "sl_settings_default"
# sl_settings = "sl_settings_lum_1e9"
# sl_settings = "sl_settings_lum_1e9_months"
# sl_settings = "sl_settings_lum_1e9_turbidity"
sl_settings = "sl_settings_lum_1e9_fog"
# sl_settings = "sl_settings_lum_1e9_clouds"
# sl_settings = "sl_settings_raytrace"
