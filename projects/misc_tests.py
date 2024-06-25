import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from osgeo import gdal
from typing import Optional, List
from scipy.optimize import curve_fit
from core.fog_functions import calc_fog_func

from utils.file_management import list_simulations
from core.global_files import BASE_PATH, DATA_PATH, img_num, img_type, save
from core.processing import get_tiff_image, get_ldr_image, get_exr_image

folders = ['C:/rFpro/2023b/rFpro/UserData/hdr000000.exr']
image = get_exr_image(folders)

plt.title('Raytraced image output')
plt.imshow(image[0])
plt.grid(False)
plt.show()

plt.title('Luminance map')
plt.imshow(image[0][:, :, 2])
plt.grid(False)
plt.colorbar()
plt.show()

print('All done!')
