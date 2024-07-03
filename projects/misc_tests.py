import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from osgeo import gdal
from typing import Optional, List
from scipy.optimize import curve_fit
from scipy.constants import pi

import core.global_variables
from core.fog_functions import calc_fog_func

from utils.file_management import list_simulations
from core.global_variables import BASE_PATH, DATA_PATH, img_num, img_type, save
from core.processing import get_tiff_image, get_ldr_image, get_exr_image


def analyse_headlight_exrs():
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/misc/headlights/'
    for xdist in [5, 10]:
        images = []
        max_lum = []
        for i in range(15):
            image = get_exr_image([data_path + f'hdr_lowbeaml_x{int(xdist)}_y{i:03d}.exr'])[0]
            max_lum.append(np.max(image[:, :, 1]))
            images.append(image)

        # Calculate angle from distance
        angle = [0]
        for j in range(1, 15):
            angle.append(np.arctan(j / xdist) * 180 / pi)
        plt.plot(angle, max_lum, label=f'X distance: {xdist}m')

    # Plot luminances
    plt.title('Headlight max luminance vs distance from sensor')
    plt.ylabel('Luminance (cd/m^2)')
    plt.xlabel('Angle between light and sensor (degrees)')
    plt.legend()
    plt.show()

    print('Ok')


def analyse_exr_image():
    folders = ['C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/misc/fully_reflective_capture.exr']
    image = get_exr_image(folders)

    # plt.title('Raytraced image output')
    # plt.imshow(image[0])
    # plt.grid(False)
    # plt.show()

    print(f'Max luminance (R): {np.max(image[0][:, :, 0])}')
    print(f'Max luminance (G): {np.max(image[0][:, :, 1])}')
    print(f'Max luminance (B): {np.max(image[0][:, :, 2])}')

    plt.title('Luminance map')
    plt.imshow(image[0][:, :, 2])
    plt.grid(False)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    analyse_headlight_exrs()
    print('All done!')
