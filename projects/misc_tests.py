import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from osgeo import gdal
from typing import Optional, List
from scipy.optimize import curve_fit
from scipy.constants import pi

import core.global_variables
from core.fog_functions import calc_fog_func

from utils.math_utils import calc_spherical_coord
from utils.file_management import list_simulations
from core.global_variables import BASE_PATH, DATA_PATH, img_num, img_type, save
from core.processing import get_tiff_image, get_ldr_image, get_exr_image

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})


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


def calc_spherical_cam_loc():
    # Calculate the required location of the camera with respect to the highlight when the camera travels in an arc
    # Start with required angles
    angles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # Convert into radians
    for i in range(len(angles)):
        angles[i] = angles[i] * pi / 180
    x, y = calc_spherical_coord(10.0, angles)

    for i in range(len(angles)):
        print(f"Angle: {angles[i]:2f}, X: {x[i]:5f}, Y: {y[i]:5f}")


def plot_surface_test_data():
    # Function for plotting the excel data acquired by testing camera/light/surface tests (simple reflective mirror)
    # Get xlss
    xlss = pd.read_excel('C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/reflective_surface_tests.xlsx')
    spacing = 18

    # Grab data
    distances = xlss['Distance (m)'][:spacing]
    luminances = xlss['Luminance (cd)'].unique()
    measurements = [xlss['Measurement (cd)'][:18], xlss['Measurement (cd)'][18:]]
    theory_values = [xlss['Theoretical Value (cd)'][:18], xlss['Theoretical Value (cd)'][18:]]

    plt.title('Testing light perpendicular to a reflective surface')
    plt.plot(distances, measurements[0], '.', markersize=8, label=f'Measurements, L={luminances[0]}')
    plt.plot(distances, theory_values[0], linewidth=2, label=f'Theory Values, L={luminances[0]}')
    plt.plot(distances, measurements[1], '.', markersize=8, label=f'Measurements, L={luminances[1]}')
    plt.plot(distances, theory_values[1], linewidth=2, label=f'Theory Values, L={luminances[1]}')
    plt.xlabel('Distance of light from surface (m)')
    plt.ylabel('Luminance (cd)')
    plt.semilogy()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_surface_test_data()
    print('All done!')
