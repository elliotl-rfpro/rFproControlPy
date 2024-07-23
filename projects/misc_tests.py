import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from osgeo import gdal
from typing import Optional, List
from scipy.optimize import curve_fit
from scipy.constants import pi
import OpenEXR as exr
import Imath

import core.global_variables
from core.fog_functions import calc_fog_func

from utils.math_utils import calc_spherical_coord
from utils.file_management import list_simulations
from core.global_variables import BASE_PATH, DATA_PATH, img_num, img_type, save
from core.processing import get_tiff_image, get_ldr_image, get_exr_image

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})


def analyse_headlight_exrs_x():
    # Analyse a sequence of images where a camera is a fixed X distance away from a headlight, moving along the Y axis.
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


def analyse_headlight_exrs_theta(angles):
    # Analyse a sequence of images where a camera moves in an arc around the headlight, always facing towards it.
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/misc/headlights/'
    for xdist in [5, 10]:
        images = []
        max_lum = []
        avg_lum = []
        n = 10
        for i in range(len(angles)):
            image = get_exr_image([data_path + f'hdr_lowbeaml_circle_x{int(xdist)}_theta{angles[i]:02d}.exr'])[0]
            max_lum.append(np.max(image[:, :, 1]))
            # Mean of N largest values
            avg_lum.append(np.mean(np.sort(image[:, :, 1].reshape(-1))[::-1][:n]))
            images.append(image)

        plt.plot(angles, max_lum, label=f'Max Luminance, X = {xdist}m')
        plt.plot(angles, avg_lum, '-.', label=f'Avg Luminance, X = {xdist}m')

    # Plot luminances
    plt.title('Headlight max luminance vs distance from sensor')
    plt.ylabel('Luminance (cd/m^2)')
    plt.xlabel('Angle between light and sensor (degrees)')
    plt.legend()
    plt.show()

    print('Ok')


def analyse_headlight_ies():
    # Load in a .exr image from a converted IES profile and process it for comparison with measured data.
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/misc/headlights/'
    exrfile = exr.InputFile(data_path + 'LowBeam.exr')
    header = exrfile.header()

    # Get headers, width, height
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channel_data = dict()

    # Convert all channels in the image to np arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.frombuffer(C, dtype=np.float32)
        C = np.reshape(C, isize)
        channel_data[c] = C

    # Reconstruct image
    img = channel_data['Y']

    # In sperical coords (x = {0, 360}, y = {-90, 90}). Adjust so that x = {-180, 180}.
    # Cut into two pieces and resew together
    img_c = np.rot90(np.concatenate((img[:, 1800:], img[:, :1800]), axis=1), 2)
    img_c_ = img_c[750:900, 1400:2200]

    # Get a horizontal lineout of the headlight profile. Will correspond to slice of an arc (angular dependence)

    plt.imshow(img_c_)
    plt.grid(False)
    plt.show()

    print(f'Max in img_c_: {np.max(img_c_)}')


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


def calc_spherical_cam_loc(angles_deg, distance_x):
    # Calculate the required location of the camera with respect to the highlight when the camera travels in an arc
    # Start with required angles
    # Convert into radians
    angles_rad = []
    for i in range(len(angles_deg)):
        angles_rad.append(angles_deg[i] * pi / 180)
    x, y = calc_spherical_coord(distance_x, angles_rad)

    for i in range(len(angles_deg)):
        print(f"Angle deg: {angles_deg[i]:2f}, Angle rad: {4.71239 - angles_rad[i]:2f}, X: {x[i]:5f}, Y: {y[i]:5f}")


def plot_surface_test_data():
    # Function for plotting the excel data acquired by testing camera/light/surface tests (simple reflective mirror)
    # Get xlss
    xlss = pd.read_excel('C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/reflective_surface_tests.xlsx',
                         sheet_name='Luminances')
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


def plot_surface_test_data_angles(angles):
    # Function for plotting the excel data acquired by testing camera/light/surface tests (simple reflective mirror)
    # This function deals with processing for a progressively changing camera angle
    # Get xlss
    xlss = pd.read_excel('C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/reflective_surface_tests.xlsx',
                         sheet_name='Angles')

    # Grab data
    angles_deg = xlss['Angle (deg)']
    max_values = xlss['Max Value (cd)']
    # avg_values = xlss['Avg Value (cd)']
    theory_values = xlss['Theoretical Value (cd)']
    theory_curve = xlss['Cos Diff']

    # Load the exrs to find a fixed pixel to evaluate
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/misc/'
    fix_values = []
    samples = [[720, 960], [715, 965], [707, 967], [718, 963], [720, 966]]
    for sample in samples:
        tmp = []
        images = []
        for angle in angles:
            image = get_exr_image([data_path + f'fully_reflective_l1_10m_theta{angle:02d}.exr'])[0]
            images.append(image[:, :, 1])
            tmp.append(image[:, :, 1][sample[0], sample[1]])
        fix_values.append(tmp)

    # Find average values (excluding "dead zone")
    avg_values = []
    for image in images:
        avg_values.append(np.mean(np.sort(image.reshape(-1))[::-1][1:20]))

    plt.title('Testing light 10m away from a reflective surface, various angles')
    # plt.plot(angles_deg, max_values, '.', markersize=8, label=f'Max Value')
    # plt.plot(angles_deg, max_values[0] * theory_curve, linewidth=2, label=f'Theory Values for Max')
    plt.plot(angles_deg, avg_values, '.-', markersize=8, label=f'Avg Value')
    # plt.plot(angles_deg, avg_values[0] * theory_curve, linewidth=2, label=f'Theory Values for Avg')
    for i in range(len(fix_values)):
        plt.plot(angles_deg, fix_values[i], '.-', markersize=8, label=f'Pixel {samples[i]}')
    plt.xlabel('Angle between camera and surface (deg)')
    plt.ylabel('Luminance (cd)')
    plt.legend()
    plt.show()

    # Plot contour of each slice
    for j in range(len(images)):
        plt.plot(images[j][720, 655:1265], '.', label=f'Angle = {angles[j]}')
    plt.title('Slice for central row (pixel 720)')
    plt.xlabel('Pixel Index [x]')
    plt.ylabel('Luminance (cd)')
    plt.legend()
    plt.show()


def plot_surface_contours(smoothnesses):
    # Function for plotting a sequence of surfaces at a fixed distance from a camera

    # Load the exrs to find a fixed pixel to evaluate
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/misc/'
    images = []
    for s in smoothnesses:
        image = get_exr_image([data_path + f'smooth{s:03d}_l1_10m_fov25_noff.exr'])[0]
        images.append(image[:, :, 1])

    # Find average values (excluding "dead zone")
    avg_values = []
    for image in images:
        avg_values.append(np.mean(np.sort(image.reshape(-1))[::-1][1:20]))

    # Theory value
    theory = 1000 / (pi * 10**2)

    # Plot contour of each slice
    for j in range(len(images)):
        plt.plot(images[j][720, 925:1000], '-', label=f'Smoothness = {smoothnesses[j]}')
    x1, y1 = [0, 75], [theory, theory]
    plt.plot(x1, y1, '-.', label='Fully diffuse theoretical value')
    # plt.plot(np.repeat(theory))
    plt.title('Slice for central row (pixel 720)')
    plt.xlabel('Pixel Index [x]')
    plt.ylabel('Luminance (cd)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    angles = [0, 5, 15, 25, 35, 45, 55, 65, 75, 85]
    distance = 10.0
    # calc_spherical_cam_loc(angles, distance)
    analyse_headlight_ies()
    analyse_headlight_exrs_theta(angles)
    # plot_surface_test_data()
    # smoothnesses = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
    # smoothnesses = [100, 98, 96, 94, 92, 90]
    # plot_surface_contours(smoothnesses)

    # angles = [0, 5, 15, 25, 35, 45, 55, 65, 75, 85]
    # plot_surface_test_data_angles(angles)
    print('All done!')
