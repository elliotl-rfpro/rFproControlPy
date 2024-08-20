from datetime import datetime, timedelta
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
from scipy.ndimage import zoom

import core.global_variables
from core.fog_functions import calc_fog_func
from utils.math_utils import visibility_func, cos_func

from utils.math_utils import calc_spherical_coord
from utils.file_management import list_simulations
from core.global_variables import BASE_PATH, DATA_PATH, img_num, img_type, save
from core.processing import get_tiff_image, get_ldr_image, get_exr_image

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})


def analyse_headlight_exrs_x():
    # Analyse a sequence of images where a camera is a fixed X distance away from a headlight, moving along the Y axis.
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/headlights/'
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
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/headlights/'
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
    plt.title('Headlight/IES max luminance vs angle at fixed 10m from sensor')
    plt.ylabel('Luminance (cd/m^2)')
    plt.xlabel('Angle between light and sensor (degrees)')
    plt.legend()
    plt.show()

    print('Ok')


def analyse_headlight_ies():
    # Load in a .exr image from a converted IES profile and process it for comparison with measured data.
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/headlights/'
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
    img_plot = img_c * 1000 / (np.pi * 10 ** 2)
    plt.title('Headlight IES profile lineouts')
    plt.plot(np.arange(-90, 90), img_plot[810:-810, 1765][::-1], label='IES profile 1765')
    plt.plot(np.arange(-90, 90), img_plot[810:-810, 1775][::-1], label='IES profile 1775')
    plt.plot(np.arange(-90, 90), img_plot[810:-810, 1785][::-1], label='IES profile 1785')
    plt.plot(np.arange(-90, 90), img_plot[810:-810, 1795][::-1], label='IES profile 1795')
    plt.plot(np.arange(-90, 90), img_plot[810:-810, 1805][::-1], label='IES profile 1805')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()
    # plt.show()

    print(f'Max in img_c_: {np.max(img_c_)}')


def analyse_headlight_wall_test():
    # Load in a .exr image from a converted IES profile and process it for comparison with measured data.
    title = 'wall_test_single_2d5m.exr'
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/headlights/'
    img_plot = get_exr_image([data_path + title])[0]

    # Find row where the data is maximal
    measured_g = img_plot[:, :, 1]
    max_val = np.max(measured_g)
    max_index = np.where(measured_g == max_val)[0][0]

    # Adjust so that max value lies in the center (to align both plots)
    measured_g = np.roll(measured_g, (measured_g.shape[1] // 2) - np.argmax(measured_g[max_index]))

    # Plot
    plt.plot(measured_g[max_index][::-1], label='Measurement max')

    # Plot lineout from .IES profile
    # Load in a .exr image from a converted IES profile and process it for comparison with measured data.
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/headlights/'
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
    img_c = img_c[::-1]

    # Interp img_c so that it has the same dimensions of img_plot
    zoom_x = img_plot[:, :, 1].shape[0] / img_c.shape[0]
    zoom_y = img_plot[:, :, 1].shape[1] / img_c.shape[1]
    img_c_interp = zoom(img_c, (zoom_x, zoom_y))

    # Rescale value
    img_c_interp = img_c_interp / (np.pi * 10 ** 2) / zoom_y

    # Plot values around max
    max_val = np.max(img_c_interp)
    max_index = np.where(img_c_interp == max_val)[0][0]

    # Adjust so that peak is in center
    ies_g = np.roll(img_c_interp, (img_c_interp.shape[1] // 2) - np.argmax(img_c_interp[max_index]))

    # Plot it
    plt.plot(ies_g[max_index], label='IES profile max')

    plt.legend()
    if '2d5' in title:
        title = '2.5m'
    plt.title(f'Single headlight luminance test: fully diffuse dielectric wall {title} away')
    plt.xlim([500, 1500])
    plt.xlabel('Pixel index')
    plt.ylabel('Normalised luminance (cd/m^2)')
    plt.show()

    print(f'Max in img_c_: {np.max(img_plot)}')


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
    xlss = pd.read_excel('C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/surface_tests/reflective_surface_tests.xlsx',
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


def plot_surface_test_data_fog(smoothness, fog_densities):
    # Function for plotting a sequence of surfaces at a fixed distance from a camera

    # Load the exrs to find a fixed pixel to evaluate
    i = 1
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/surface_tests/'
    images = []
    for d in fog_densities:
        image = get_exr_image([data_path + f'smooth{smoothness:03d}_l1_10m_fog{int(d * 1000):04d}_theta45.exr'])[0]
        images.append(image[:, :, 1])

    # Find max luminance value
    max_values = []
    averages = []
    for image in images:
        max_values.append(np.amax(image))
        averages.append(image[image >= 0.5].mean())

    # Plot contour of each slice
    plt.plot(fog_densities, max_values, label=f'Max luminance')
    plt.plot(fog_densities, averages, label=f'Average luminance')
    # plt.fill_between([0.05, 0.1], np.amax(max_values), color='r', alpha=0.1)
    # plt.text(0.075, np.amax(max_values) / 2, 'Noise-dominant\n region', horizontalalignment='center')
    plt.title(f'Luminance statistics. Smoothness :{smoothness}%')
    plt.xlabel('Fog density (arb. units)')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()
    plt.show()

    # Plot contour of each slice
    for j in range(len(images)):
        plt.plot(images[j][719, 925:1000], label=f'Fog density = {fog_densities[j]}')
    plt.title(f'Slide for central row (pixel 720). Smoothness: {smoothness}%')
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


def analyse_overcast_exrs(cloud_densities):
    # Function for analysing the luminance of .exr files produced during a sweep of overcast cloud conditions
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/2kflat/sky/clouds/overcast_sweep/'
    images = []
    for d in cloud_densities:
        image = get_exr_image([data_path + f'overcast_sweep_{int(d * 100):03d}.exr'])[0]
        images.append(image[:, :, 1])

    # Find max luminance value
    max_values = []
    sky_values = []
    sample_1 = []
    sample_2 = []
    sky_averages = []
    for image in images:
        max_values.append(np.amax(image))
        sky_values.append(image[int(image.shape[0] / 2), int(image.shape[1] / 2)])
        sample_1.append(image[int(image.shape[0] / 2) - 200, int(image.shape[1] / 2) - 200])
        sample_2.append(image[int(image.shape[0] / 2) + 200, int(image.shape[1] / 2) + 200])
        sky_averages.append(image[image <= 1e5].mean())

    # Plot curve fit according to Beer-Lambert law
    params, cov = curve_fit(visibility_func, cloud_densities, max_values)
    print(params)

    # Plot contour of each slice
    # plt.plot(cloud_densities, max_values, '.', label=f'Max luminance (sun)', markersize=8)
    # plt.plot(cloud_densities, visibility_func(cloud_densities, *params), label=r'$I_0e^{-kx}$')
    plt.plot(cloud_densities, sky_values, label=f'Fixed luminance (sky, center)')
    plt.plot(cloud_densities, sample_1, label=f'Fixed luminance (sky, sample 1)')
    plt.plot(cloud_densities, sample_2, label=f'Fixed luminance (sky, sample 2)')
    plt.plot(cloud_densities, sky_averages, label=f'Average luminance (sky)')
    plt.title('Luminance vs density for overcast cloud layer (sky region)')
    plt.xlabel('Cloud density (arb. units)')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()
    plt.show()


def analyse_overcast_xlsx():
    # Function for analysing the luminance of theoretical overcast sky luminance values from a .xlsx file
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/sky/'
    xlsx = pd.read_excel(data_path + 'overcast_luminances.xlsx', sheet_name='Sun')

    # Get data
    cloud_perc = np.arange(0, 110, 10)
    data_1 = xlsx['Norm. Luminance'][0:11].values
    data_2 = xlsx['Norm. Luminance'][13:24].values
    data_3 = xlsx['Norm. Luminance'][26:37].values
    data_4 = xlsx['Norm. Luminance'][39:].values

    # Curvefit sin wave to max/min points
    max_values = np.zeros(11)
    min_values = np.zeros(11)
    for i in range(len(data_1)):
        max_values[i] = max(data_1[i], data_2[i], data_3[i], data_4[i])
        min_values[i] = min(data_1[i], data_2[i], data_3[i], data_4[i])
    avg = (max_values + min_values) / 2
    diff = max_values - min_values
    params, cov = curve_fit(cos_func, cloud_perc, avg, p0=[1/2, pi/100, pi/2, 0.5], bounds=((0.475, 0, 0, 0.475), (0.5, 1, 100, 0.5)))
    print(params)

    # Plot contour of each slice
    plt.errorbar(cloud_perc, avg, yerr=diff/2, label='Average luminance over 12 months', marker='.', markersize=8,
                 linestyle='none', linewidth=1, capsize=5)
    # plt.plot(cloud_perc, max_values, '.', label=f'Max luminance', markersize=8)
    # plt.plot(cloud_perc, avg, '.', label=f'Average luminance', markersize=8)
    # plt.plot(cloud_perc, min_values, '.', label=f'Min luminance', markersize=8)
    plt.plot(cloud_perc, cos_func(cloud_perc, *params), label=r'$A\cos(Bx + C) + D$')
    plt.title('Solar luminance vs cloud coverage from CIE data')
    plt.xlabel('Cloud coverage, %')
    plt.ylabel('Normalised luminance (cd/m^2)')
    plt.legend()
    plt.show()


def analyse_sky_xlsx():
    # Function for analysing the luminance of a clear sky throughout an entire day using measurement .xlsx
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/sky/'
    xlsx = pd.read_excel(data_path + 'overcast_luminances.xlsx', sheet_name='Sky')

    # Get CIE data
    times = xlsx['Time from zenith (170, 50)'].values
    data_1 = xlsx['Sky Luminance'].values / np.max(xlsx['Sky Luminance'].values)
    data_2 = xlsx['Sky Luminance.1'].values / np.max(xlsx['Sky Luminance.1'].values)

    # Get sim data and adjust to match CIE range
    sim_data = np.load('C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/sky/clear_sky_luminance.npy')
    sim_data /= np.max(sim_data)
    # sd_mid = len(sim_data) // 2
    # sim_data = sim_data[sd_mid - 13:sd_mid + 16]

    # Plot contour of each slice
    plt.plot(times, data_1, label='Sky at location (170, 50)')
    plt.plot(times, data_2, label='Sky at location (190, 50)')
    plt.plot(np.arange(-9.5, 8.5, 0.5), sim_data[4:-8], label='Simulation data')
    plt.title('Sky luminance vs time of day from CIE data (fully clear day)')
    plt.xlabel('Hours from solar zenith')
    plt.ylabel('Normalised luminance (cd/m^2)')
    plt.legend()
    plt.show()


def analyse_visibility_xlsx():
    # Function to import a visibility spreadsheet, plot and analyse the data.
    data_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/simperftests/visibility_tests/'
    vis_xlss = pd.read_excel(data_path + 'visibility.xlsx')

    # Extract data
    dist = vis_xlss['Distance'][1:-1]
    values = vis_xlss['Threshold'][1:-1]
    uncertainties = np.abs(vis_xlss['Visible'] - vis_xlss['Invisible'])[1:-1]

    # Curvefit visibility func
    params, cov = curve_fit(visibility_func, dist, values, p0=[1.5e-2, 2, 1, 0], sigma=uncertainties)
    print(params)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(dist, values, yerr=uncertainties, label="Measured visibility threshold", marker=".", markersize=8,
                linestyle="none", linewidth=1, capsize=5)
    # plt.plot(dist, values, '.', label=f'Measured visibility threshold')
    plt.plot(dist, visibility_func(dist, *params), '-', label=r'$I_0 e^{(-kx)}$')
    plt.title('Approximate visibility threshold for 1kc point light within volumetric fog')
    plt.ylabel('Fog density (arb. units)')
    plt.xlabel('Distance between camera and light (m)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    angles = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # angles = [0, 5, 15, 25, 35, 45, 55, 65, 75, 85]
    distance = 10.0
    # calc_spherical_cam_loc(angles, distance)

    # Headlight testing
    # analyse_headlight_ies()
    # analyse_headlight_exrs_theta(angles)
    # analyse_headlight_wall_test()

    # Surface testing
    # plot_surface_test_data()
    # smoothnesses = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
    # smoothnesses = [100, 98, 96, 94, 92, 90]
    # plot_surface_contours(smoothnesses)

    # Angled surface testing
    # angles = [0, 5, 15, 25, 35, 45, 55, 65, 75, 85]
    # plot_surface_test_data_angles(angles)

    # Fog vs surface/backscattering tests
    # smoothnesses = [0, 100]
    # fog_densities = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    # for s in smoothnesses:
    #     plot_surface_test_data_fog(s, fog_densities)

    # # Cloud test: measured overcast
    # cloud_densities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # analyse_overcast_exrs(cloud_densities)

    # Cloud test: theoretical overcast
    # analyse_overcast_xlsx()
    analyse_sky_xlsx()

    # Visibility tests
    # analyse_visibility_xlsx()

    print('All done!')
