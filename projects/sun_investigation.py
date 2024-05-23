import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from math import pi
from osgeo import gdal
from typing import Optional, List
from scipy.optimize import curve_fit
from core.fog_functions import calc_fog_func

from utils.file_management import list_simulations

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})
images = 50


def sin_func(x, a, b, c, d):
    # sin function
    return a * np.sin(b * (x + c)) + d


def gauss_func(x, a, x0, sigma):
    # Gaussian function
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def calc_zenith_function(array_len: int, ydata: np.array([float])) -> np.array([float]):
    # A function for calculating the sun's luminance for a given time of day. Input and output should follow
    # minutes from zenith convention, i.e., x=0 is zenith, x=-30 is 30 minutes before zenith, x=30 is 30 minutes after
    # zenith, etc.

    # Init necessary variables
    adj_len = (array_len - 1) / 2

    # Convert minutes from zenith to angle
    latitude = 51.48 * np.pi / 180
    dec_angle = -23.45 * np.cos(360/365 * (172 + 10)) * np.pi / 180     # Angle of declination
    hour_angle = np.linspace(-adj_len * 7.5, adj_len * 7.5, array_len) * np.pi / 180    # Local hour angle

    # Solar elevation
    # https://www.omnicalculator.com/physics/sun-angle
    elev_angle = np.arcsin(
                (np.sin(dec_angle) * np.sin(latitude)) +
                (np.cos(latitude) * np.cos(hour_angle) * np.cos(dec_angle)))

    # Convert solar elevation to luminosity scale
    # https://www.pveducation.org/pvcdrom/properties-of-sunlight/calculation-of-solar-insolation
    i_d = 1.353 * 0.7 ** ((1 / np.sin(elev_angle)) ** 0.678)
    i_d_norm = i_d - max(i_d) + max(ydata)

    return i_d_norm


def plot_rgb_data(data, colour: str, label: str):
    plt.plot(data, '.', color=colour, label=label, markersize=1)
    plt.title("HDR image readout")
    plt.xlabel("Pixel #")
    plt.ylabel("Intensity (knits)")
    # Fix duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def create_circular_mask(h, w, center, radius):
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


# Init and loads
BASE_PATH = Path(r'C:\Users\ElliotLondon\Documents\PythonLocal\rFproControlPy\data\sun')
DATA_PATH = BASE_PATH / 'sweep_fog_hg10_max'


def analyse_luminosity(folder_time: Optional[str] = None) -> float:
    """
    Analyse the luminosity of a single image within a specified folder
    :param folder_time: The name of the folder in str(yyyymmdd_hhmmss) format. If not specified, default is used
    :return: None
    """
    # Load data path according to whether there is a folder time passed in
    if folder_time is not None:
        current_data_path = DATA_PATH / folder_time
    else:
        current_data_path = DATA_PATH / '20240501_161650'

    # Load .tiff image using GDAL
    data_set = gdal.Open(str(current_data_path / f'TrainingTruthHDR_0004.tiff'))

    # As, there are 3 bands, we will store in 3 different variables
    band_1 = data_set.GetRasterBand(1)  # red channel
    band_2 = data_set.GetRasterBand(2)  # green channel
    band_3 = data_set.GetRasterBand(3)  # blue channel
    b1 = band_1.ReadAsArray()
    b2 = band_2.ReadAsArray()
    b3 = band_3.ReadAsArray()

    # # Show initial plots
    # plot_rgb_data(b1, 'r', 'Channel: R')
    # plot_rgb_data(b2, 'g', 'Channel: G')
    # plot_rgb_data(b3, 'b', 'Channel: B')
    # plt.show()

    # Find max ind, use to create and apply a circular mask
    max_value = 0
    temp_ind = 0
    for i in range(len(b1[:, 0])):
        for j in range(len(b1[0, :])):
            if b1[i, j] > max_value:
                max_value = b1[i, j]
                temp_ind = [j, i]
    sun_mask = create_circular_mask(len(b1[:, 0]), len(b1[0, :]), temp_ind, 50)
    new_b1 = b1.copy()
    new_b2 = b2.copy()
    new_b3 = b3.copy()
    new_b1[~sun_mask] = 0
    new_b2[~sun_mask] = 0
    new_b3[~sun_mask] = 0
    # Remove the min value (noise) from the remaining data
    new_b1 -= min(new_b1[new_b1 > 0])
    new_b2 -= min(new_b2[new_b2 > 0])
    new_b3 -= min(new_b3[new_b3 > 0])
    # Anything below 0 is now noise, set it to 0
    new_b1[new_b1 < 0] = 0
    new_b2[new_b2 < 0] = 0
    new_b3[new_b3 < 0] = 0

    # # Show cleaned plots
    # plot_rgb_data(new_b1, 'r', 'Channel: R')
    # plot_rgb_data(new_b2, 'g', 'Channel: G')
    # plot_rgb_data(new_b3, 'b', 'Channel: B')
    # plt.show()

    # Units are in knits (kcd / m^2)
    tot_max = sum(sum(new_b1) + sum(new_b2) + sum(new_b3))

    # Calulate solar luminance, or candela / m^2
    tot_lums = tot_max * 1000

    # Calculate solar illuminance, or lux / m^2
    total_illums = (0.21 * sum(sum(new_b1)) + 0.72 * sum(sum(new_b2)) + 0.07 * sum(sum(new_b3))) * pi * 1000

    print(f"Simulated max solar luminance (nits): {max_value * 1000:.2e}")
    print(f"Simulated total solar luminance (nits): {tot_lums:.2e}")
    print(f"Simulated solar illuminance (lux): {total_illums:.2e}\n")

    return max_value * 1000


def analyse_turbidity_sequence(folder_times: List[str] = None) -> None:
    """
    Analyse the luminosity of a single image within an array of specified folders. Plots according to time.
    :param folder_times: List of folders in str(yyyymmdd_hhmmss) format.
    :return: None
    """
    ydata = []
    for folder in folder_times:
        ydata.append(analyse_luminosity(folder) / 1e9)

    # Number of multiples of 30 minutes from zenith at zero for 1st value in array
    xdata = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.5, 15, 20, 30]

    # Print ratio between max and min luminance difference
    print(f'Max/min luminance: {ydata[-1]/ydata[0]}')

    # Plot the data
    plt.plot(xdata, ydata, '.', label='Measured data')
    plt.title('Solar luminosity at zenith')
    plt.xlabel('Turbidity (SilverLining units)')
    plt.ylabel('Solar luminosity (candela / m^2) * 1e9')
    plt.legend()
    plt.show()


def analyse_fog_sequence(folder_times: List[str] = None):
    """
    Analyse the luminosity of a single image within an array of specified folders. Plots according to time.
    :param folder_times: List of folders in str(yyyymmdd_hhmmss) format.
    :return: None
    """
    ydata = []
    for folder in folder_times:
        ydata.append(analyse_luminosity(folder) / 1e9)

    xdata = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
    xinterp = np.linspace(xdata[0], xdata[-1], 250)

    # Calculate curve fit of fog luminance function
    p1, cov = curve_fit(calc_fog_func, xdata, ydata, p0=[ydata[0], 10e-6], method='dogbox')
    print(p1)

    fog_y = []
    for i in range(len(xinterp)):
        fog_y.append(calc_fog_func(xinterp[i], ydata[0], p1[1]))
    fog_y = np.array(fog_y)

    # Print ratio between max and min luminance difference
    print(f'Max/min luminance: {ydata[-1]/ydata[0]}')

    # # Plot the data
    # plt.plot(xdata, ydata, '.', label='Measured data')
    # plt.plot(xinterp, fog_y, label=r'$y=I_0\exp^{-2\pi r^2 nkZ}$')
    # plt.title('Solar luminosity at zenith')
    # plt.xlabel(r'$Fog\;density\;(Max.\;Extinction\;Coeff.,\;[1/m])$')
    # plt.ylabel(r'$Solar\;luminosity\;[cd / m^2] * 1e9$')
    # plt.legend()
    # plt.show()

    return xdata, ydata, xinterp, fog_y


def analyse_luminosity_sequence(folder_times: List[str] = None) -> None:
    """
    Analyse the luminosity of a single image within an array of specified folders. Plots according to time.
    :param folder_times: List of folders in str(yyyymmdd_hhmmss) format.
    :return: None
    """
    ydata = []
    for folder in folder_times:
        ydata.append(analyse_luminosity(folder) / 1e9)

    # Number of multiples of 30 minutes from zenith at zero for 1st value in array
    array_len = len(ydata)
    start_time = 30 * (array_len - 1) / 2
    xdata = np.linspace(-start_time, start_time, array_len)
    xinterp = np.linspace(-start_time, start_time, 250)
    yinterp = np.interp(xinterp, xdata, ydata)

    # Get luminance from ydata
    i_d_norm = calc_zenith_function(array_len, ydata)

    # Fit the data
    # p1, cov = curve_fit(sin_func, xinterp, yinterp, p0=[1.55 / 2, 1.5e-3, 0, 1.55 / 2], method='dogbox')
    # print(p1)

    # Plot the data
    plt.plot(xdata, ydata, '.', label='Measured data')
    # plt.plot(xinterp, yinterp, '.', label='Extrapolation')
    plt.plot(xdata, i_d_norm, '-', label='Solar elevation (normalised)')
    # plt.plot(xdata, sin_func(xdata, p1[0], p1[1], p1[2], p1[3]), label='y=a sin(bx + c) + d')
    plt.title('Solar luminosity at zenith')
    plt.xlabel('Minutes from zenith')
    plt.ylabel('Solar luminosity (candela / m^2) * 1e9')
    plt.legend()
    plt.show()


def analyse_luminosity_month(folder_times: List[str] = None) -> None:
    """
    Analyse the luminosity of a single image within an array of specified folders. Plots according to time.
    :param folder_times: List of folders in str(yyyymmdd_hhmmss) format.
    :return: None
    """
    ydata = []
    for folder in folder_times:
        ydata.append(analyse_luminosity(folder) / 1e9)
    xdata = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # # Interpolate the data for fitting
    # xinterp = np.linspace(0, 100, 1000)
    # yinterp = np.interp(xinterp, xdata, ydata)
    #
    # # Fit the data
    # params, cov = curve_fit(sin_func, xinterp, yinterp, p0=[5e8, 1e-2, 5e8, 5e8], method='dogbox')
    # print(params)

    # Plot the data
    plt.plot(xdata, ydata, '.', label='Measured data')
    # plt.plot(xdata, sin_func(xdata, params[0], params[1], params[2], params[3]), label='y=a sin(bx + c) + d')
    plt.title('Solar luminosity at zenith, 21st of each calendar month')
    plt.xlabel('Month')
    plt.ylabel('Solar luminosity (candela / m^2) * 1e9')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Define the sequence of folders to grab data from
    fnames = ['sweep_fog_hg00_max', 'sweep_fog_hg10_max', 'sweep_fog_hg25_max']
    xdata = []
    ydata = []
    xinterp = []
    fog_y = []
    for fname in fnames:
        DATA_PATH = BASE_PATH / fname
        list_simulations(fname)

        with open(DATA_PATH / 'names.txt') as file:
            folders = eval(file.read())

        if 'month' in fname:
            analyse_luminosity_month(folders)
        elif 'turbidity' in fname:
            analyse_turbidity_sequence(folders)
        elif 'fog' in fname and 'hg' in fname:
            x1, y1, x2, y2 = analyse_fog_sequence(folders)
            xdata.append(x1)
            ydata.append(y1)
            xinterp.append(x2)
            fog_y.append(y2)
        else:
            analyse_luminosity_sequence(folders)

    # Plot the data
    labels = [
        'g=0.00',
        'g=0.10',
        'g=0.25'
    ]
    for i in range(len(fnames)):
        plt.plot(xdata[i], ydata[i], '.', label=f'Measured data, {labels[i]}')
        plt.plot(xinterp[i], fog_y[i], label=r'$y=I_0\exp^{-2\pi r^2 nkZ}$')
    # plt.yscale('log')
    plt.title('Solar luminosity at zenith')
    plt.xlabel(r'$Fog\;density\;(Max.\;Extinction\;Coeff.,\;[1/m])$')
    plt.ylabel(r'$Solar\;luminosity\;[cd / m^2] * 1e9$')
    plt.legend()
    plt.show()
