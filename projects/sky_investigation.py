import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from math import pi
from osgeo import gdal
from typing import Optional, List

from utils.file_management import list_simulations

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})
images = 50


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
DATA_PATH = Path(r'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/data/sky')


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
    data_set = gdal.Open(str(current_data_path / f'TrainingTruthHDR_0008.tiff'))

    # As, there are 3 bands, we will store in 3 different variables
    band_1 = data_set.GetRasterBand(1)  # red channel
    band_2 = data_set.GetRasterBand(2)  # green channel
    band_3 = data_set.GetRasterBand(3)  # blue channel
    b1 = band_1.ReadAsArray()
    b2 = band_2.ReadAsArray()
    b3 = band_3.ReadAsArray()

    # Show initial plots
    plot_rgb_data(b1, 'r', 'Channel: R')
    plot_rgb_data(b2, 'g', 'Channel: G')
    plot_rgb_data(b3, 'b', 'Channel: B')
    plt.show()

    # Find max value
    max_value = 0
    for i in range(len(b1[:, 0])):
        for j in range(len(b1[0, :])):
            if b1[i, j] > max_value:
                max_value = b1[i, j]

    sky_mask = create_circular_mask(len(b1[:, 0]), len(b1[0, :]), [len(b1[:, 0]) / 2, len(b1[0, :]) / 2], 30)
    new_b1 = b1.copy()
    new_b2 = b2.copy()
    new_b3 = b3.copy()
    new_b1[~sky_mask] = 0
    new_b2[~sky_mask] = 0
    new_b3[~sky_mask] = 0

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

    return tot_lums


def analyse_luminosity_sequence(folder_times: List[str] = None) -> None:
    """
    Analyse the luminosity of a single image within an array of specified folders. Plots according to time.
    :param folder_times: List of folders in str(yyyymmdd_hhmmss) format.
    :return: None
    """
    ydata = []
    for folder in folder_times:
        ydata.append(analyse_luminosity(folder))
    xdata = np.linspace(0, 100, 20)

    # Interpolate the data for fitting
    xinterp = np.linspace(0, 100, 1000)
    # yinterp = np.interp(xinterp, xdata, ydata)

    # Fit the data
    # params, cov = curve_fit(sin_func, xinterp, yinterp, p0=[5e8, 1e-2, 5e8, 5e8], method='dogbox')
    # print(params)

    # Plot the data
    plt.plot(xdata, ydata, '.', label='Measured data')
    # plt.plot(xdata, sin_func(xdata, params[0], params[1], params[2], params[3]), label='y=a sin(bx + c) + d')
    plt.title('Solar luminosity at zenith, 21st June 2008')
    plt.xlabel('Turbidity')
    plt.ylabel('Solar luminosity (candela / m^2)')
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
        ydata.append(analyse_luminosity(folder))
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
    plt.ylabel('Solar luminosity (candela / m^2)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Define a folder to investigate, else, define the sequence of folders to grab data from
    folder = None
    folder = str(DATA_PATH / '20240509_141940')
    if folder is not None:
        analyse_luminosity(folder)
    else:
        fname = 'timelapse_clear_sum'
        list_simulations(fname)

        with open(DATA_PATH / 'names.txt') as file:
            folders = eval(file.read())

        if 'month' in fname:
            analyse_luminosity_month(folders)
        else:
            analyse_luminosity_sequence(folders)
