import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from osgeo import gdal
from typing import Optional, List
from scipy.optimize import curve_fit
from core.fog_functions import calc_fog_func

from utils.file_management import list_simulations

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

# Init, loads, globals...
BASE_PATH = Path(r'C:\Users\ElliotLondon\Documents\PythonLocal\rFproControlPy\data\sun\clouds')
# BASE_PATH = Path(r'C:\Users\ElliotLondon\Documents\PythonLocal\rFproControlPy\data\sky\clouds')
DATA_PATH = BASE_PATH / 'clouds_raster_1414_g-085'
img_num = '0004'
# img_type = 'LDR'
img_type = 'HDR'
save = True


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
    dec_angle = -23.45 * np.cos(360 / 365 * (172 + 10)) * np.pi / 180  # Angle of declination
    hour_angle = np.linspace(-adj_len * 7.5, adj_len * 7.5, array_len) * np.pi / 180  # Local hour angle

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


def get_tiff_image(folders):
    # Load data path according to whether there is a folder time passed in
    images = []
    for folder in folders:
        current_data_path = DATA_PATH / folder
        data_set = gdal.Open(str(current_data_path / f'TrainingTruthHDR_{img_num}.tiff'))

        # As, there are 3 bands, we will store in 3 different variables
        band_1 = data_set.GetRasterBand(1)  # red channel
        band_2 = data_set.GetRasterBand(2)  # green channel
        band_3 = data_set.GetRasterBand(3)  # blue channel
        b1 = band_1.ReadAsArray()
        b2 = band_2.ReadAsArray()
        b3 = band_3.ReadAsArray()

        images.append(np.dstack((b1, b2, b3)))

    return images


def get_ldr_image(folders):
    images = []
    for folder in folders:
        current_data_path = DATA_PATH / folder
        fpath = (str(current_data_path / f'TrainingTruthLDR_{img_num}.bmp'))
        images.append(plt.imread(fpath))

    return images


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
    data_set = gdal.Open(str(current_data_path / f'TrainingTruthHDR_{img_num}.tiff'))

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

    if 'sky' in str(DATA_PATH):
        # Just use the central value
        max_value = b2[b2.shape[1] // 2, b2.shape[0] // 2 - 50]
    else:
        # Find and use max value
        max_value = 0
        for i in range(len(b1[:, 0])):
            for j in range(len(b1[0, :])):
                if b1[i, j] > max_value:
                    max_value = b1[i, j]

    # # Show cleaned plots
    # plot_rgb_data(new_b1, 'r', 'Channel: R')
    # plot_rgb_data(new_b2, 'g', 'Channel: G')
    # plot_rgb_data(new_b3, 'b', 'Channel: B')
    # plt.show()

    # # Units are in knits (kcd / m^2)
    # tot_max = sum(sum(new_b1) + sum(new_b2) + sum(new_b3))

    # knits to nits (cd/m^2)
    max_value *= 1000
    print(f"Simulated max solar luminance (nits): {max_value:.2e}")

    return max_value


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
    print(f'Max/min luminance: {ydata[-1] / ydata[0]}')

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
        ydata.append(analyse_luminosity(folder))

    # xdata = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
    xdata = [0.0001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    xinterp = np.linspace(xdata[0], xdata[-1], 250)

    # Calculate curve fit of fog luminance function
    try:
        p1, cov = curve_fit(calc_fog_func, xdata, ydata, p0=[ydata[0], 1], method='dogbox')
    except RuntimeError:
        p1 = [np.nan, np.nan]
    print(p1)

    # p1 = [0, 0]
    fog_y = []
    for i in range(len(xinterp)):
        fog_y.append(calc_fog_func(xinterp[i], ydata[0], p1[1]))
    fog_y = np.array(fog_y)

    # Print ratio between max and min luminance difference
    print(f'Max/min luminance: {ydata[-1] / ydata[0]}')

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
        ydata.append(analyse_luminosity(folder))

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
    # plt.plot(xdata, i_d_norm, '-', label='Solar elevation (normalised)')
    # plt.plot(xdata, sin_func(xdata, p1[0], p1[1], p1[2], p1[3]), label='y=a sin(bx + c) + d')
    plt.title('Sky luminosity perpendicular to earth surface')
    plt.xlabel('Minutes from zenith')
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
    # Fog HG anisotropy and albedo tests
    # fnames = ['sweep_fog_hg00_alb07_max', 'sweep_fog_hg50_alb07_max', 'sweep_fog_hg100_alb07_max']
    # fnames = ['sweep_fog_hg00_alb07_max', 'sweep_fog_hg10_alb07_max', 'sweep_fog_hg20_alb07_max']
    # fnames = ['sweep_fog_hg00_alb1_max', 'sweep_fog_hg10_alb1_max', 'sweep_fog_hg20_alb1_max']
    # fnames = ['timelapse_clear_full']

    # Fog image comparisons
    if 'sky' in str(BASE_PATH):
        fnames = ['fog_raster_0714_hg000_alb07', 'fog_raster_1414_hg000_alb07', 'fog_raster_2114_hg000_alb07',
                  'fog_raytrace_0714_hg000_alb07', 'fog_raytrace_1414_hg000_alb07', 'fog_raytrace_2114_hg000_alb07']
    elif 'sun' in str(BASE_PATH):
        fnames = ['fog_raster_1414_hg000_alb07', 'fog_raytrace_1414_hg000_alb07']
    xdata = []
    ydata = []
    xinterp = []
    fog_y = []
    images = []
    for fname in fnames:
        DATA_PATH = BASE_PATH / fname
        list_simulations(DATA_PATH)

        with open(DATA_PATH / 'names.txt') as file:
            folders = eval(file.read())

        if 'month' in fname:
            analyse_luminosity_month(folders)
        elif 'turbidity' in fname:
            analyse_turbidity_sequence(folders)
        elif 'fog' in fname and 'hg' in fname:
            # Data
            x1, y1, x2, y2 = analyse_fog_sequence(folders)
            xdata.append(x1)
            ydata.append(y1)
            xinterp.append(x2)
            fog_y.append(y2)

            # Images
            if img_type == 'LDR':
                images.append(get_ldr_image(folders))
            elif img_type == 'HDR':
                images.append(get_tiff_image(folders))
        else:
            analyse_luminosity_sequence(folders)

    if len(fnames) >= 6:
        sp_int = 230
    elif len(fnames) == 2:
        sp_int = 210

    # Plot luminance analysis data
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.subplots_adjust(hspace=0.5)
    for k in range(len(fog_y)):
        ax = fig.add_subplot(sp_int + k + 1)
        ax.plot(xinterp[k], fog_y[k])
        ax.plot(xdata[k], ydata[k], '.')
        ax.set_title(fnames[k])
        plt.ylim([0, max(max(ydata)) + round(max(max(ydata)) / 10)])
    fig.text(0.5, 0.04, r'$Fog\;density\;(Max.\;Extinction\;Coeff.,\;[1/m])$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$Solar\;luminosity\;[cd / m^2] * 1e9$', ha='center', va='center', rotation='vertical')
    if save:
        if 'sun' in str(BASE_PATH):
            sname = f'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/analysis_sun.png'
        elif 'sky' in str(BASE_PATH):
            sname = f'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/analysis_sky.png'
        plt.savefig(sname, bbox_inches="tight")

    # Plot timelapse of LDR/HDR images
    i = 0
    densities = [0.0001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    for image in images:
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(fnames[i])
        for k in range(len(image)):
            ax = fig.add_subplot(240 + k + 1)
            ax.grid(False)
            ax.imshow(image[k])
            if 'sky' in str(BASE_PATH):
                ax.plot(image[k].shape[1] // 2, image[k].shape[0] // 2 - 50, 'o', markersize=2)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f'Density={densities[k]}')
        if save:
            sname = f'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/timelapse_{fnames[i]}_{img_type}.png'
            plt.savefig(sname, bbox_inches="tight")
        i += 1
    plt.show()

    # # Plot fog anisotropy analysis data
    # labels = []
    # for element in fnames:
    #     if 'hg00' in element:
    #         labels.append('g=0.00')
    #     elif 'hg10' in element and 'hg100' not in element:
    #         labels.append('g=0.10')
    #     elif 'hg20' in element:
    #         labels.append('g=0.20')
    #     elif 'hg50' in element:
    #         labels.append('g=0.50')
    #     elif 'hg100' in element:
    #         labels.append('g=1.0')
    # for k in range(len(fnames)):
    #     plt.plot(xinterp[k], fog_y[k], label=r'$y=I_0\exp^{-2\pi r^2 nkZ}$')
    #     plt.plot(xdata[k], ydata[k], '.', label=f'Measured data, {labels[k]}')
    # # plt.yscale('log')
    # plt.title('Solar luminosity at zenith, albedo = 0.7')
    # plt.xlabel(r'$Fog\;density\;(Max.\;Extinction\;Coeff.,\;[1/m])$')
    # plt.ylabel(r'$Solar\;luminosity\;[cd / m^2] * 1e9$')
    # plt.legend()
    # plt.show()
