import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from osgeo import gdal
from typing import Optional, List
from scipy.optimize import curve_fit

import core.global_variables
from core.fog_functions import calc_fog_func

from core.lighting_functions import calc_zenith_function
from utils.file_management import list_simulations
from utils.math_utils import calc_theoretical_luminance
from core.global_variables import BASE_PATH, DATA_PATH, img_num, img_type, save
from core.processing import get_tiff_image, get_ldr_image

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})


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

    if 'sky' in str(DATA_PATH):
        # Just use the central value
        # max_value = b2[b2.shape[1] // 2, b2.shape[0] // 2]
        max_value = b2[200, 500]
    else:
        # Find and use max value
        max_value = 0
        for i in range(len(b1[:, 0])):
            for j in range(len(b1[0, :])):
                if b1[i, j] > max_value:
                    max_value = b1[i, j]

    # plt.imshow(b2)
    # plt.show()

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
    # xdata = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1]
    xinterp = np.linspace(xdata[0], xdata[-1], 250)

    # Calculate curve fit of fog luminance function
    try:
        p1, cov = curve_fit(calc_fog_func, xdata, ydata, p0=[ydata[0], 1], method='dogbox')
    except RuntimeError:
        p1 = [np.nan, np.nan]
    except ValueError:
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

    # Save sky data
    np.save('C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/sky/clear_sky_luminance.npy', ydata)

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


def plot_luminance_data(fnames, save: bool = False):
    # Plot luminance analysis data
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.subplots_adjust(hspace=0.5)

    # Work out what the correct subplot indices are
    if len(fnames) >= 6:
        sp_int = 230
    elif len(fnames) == 2:
        sp_int = 210

    # Plot images in a loop
    for k in range(len(fog_y)):
        ax = fig.add_subplot(sp_int + k + 1)
        ax.plot(xinterp[k], fog_y[k])
        try:
            ax.plot(xdata[k], ydata[k], '.')
        except ValueError:
            continue
        ax.set_title(fnames[k])
        # plt.ylim([0, max(max(ydata))])
    fig.text(0.5, 0.04, r'$Fog\;density\;(Max.\;Extinction\;Coeff.,\;[1/m])$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$Solar\;luminosity\;[cd / m^2] * 1e9$', ha='center', va='center', rotation='vertical')
    # Handle saving
    if save:
        if 'sun' in str(BASE_PATH):
            sname = f'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/analysis_sun.png'
        elif 'sky' in str(BASE_PATH):
            sname = f'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/analysis_sky.png'
        plt.savefig(sname, bbox_inches="tight")


def plot_luminance_images(images, save: bool = False):
    for image in images[0]:
        # plt.title('Raytraced image output')
        # plt.imshow(image)
        # plt.grid(False)
        # plt.show()

        plt.title('Luminance map')
        plt.imshow(0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2])
        plt.grid(False)
        plt.colorbar()
        plt.show()

        light_details = {'X': 5, 'Y': 0, 'Z': 1, 'I': 18000}
        surface_details = {'X': 5, 'Y': 0, 'Z': 0, 'rho': 0.75}
        sensor_details = {'X': 0, 'Y': 0, 'Z': 1}
        print(f'Theoretical luminance: {calc_theoretical_luminance(light_details, surface_details, sensor_details)}')
        print(f'Max luminance: {np.max(0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2])}')


def plot_image_comparison(fnames, save: bool = False):
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
        # Handle saving
        if save:
            sname = f'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/timelapse_{fnames[i]}_{img_type}.png'
            plt.savefig(sname, bbox_inches="tight")
        i += 1


def plot_anisotropy_analysis(save: bool = False):
    # Plot fog anisotropy analysis data
    labels = []
    for element in fnames:
        if 'hg00' in element:
            labels.append('g=0.00')
        elif 'hg10' in element and 'hg100' not in element:
            labels.append('g=0.10')
        elif 'hg20' in element:
            labels.append('g=0.20')
        elif 'hg50' in element:
            labels.append('g=0.50')
        elif 'hg100' in element:
            labels.append('g=1.0')
    for k in range(len(fnames)):
        plt.plot(xinterp[k], fog_y[k], label=r'$y=I_0\exp^{-2\pi r^2 nkZ}$')
        plt.plot(xdata[k], ydata[k], '.', label=f'Measured data, {labels[k]}')
    # plt.yscale('log')
    plt.title('Solar luminosity at zenith, albedo = 0.7')
    plt.xlabel(r'$Fog\;density\;(Max.\;Extinction\;Coeff.,\;[1/m])$')
    plt.ylabel(r'$Solar\;luminosity\;[cd / m^2] * 1e9$')
    plt.legend()
    if save:
        sname = 'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/hg_slices.png'
        plt.savefig(sname, bbox_inches="tight")


def plot_anisotropy_analysis_mesh(images, save=save):
    densities = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1]
    # gs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    gs = [0]
    spacing = 550

    # For each image in fnames, get the .tiff
    meshes = []
    for folders in images:
        i = 0
        tmp_meshes = []
        for image in folders:
            b2 = image[:, :, 2]
            # plt.imshow(0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2])
            # plt.colorbar()
            # plt.grid(False)
            plt.show()
            # Find the indices of the fixed/max x and y values
            if 'sky' in str(BASE_PATH):
                max_x = b2.shape[0] // 2 - 50
                max_y = b2.shape[1] // 2
            else:
                max_x, max_y = np.unravel_index(b2.argmax(), b2.shape)
            max_x_arr = slice(max_x - spacing, max_x + spacing)
            max_y_arr = slice(max_y - spacing, max_y + spacing)
            tmp_meshes.append(b2[max_x_arr, max_y_arr])

            # Handle removal of dead pixels
            if i == 0 or i == 1:
                # tmp_meshes[0][tmp_meshes[0] > 5] = 1
                plt.imshow(tmp_meshes[0])
                plt.colorbar()
                plt.grid(False)
                plt.show()
                i += 1
        meshes.append(tmp_meshes)

    # Sort the meshes such that all images with g=0.0 are in one array, then g=0.2, etc.
    for k in range(len(gs)):
        # Plot mesh with increasing g on subplots
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f'Solar luminosity at zenith, albedo = 0.7, g={gs[k]}')
        for j in range(len(meshes[k])):
            ax = fig.add_subplot(230 + j + 1)
            ax.grid(False)
            plt.title(f'Fog density={densities[j]}')
            ax.imshow(meshes[k][j])

    if save:
        sname = f'C:/Users/ElliotLondon/Documents/PythonLocal/rFproControlPy/results/fog/anisotropy_analysis.png'
        plt.savefig(sname, bbox_inches="tight")


if __name__ == '__main__':
    # Define the sequence of folders to grab data from
    # Fog HG anisotropy and albedo tests
    # fnames = ['sweep_fog_hg00_alb07_max', 'sweep_fog_hg50_alb07_max', 'sweep_fog_hg100_alb07_max']
    # fnames = ['sweep_fog_hg00_alb07_max', 'sweep_fog_hg10_alb07_max', 'sweep_fog_hg20_alb07_max']
    # fnames = ['sweep_fog_hg00_alb1_max', 'sweep_fog_hg10_alb1_max', 'sweep_fog_hg20_alb1_max']
    fnames = ['timelapse_clear_full']

    # Fog image comparisons
    if 'sky' in str(BASE_PATH):
        fnames = ['fog_raytrace_1414_hg000_alb04', 'fog_raytrace_1414_hg020_alb04', 'fog_raytrace_1414_hg040_alb04',
                  'fog_raytrace_1414_hg060_alb04', 'fog_raytrace_1414_hg080_alb04', 'fog_raytrace_1414_hg100_alb04']
        # fnames = ['fog_raytrace_1414_hg000_alb07', 'fog_raytrace_1414_hg020_alb07', 'fog_raytrace_1414_hg040_alb07',
        #           'fog_raytrace_1414_hg060_alb07', 'fog_raytrace_1414_hg080_alb07', 'fog_raytrace_1414_hg100_alb07']
        # fnames = ['fog_raytrace_2114_hg000_alb07', 'fog_raytrace_2114_hg020_alb07', 'fog_raytrace_2114_hg040_alb07',
        #           'fog_raytrace_2114_hg060_alb07', 'fog_raytrace_2114_hg080_alb07', 'fog_raytrace_2114_hg100_alb07']
    elif 'fog_anisotropy' in str(BASE_PATH):
        # Fog anisotropy analysis
        fnames = ['fog_raytrace_1414_hg000_alb07', 'fog_raytrace_1414_hg020_alb07', 'fog_raytrace_1414_hg040_alb07',
                  'fog_raytrace_1414_hg060_alb07', 'fog_raytrace_1414_hg080_alb07', 'fog_raytrace_1414_hg100_alb07']
    elif 'sun' in str(BASE_PATH):
        fnames = ['fog_raster_1414_hg000_alb07', 'fog_raytrace_1414_hg000_alb07']

    # Anything temporary to override current analysis
    fnames = [core.global_variables.data_name]

    # Loop through all folders
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

        # Switches for what is being analysed
        if 'month' in fname:
            analyse_luminosity_month(folders)
        elif 'turbidity' in fname:
            analyse_turbidity_sequence(folders)
        elif 'fog_turbidity' in str(BASE_PATH):
            images.append(get_tiff_image(fname)[0])
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
            # Images
            if img_type == 'LDR':
                images.append(get_ldr_image(folders))
            elif img_type == 'HDR':
                images.append(get_tiff_image(folders))
            # break
            analyse_luminosity_sequence(folders)

    plot_luminance_data(fnames, save=save)          # Plot Beer's law curves for luminances
    # plot_luminance_images(images, save=save)        # Plot raw image, luminance images
    # plot_image_comparison(fnames, save=save)        # Compare HDR/LDR images in 2x3 grid
    # plot_anisotropy_analysis(save=save)             # Plot exponential curves of HG anisotropy
    plot_anisotropy_analysis_mesh(images, save=save)  # Plot slices of image data to check anisotropy behaviour

    # Call show at the end to plot stuff!
    plt.show()
