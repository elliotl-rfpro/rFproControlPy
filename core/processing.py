from osgeo import gdal
from core.global_files import DATA_PATH, img_num
import numpy as np
from matplotlib import pyplot as plt
import OpenEXR as exr
import Imath


def plot_rgb_data(data, colour: str, label: str):
    plt.plot(data, '.', color=colour, label=label, markersize=1)
    plt.title("HDR image readout")
    plt.xlabel("Pixel #")
    plt.ylabel("Intensity (knits)")
    # Fix duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def get_exr_image(folders):
    images = []
    for file in folders:
        # Load the file
        exrfile = exr.InputFile(file)
        header = exrfile.header()

        # Get headers, width, height
        dw = header['dataWindow']
        isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        channelData = dict()

        # Convert all channels in the image to np arrays
        for c in header['channels']:
            C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
            C = np.fromstring(C, dtype=np.float32)
            C = np.reshape(C, isize)
            channelData[c] = C

        # Reconstruct image
        colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
        img = np.concatenate([channelData[c][..., np.newaxis] for c in colorChannels], axis=2)
        images.append(img)

    return images


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
