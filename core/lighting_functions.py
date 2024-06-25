import numpy as np


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
