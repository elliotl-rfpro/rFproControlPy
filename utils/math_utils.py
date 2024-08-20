import numpy as np
from scipy.constants import pi


def sin_func(x, a, b, c, d):
    return a * np.sin(b * (x + c)) + d


def cos_func(x, a, b, c, d):
    return a * np.cos(b * (x + c)) + d


def gauss_func(x, a, x0, sigma):
    # Gaussian function
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def visibility_func(x, k, a, c, b):
    # Exponential decay with an unknown param k
    return a * np.exp(- k * (x - b)) + c


def create_circular_mask(h, w, center, radius):
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def calc_spherical_coord(distance, angle):
    # For a given object, O, and camera, C, at a fixed distance D, calculate the X and Y position of the camera, and
    # The angle that permits it to face the object.
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return x, y


def calc_theoretical_luminance(light_details, surface_details, sensor_details) -> float:
    """
    Given an input light, surface, and camera orientation, use trig and Lambert's cosine law to calculate the
    expected luminance to be measured by a sensor. Assumes that the light is directly above the point to be measured
    upon the surface.
    """
    # First, find the angle that the sensor makes with respect to the light source
    ls_dist_z = abs(light_details['Z'] - surface_details['Z'])          # Distance between light and surface, [m]
    ss_dist_x = abs(surface_details['X'] - sensor_details['X'])         # Distance between surface and sensor, X, [m]
    ss_dist_z = abs(surface_details['Z'] - sensor_details['Z'])         # Distance between surface and sensor, Z, [m]
    elev_angle = np.arctan(ss_dist_z / ss_dist_x)                       # Elevation angle, [rads]
    theta = np.deg2rad(90 - np.rad2deg(elev_angle))                     # Lambert cos angle, [rads]
    distance = np.sqrt(ss_dist_x ** 2 + ss_dist_z ** 2)                 # Distance between surface and sensor, [m]

    # Adjust luminous intensity [lm] with Lambert cosine law (reflection viewed at an angle). I = I_0 cos(theta)
    intensity_ = light_details['I'] * np.cos(theta)

    # Calculate illuminance [lm/m^2] from I / 4 pi d^2
    illuminance = intensity_ / (ls_dist_z + distance) ** 2

    # Calculate luminance [cd/m^2] from L = R*rho/pi
    luminance = illuminance * surface_details['rho'] / pi

    return luminance
