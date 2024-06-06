# Functions for the analysis of fog, mist, haze, smog, etc.
import numpy as np
from scipy.constants import pi
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})
images = 50


def calc_fog_func(x, i0, r):
    # Basic behaviour taken from https://apps.dtic.mil/sti/trecms/pdf/AD1159612.pdf, plate 7. Essentially Beer's law.
    return i0 * np.exp(-2 * pi * r ** 2 * x)


def hg_phase_func(cos_angle, g):
    """
    Calculate value of Henyey-Greenstein phase function.
    https://pbr-book.org/3ed-2018/Volume_Scattering/Phase_Functions
    Henyey, L. G., and J. L. Greenstein. 1941. Diffuse radiation in the galaxy. Astrophysical Journal 93, 70–83.
    """
    denom = 1 + g * g + 2 * g * cos_angle
    return 1/(4*pi) * (1 - g * g) / (denom * np.sqrt(denom))


def check_hg_phase(plot: bool = False):
    # Function to investigate the behaviour of theta and g in the HG phase function.
    angles = np.linspace(0, 360, 1000)
    gs = [0, 0.3, 0.5, 0.8, 0.9, 0.95]
    y = []
    for ghg in gs:
        sol = []
        for angle in angles:
            cos_angle_ = np.cos(np.deg2rad(angle))
            sol.append(hg_phase_func(cos_angle_, ghg))
        y.append(sol)

    if plot:
        for i in range(len(gs)):
            plt.plot(angles, y[i], label=f'g={gs[i]}')
        plt.yscale('log')
        plt.legend()
        plt.ylabel(r'$\log_{10}\left({\frac{1}{4\pi} \frac{1 - g^2}{(1 + g^2 + 2gcos(\theta))^{3/2}}}\right)$')
        plt.xlabel(r'$\theta [°]$')
        plt.show()

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for i in range(len(gs)):
            ax.plot(angles * pi / 180, y[i], label=f'g={gs[i]}')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend(loc="lower right")
        plt.show()


def get_scattering_coeffs(albedo, fog_density):
    scattering_coeff = fog_density * albedo
    absorption_coeff = fog_density - scattering_coeff
    return scattering_coeff, absorption_coeff


def check_scattering_coeffs(plot: bool = False):
    albedos = np.linspace(0.0, 1.0, 11)
    fog_density = np.linspace(0.0, 0.4, 100)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    x, y = np.meshgrid(albedos, fog_density)
    alpha = []
    rho = []
    for i in albedos:
        a = []
        r = []
        for j in fog_density:
            tmp_a, tmp_r = get_scattering_coeffs(i, j)
            a.append(tmp_a)
            r.append(tmp_r)
        alpha.append(a)
        rho.append(r)

    if plot:
        ax = fig.add_subplot(131, projection='3d')
        ax.plot_surface(x, y, np.asarray(alpha).transpose(), cmap=cm.coolwarm, alpha=0.5,
                               linewidth=0, antialiased=False, label='Scattering coeff.')
        plt.xlabel('Albedo')
        plt.ylabel('Fog Density')
        plt.legend()
        ax = fig.add_subplot(132, projection='3d')
        ax.plot_surface(x, y, np.asarray(rho).transpose(), cmap=cm.coolwarm, alpha=0.5,
                           linewidth=0, antialiased=False, label='Absorption coeff.')
        plt.xlabel('Albedo')
        plt.ylabel('Fog Density')
        plt.legend()
        ax = fig.add_subplot(133, projection='3d')
        ax.plot_surface(x, y, np.asarray(alpha).transpose() - np.asarray(rho).transpose(), cmap=cm.coolwarm, alpha=0.5,
                           linewidth=0, antialiased=False, label='Absorption coeff.')
        plt.xlabel('Albedo')
        plt.ylabel('Fog Density')
        plt.legend()
        plt.show()


# check_hg_phase(plot=True)
# check_scattering_coeffs(plot=True)
