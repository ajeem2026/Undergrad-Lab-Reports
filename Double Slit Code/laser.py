import numpy as np
import matplotlib.pyplot as plt

# Lab parameters
D = 0.085e-3  # slit width (m)
d = 0.353e-3  # slit separation (m)
wavelength = 670e-9  # laser wavelength (m)
k = 2 * np.pi / wavelength  # wave number
L = 0.50  # distance from slits to detector (m)

# Refined theoretical model
def refined_double_slit_intensity(x, center, amplitude, offset):
    theta = (x - center) / 1000 / L  # mm to m to radians
    phi = k * D * np.sin(theta)
    delta_phi = k * d * np.sin(theta)
    intensity = (np.cos(delta_phi / 2) ** 2) * (np.sinc(phi / (2 * np.pi)) ** 2)
    return amplitude * intensity + offset

# Voltage data from experiment
laser_voltage_data = np.array([
    0.103, 0.131, 0.122, 0.092, 0.093, 0.171, 0.323, 0.454, 0.466, 0.336,
    0.147, 0.073, 0.240, 0.616, 0.973, 1.040, 0.787, 0.321, 0.045, 0.217,
    0.811, 1.398, 1.607, 1.245, 0.573, 0.084, 0.155, 0.749, 1.499, 1.856,
    1.573, 0.828, 0.210, 0.092, 0.585, 1.268, 1.666, 1.484, 0.887, 0.297,
    0.090, 0.338, 0.804, 1.121, 1.082, 0.712, 0.312, 0.120, 0.187, 0.402,
    0.555, 0.539, 0.387, 0.210, 0.116, 0.113
])

# Normalize voltages
V_max = np.max(laser_voltage_data)
laser_voltage_normalized = laser_voltage_data / V_max

# Error: 2% of the unnormalized values, scaled after normalization
laser_voltage_error = 0.02 * laser_voltage_data / V_max

# Detector positions (mm)
laser_positions = np.linspace(-6, 6, len(laser_voltage_data))

# Best-fit parameters (you may refit if needed)
popt_refined = [0.0, 1.7, 0.05]  # center shift, amplitude, offset

# Theoretical fit
laser_theoretical_fit = refined_double_slit_intensity(laser_positions, *popt_refined)
laser_theoretical_fit_normalized = laser_theoretical_fit / np.max(laser_theoretical_fit)

# Plot with error bars and a line over experimental data
plt.figure(figsize=(12, 6))

# Error bars for experimental data
plt.errorbar(
    laser_positions,
    laser_voltage_normalized,
    yerr=laser_voltage_error,
    fmt='o',
    label='Experimental Data (2% error)',
    capsize=3,
    color='tab:blue'
)

# Smooth line connecting experimental data
plt.plot(
    laser_positions,
    laser_voltage_normalized,
    linestyle='-',
    linewidth=1.5,
    color='tab:blue',
    label='Experimental Trend'
)

# Theoretical fit line
plt.plot(
    laser_positions,
    laser_theoretical_fit_normalized,
    '--',
    linewidth=2,
    color='tab:orange',
    label='Theoretical Fit (Normalized)'
)

# Labels and styling
plt.title('Double-Slit Interference: Laser Source', fontsize=14)
plt.xlabel('Detector Position (mm)', fontsize=12)
plt.ylabel('Normalized Voltage / Intensity', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
