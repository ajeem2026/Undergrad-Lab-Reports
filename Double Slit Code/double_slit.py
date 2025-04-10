# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load experimental data
# data = pd.read_csv("offset.csv")
# x_experimental = data["Detector Position (mm)"].values
# y_experimental = data["Voltage (arb. units)"].values

# # Adjusted parameters for 6 peaks in [-6, 6] mm
# D = 0.085e-3       # Slit width (m) (fixed)
# d = 0.250e-3       # Reduced slit separation to stretch fringes (originally 0.353e-3)
# lambda_ = 670e-9   # Wavelength (m) (laser)
# L = 0.5            # Distance to detector (m)
# I0 = 1.8           # Central intensity (arb. units)
# x0 = 0.0           # Central position (mm)
# stretch_factor = 1.8  # Horizontal stretch factor (1.0 = no stretch, >1.0 = stretch)

# # Theoretical calculation with horizontal stretch
# def double_slit_intensity(x, I0, x0, D, d, lambda_, L, stretch_factor):
#     theta = (x / stretch_factor - x0) * 1e-3 / L  # Apply stretch to x
#     k = 2 * np.pi / lambda_
#     phi = k * D * np.sin(theta)   # Single-slit diffraction term
#     psi = k * d * np.sin(theta)   # Double-slit interference term
#     intensity = I0 * (np.sin(phi/2) / (phi/2 + 1e-10))**2 * np.cos(psi/2)**2
#     return intensity

# # Generate theoretical curve
# x_theoretical = np.linspace(-6, 6, 1000)
# y_theoretical = double_slit_intensity(x_theoretical, I0, x0, D, d, lambda_, L, stretch_factor)

# # Plot
# plt.figure(figsize=(12, 6))
# plt.plot(x_experimental, y_experimental, 'o', markersize=4, label='Experimental Data')
# plt.plot(x_theoretical, y_theoretical, '-', linewidth=1.5, label=f'Theoretical Fit (stretch={stretch_factor:.1f}x)')
# plt.xlabel("Detector Position (mm)")
# plt.ylabel("Intensity (arb. units)")
# plt.title("Double-Slit Pattern with Horizontal Stretch")
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Load experimental data
data = pd.read_csv("offset.csv")
x_experimental = data["Detector Position (mm)"].values
y_experimental = data["Voltage (arb. units)"].values

# Clean data: Remove NaN or inf values
mask = np.isfinite(y_experimental)
x_experimental_clean = x_experimental[mask]
y_experimental_clean = y_experimental[mask]

# Rescale experimental data to max=1
y_experimental_normalized = y_experimental_clean / np.max(y_experimental_clean)

# PMT error (2%)
pmt_error = 0.02 * y_experimental_normalized  # 2% of each data point

# Theoretical model with horizontal stretch
def double_slit_intensity(x, I0, x0, D, d, lambda_, L, stretch_factor):
    theta = (x / stretch_factor - x0) * 1e-3 / L  # Apply stretch to x
    k = 2 * np.pi / lambda_
    phi = k * D * np.sin(theta)   # Single-slit diffraction term
    psi = k * d * np.sin(theta)   # Double-slit interference term
    intensity = I0 * (np.sin(phi/2) / (phi/2 + 1e-10))**2 * np.cos(psi/2)**2
    return intensity

# Wrapper function for fitting (only fitting I0, x0, stretch_factor)
def fit_function(x, I0, x0, stretch_factor):
    D = 0.085e-3      # Fixed slit width (m)
    d = 0.250e-3       # Fixed slit separation (m)
    lambda_ = 670e-9   # Fixed wavelength (m)
    L = 0.5            # Fixed distance to detector (m)
    return double_slit_intensity(x, I0, x0, D, d, lambda_, L, stretch_factor)

# Initial parameter guesses
initial_guess = [1.0, 0.0, 1.8]  # [I0, x0, stretch_factor]

# Perform the fit (with uncertainties)
popt, pcov = curve_fit(
    fit_function,
    x_experimental_clean,
    y_experimental_normalized,
    p0=initial_guess,
    sigma=pmt_error,
    absolute_sigma=True,
)

# Extract fitted parameters
I0_fit, x0_fit, stretch_factor_fit = popt
I0_err, x0_err, stretch_factor_err = np.sqrt(np.diag(pcov))

# Generate theoretical curve with fitted parameters
x_theoretical = np.linspace(-6, 6, 1000)
y_theoretical = double_slit_intensity(
    x_theoretical,
    I0_fit, x0_fit,
    D=0.085e-3, d=0.250e-3,
    lambda_=670e-9, L=0.5,
    stretch_factor=stretch_factor_fit,
)

# Plot
plt.figure(figsize=(12, 6))
plt.errorbar(
    x_experimental_clean,
    y_experimental_normalized,
    yerr=pmt_error,
    fmt='o',
    markersize=4,
    capsize=3,
    label='Experimental Data (±2% error)',
)
plt.plot(
    x_theoretical,
    y_theoretical,
    '-',
    linewidth=1.5,
    label=f'Theoretical Fit:\nI0={I0_fit:.2f}±{I0_err:.2f}',
)
plt.xlabel("Detector Position (mm)")
plt.ylabel("Intensity (volts)")
plt.title("Double-Slit Pattern with Error Bars and Fit")
plt.legend()
plt.grid(True)
plt.show()

# Print fitted parameters
print(f"Fitted Parameters:")
print(f"I0 = {I0_fit:.4f} ± {I0_err:.4f}")
