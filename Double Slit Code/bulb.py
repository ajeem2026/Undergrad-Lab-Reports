import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Load bulb experimental data
data = pd.read_csv("Bulb_Double-Slit_Photon_Count_Data.csv")
x_experimental = data["Detector Position (mm)"].values
y_experimental = data["Photon Count"].values

# Clean data: Remove NaNs or infs
mask = np.isfinite(y_experimental)
x_experimental_clean = x_experimental[mask]
y_experimental_clean = y_experimental[mask]

# Normalize intensity
y_experimental_normalized = y_experimental_clean / np.max(y_experimental_clean)

# Poisson error for photon count (sqrt(N)), normalized
photon_error = np.sqrt(y_experimental_clean) / np.max(y_experimental_clean)

# Theoretical intensity model (no stretch factor)
def double_slit_intensity(x, I0, x0, D, d, lambda_, L):
    theta = (x - x0) * 1e-3 / L  # x in mm to m
    k = 2 * np.pi / lambda_
    phi = k * D * np.sin(theta)
    psi = k * d * np.sin(theta)
    intensity = I0 * (np.sin(phi/2) / (phi/2 + 1e-10))**2 * np.cos(psi/2)**2
    return intensity

# Wrapper for curve fitting (fit I0 and x0 only)
def fit_function(x, I0, x0):
    D = 0.085e-3         # Slit width (m)
    d = 0.250e-3         # Slit separation (m)
    lambda_ = 541e-9     # Central wavelength (m)
    L = 0.5              # Distance to detector (m)
    return double_slit_intensity(x, I0, x0, D, d, lambda_, L)

# Initial guesses
initial_guess = [1.0, 0.0]  # [I0, x0]

# Fit data
popt, pcov = curve_fit(
    fit_function,
    x_experimental_clean,
    y_experimental_normalized,
    p0=initial_guess,
    sigma=photon_error,
    absolute_sigma=True
)

# Extract fitted parameters
I0_fit, x0_fit = popt
I0_err, x0_err = np.sqrt(np.diag(pcov))

# Generate theoretical fit curve
x_theoretical = np.linspace(min(x_experimental_clean), max(x_experimental_clean), 1000)
y_theoretical = double_slit_intensity(
    x_theoretical,
    I0_fit, x0_fit,
    D=0.085e-3, d=0.250e-3,
    lambda_=546e-9, L=0.5
)

# Plot
plt.figure(figsize=(12, 6))

# Error bars with data points
plt.errorbar(
    x_experimental_clean,
    y_experimental_normalized,
    yerr=photon_error,
    fmt='o',
    markersize=4,
    capsize=3,
    label='Bulb Experimental Data (Poisson Error)',
)

# Connect experimental points with a smooth line
plt.plot(
    x_experimental_clean,
    y_experimental_normalized,
    '-', linewidth=1.0, color='gray', alpha=0.6,
    label='Experimental Trend'
)

# Theoretical model
plt.plot(
    x_theoretical,
    y_theoretical,
    '-', linewidth=1.5,
    label=f'Theoretical Fit'
)

plt.xlabel("Detector Position (mm)")
plt.ylabel("Photon Count (normalized)")
plt.title("Double-Slit Pattern (Bulb Source)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print fitted parameters
print("Fitted Parameters (Bulb Data, No Stretch):")
print(f"I0 = {I0_fit:.4f} ± {I0_err:.4f}")
print(f"x0 = {x0_fit:.4f} ± {x0_err:.4f}")
