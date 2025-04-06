import numpy as np
import matplotlib.pyplot as plt

def intensity(theta, I0, d, D, wavelength):
  """
  Calculates the intensity of light in a double-slit diffraction pattern.

  Args:
    theta: Array of angles in radians.
    I0: Maximum intensity.
    d: Slit separation.
    D: Slit width.
    wavelength: Wavelength of light.

  Returns:
    Array of intensities.
  """
  alpha = np.pi * d * np.sin(theta) / wavelength
  beta = np.pi * D * np.sin(theta) / wavelength
  
  # Avoid division by zero
  sinc_term = np.where(beta != 0, np.sin(beta) / beta, 1)

  return I0 * (np.cos(alpha))**2 * (sinc_term)**2

# Parameters (using provided values)
I0 = 1.0  # Maximum intensity (arbitrary)
D = 0.085e-3  # Slit width (0.085 mm = 0.085e-3 meters)
d = 0.353e-3  # Slit separation (0.353 mm = 0.353e-3 meters)
wavelength_mean = 0.670e-6 # mean wavelength in meters
wavelength_deviation = 0.005e-6 # wavelength deviation in meters

# Generate angles (adjust range as needed)
theta_max = 0.01 # adjust max angle to display more or less of the pattern.
theta = np.linspace(-theta_max, theta_max, 1000)

# Calculate intensity for different wavelengths (mean +/- deviation)
wavelengths = [wavelength_mean - wavelength_deviation, wavelength_mean, wavelength_mean + wavelength_deviation]
intensities = []

for wavelength in wavelengths:
    intensities.append(intensity(theta, I0, d, D, wavelength))

# Plotting
plt.figure(figsize=(10, 6))

labels = [f"λ = {wavelength_mean - wavelength_deviation*1e6:.3f} µm",
          f"λ = {wavelength_mean*1e6:.3f} µm",
          f"λ = {wavelength_mean + wavelength_deviation*1e6:.3f} µm"]

for i, intensity_values in enumerate(intensities):
    plt.plot(theta, intensity_values, label=labels[i])

plt.xlabel('Angle (radians)')
plt.ylabel('Intensity')
plt.title('Double-Slit Diffraction Pattern (Varying Wavelength)')
plt.grid(True)
plt.legend() # add legend
plt.show()