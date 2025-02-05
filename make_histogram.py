import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

def exponential_fit(x, A, lambda_):
    return A * np.exp(-lambda_ * x)

# Load the data
file_path = "/Users/jeem/downloads/histogram_data.csv"
df = pd.read_csv(file_path)
data = df['pulse_interval_ns']

# Histogram settings
bins = 20  # Fixed number of bins
hist_values, bin_edges = np.histogram(data, bins=bins, density=True)

# Compute bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit an exponential function (assuming muon decay follows an exponential distribution)
popt, pcov = curve_fit(exponential_fit, bin_centers, hist_values, p0=[max(hist_values), 1/max(data)])

# Generate fitted curve
x_fit = np.linspace(min(data), max(data), 1000)
y_fit = exponential_fit(x_fit, *popt)

# Plot histogram with fit
plt.figure(figsize=(8, 6))
# plt.hist(data, bins=bins, edgecolor="black", density=True, alpha=0.6, color='b', label='Muon Data')
plt.hist(data, bins=bins, edgecolor="black",alpha=0.6, color='b', label='Muon Data')
plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit: A={popt[0]:.2f}, λ={popt[1]:.5f}')
plt.xlabel('Pulse Interval (ns)')
plt.ylabel('Probability Density')
plt.title('Muon Decay Time Histogram')
plt.legend()
plt.grid(True)
plt.savefig("muon_histogram_fit.png")
plt.show()

# Supplementary analysis: Log-log plot
plt.figure(figsize=(8, 6))
plt.hist(data, bins=bins, density=True, alpha=0.6, color='b', label='Muon Data')
plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit: A={popt[0]:.2f}, λ={popt[1]:.5f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Pulse Interval (ns) (log scale)')
plt.ylabel('Probability Density (log scale)')
plt.title('Muon Decay Time Histogram (Log-Log Scale)')
plt.legend()
plt.grid(True, which='both')
plt.savefig("muon_histogram_loglog.png")
plt.show()

# Extract lambda from the fit
lambda_fit = popt[1]
muon_lifetime = 1 / lambda_fit  # Convert lambda to tau

print(f"Estimated Muon Lifetime: {muon_lifetime:.2f} ns")
