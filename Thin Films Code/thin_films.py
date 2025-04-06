import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'si02_si_refSample.csv'  # Make sure the file is in your working directory
df = pd.read_csv(file_path)

# Clean up column names
df.columns = ['Wavelength (nm)', 'Reflectance (%)']

# Convert reflectance values to numeric (just in case quotes or formatting issues exist)
df['Reflectance (%)'] = pd.to_numeric(df['Reflectance (%)'], errors='coerce')

# Plotting the reflectance spectrum
plt.figure(figsize=(10, 6))
plt.plot(df['Wavelength (nm)'], df['Reflectance (%)'], color='blue', linewidth=1.5)
plt.title('Reflectance Spectrum of SiO₂/Si Sample')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (%)')
plt.grid(True)
plt.tight_layout()
plt.show()


