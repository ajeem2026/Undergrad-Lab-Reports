import pandas as pd
import numpy as np

def process_data_file_for_histogram(input_filename, output_filename):
    """
    Reads a .data file with two columns:
      1. A timing field: If < 40000, it represents the pulse interval in nanoseconds;
         if >= 40000, it indicates a timeout (with the units digit giving the timeout count).
      2. A Unix epoch timestamp (seconds since 1 Jan 1970).
    
    For the purpose of creating a muon decay time histogram, we are only interested in the valid
    pulse intervals (i.e. where the timing field is less than 40000). This function:
      - Reads the file,
      - Flags timeouts,
      - Extracts the pulse intervals from valid rows,
      - Filters out the rows corresponding to timeouts,
      - Writes a CSV file containing only the pulse interval data.
    """
    # Read the .data file assuming whitespace-delimited values and no header row.
    df = pd.read_csv(
        input_filename,
        delim_whitespace=True,
        header=None,
        names=["time_field", "computer_time"]
    )
    
    # Determine which rows are timeouts (i.e. time_field >= 40000).
    df["is_timeout"] = df["time_field"] >= 40000
    
    # For valid pulses (time_field < 40000), the pulse interval is the value itself.
    # For timeouts, we set the pulse interval to NaN so they can be removed.
    df["pulse_interval_ns"] = np.where(df["is_timeout"], np.nan, df["time_field"])
    
    # Filter out timeouts: keep only rows with a valid pulse interval.
    valid_df = df.dropna(subset=["pulse_interval_ns"]).copy()
    
    # Optionally, if you want to work with a specific datatype (e.g., float), convert here.
    valid_df["pulse_interval_ns"] = valid_df["pulse_interval_ns"].astype(float)
    
    # If desired, convert the Unix timestamp to a human-readable datetime.
    # (Not needed for the histogram, but available if you want to inspect timing information.)
    valid_df["datetime"] = pd.to_datetime(valid_df["computer_time"], unit="s")
    
    # For your histogram, you may only need the pulse intervals.
    histogram_df = valid_df[["pulse_interval_ns"]]
    
    # Write the filtered data to a CSV file without the index.
    histogram_df.to_csv(output_filename, index=False)
    print(f"Filtered data for muon decay histogram saved to '{output_filename}'.")

if __name__ == '__main__':
    # Specify your input .data file and desired output CSV file.
    input_file = "/Users/jeem/downloads/muon.data"        # Replace with the path to your .data file
    output_file = "histogram_data.csv"  # Output file containing only pulse intervals
    
    process_data_file_for_histogram(input_file, output_file)
