#Code to post process results
#2025-06-20 Merlin Tétrault-Leclerc
#_________________________________________________________________________________
#_________________________________________________________________________________

#LIBRARIES
#_________________________________________________________________________________
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.optimize import curve_fit
from collections import defaultdict

#CODE
#_________________________________________________________________________________

# set the folder that holds the data to be processed
filepath  = 'SIMRESULTS_Reference'

# Find all result files
files = glob(os.path.join(filepath, 'mass_*.csv'))

# Group files by (motor, solar, battery, path)
groups = defaultdict(list)

for file in files:
    # Parse parameters from filename
    # Example: mass_100_motor_250_solar_1.0_battery_1000_CGV.csv
    basename = os.path.basename(file)
    parts = basename.replace('.csv', '').split('_')
    mass = float(parts[1])
    motor = float(parts[3])
    solar = float(parts[5])
    battery = float(parts[7])
    path = '_'.join(parts[8:])  # In case path name has underscores

    key = (motor, solar, battery, path)
    groups[key].append((mass, file))

# Prepare a list to collect all summary rows
summary_rows = []

for key, mass_files in groups.items():
    motor, solar, battery, path = key
    mass_files.sort()
    for mass, file in mass_files:
        # Read only the last line
        with open(file, 'rb') as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
        columns = last_line.strip().split(',')
        distance = float(columns[1])  # Adjust index if needed
        avg_speed = distance/(15*8*60*60)
        # Add a row with all parameters and results
        summary_rows.append([mass, motor, solar, battery, path, distance, avg_speed])

# Save to CSV
summary_df = pd.DataFrame(summary_rows, columns=[
    'mass', 'motor', 'solar', 'battery', 'path', 'final_distance', 'avg_speed'
])
summary_df.to_csv('simulation_summary.csv', index=False)