"""
Simple script to find min/max bounds of x,y coordinates across all data files.
"""

import pandas as pd
import os

def get_coordinate_bounds(data_dir):
    global_x_min = float('inf')
    global_x_max = float('-inf')
    global_y_min = float('inf')
    global_y_max = float('-inf')
    global_angle_min = float('inf')
    global_angle_max = float('-inf')
    global_speed_min = float('inf')
    global_speed_max = float('-inf')
    
    file_count = 0
    angle_file_count = 0
    speed_file_count = 0
    
    print("Scanning for CSV files...")
    
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.csv') and ('coordinates_highestspeed_' in filename or 'preprocessed' in filename):
                file_path = os.path.join(root, filename)
                file_count += 1
                
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Handle different column names
                    if 'X' in df.columns and 'Y' in df.columns:
                        x_col, y_col = 'X', 'Y'
                    elif 'x' in df.columns and 'y' in df.columns:
                        x_col, y_col = 'x', 'y'
                    else:
                        continue
                    
                    # Get valid coordinates (non-NaN)
                    x_valid = df[x_col].dropna()
                    y_valid = df[y_col].dropna()
                    
                    if len(x_valid) > 0 and len(y_valid) > 0:
                        # Update global bounds
                        global_x_min = min(global_x_min, x_valid.min())
                        global_x_max = max(global_x_max, x_valid.max())
                        global_y_min = min(global_y_min, y_valid.min())
                        global_y_max = max(global_y_max, y_valid.max())
                    
                    # Check for turning angle column (from preprocessed files)
                    if 'turning_angle' in df.columns:
                        angle_valid = df['turning_angle'].dropna()
                        if len(angle_valid) > 0:
                            global_angle_min = min(global_angle_min, angle_valid.min())
                            global_angle_max = max(global_angle_max, angle_valid.max())
                            angle_file_count += 1
                    
                    # Check for speed column (from preprocessed files)
                    if 'speed' in df.columns:
                        speed_valid = df['speed'].dropna()
                        if len(speed_valid) > 0:
                            global_speed_min = min(global_speed_min, speed_valid.min())
                            global_speed_max = max(global_speed_max, speed_valid.max())
                            speed_file_count += 1
                
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
    
    print(f"\nAnalyzed {file_count} files")
    print(f"\nGlobal coordinate bounds:")
    print(f"X: [{global_x_min:.2f}, {global_x_max:.2f}]")
    print(f"Y: [{global_y_min:.2f}, {global_y_max:.2f}]")
    print(f"\nX range: {global_x_max - global_x_min:.2f}")
    print(f"Y range: {global_y_max - global_y_min:.2f}")
    
    if angle_file_count > 0:
        print(f"\nTurning angle bounds (from {angle_file_count} files):")
        print(f"Angle: [{global_angle_min:.2f}°, {global_angle_max:.2f}°]")
        print(f"Angle range: {global_angle_max - global_angle_min:.2f}°")
    else:
        print(f"\nNo turning_angle column found in files")
        global_angle_min = -180.0
        global_angle_max = 180.0
    
    if speed_file_count > 0:
        print(f"\nSpeed bounds (from {speed_file_count} files):")
        print(f"Speed: [{global_speed_min:.4f}, {global_speed_max:.4f}]")
        print(f"Speed range: {global_speed_max - global_speed_min:.4f}")
    else:
        print(f"\nNo speed column found in files")
        global_speed_min = 0.0
        global_speed_max = 1.0
    
    return {
        'x_min': global_x_min,
        'x_max': global_x_max,
        'y_min': global_y_min,
        'y_max': global_y_max,
        'angle_min': global_angle_min,
        'angle_max': global_angle_max,
        'speed_min': global_speed_min,
        'speed_max': global_speed_max,
        'files_analyzed': file_count,
        'angle_files_analyzed': angle_file_count,
        'speed_files_analyzed': speed_file_count
    }

if __name__ == "__main__":
    # Check both raw data and preprocessed data for angle bounds
    data_directories = ["data", "preprocessed_data"]
    
    all_bounds = {}
    
    for data_directory in data_directories:
        if os.path.exists(data_directory):
            print(f"\n=== Analyzing {data_directory} ===")
            bounds = get_coordinate_bounds(data_directory)
            all_bounds[data_directory] = bounds
        else:
            print(f"Directory '{data_directory}' not found, skipping...")
    
    # Use the bounds from the directory that has the most complete data
    if "preprocessed_data" in all_bounds and all_bounds["preprocessed_data"]["angle_files_analyzed"] > 0:
        final_bounds = all_bounds["preprocessed_data"]
        print(f"\n=== Using bounds from preprocessed_data ===")
    elif "data" in all_bounds:
        final_bounds = all_bounds["data"]
        print(f"\n=== Using bounds from data ===")
    else:
        print("No valid data directories found!")
        exit(1)
    
    # Save bounds to a simple text file for reference
    with open("coordinate_bounds.txt", "w") as f:
        f.write(f"Global bounds analysis:\n")
        f.write(f"Files analyzed: {final_bounds['files_analyzed']}\n\n")
        f.write(f"Coordinate bounds:\n")
        f.write(f"X: [{final_bounds['x_min']:.6f}, {final_bounds['x_max']:.6f}]\n")
        f.write(f"Y: [{final_bounds['y_min']:.6f}, {final_bounds['y_max']:.6f}]\n")
        f.write(f"X range: {final_bounds['x_max'] - final_bounds['x_min']:.6f}\n")
        f.write(f"Y range: {final_bounds['y_max'] - final_bounds['y_min']:.6f}\n\n")
        f.write(f"Turning angle bounds:\n")
        f.write(f"Files with angles: {final_bounds['angle_files_analyzed']}\n")
        f.write(f"Angle: [{final_bounds['angle_min']:.6f}°, {final_bounds['angle_max']:.6f}°]\n")
        f.write(f"Angle range: {final_bounds['angle_max'] - final_bounds['angle_min']:.6f}°\n\n")
        f.write(f"Speed bounds:\n")
        f.write(f"Files with speed: {final_bounds['speed_files_analyzed']}\n")
        f.write(f"Speed: [{final_bounds['speed_min']:.6f}, {final_bounds['speed_max']:.6f}]\n")
        f.write(f"Speed range: {final_bounds['speed_max'] - final_bounds['speed_min']:.6f}\n")
    
    print(f"\nBounds saved to 'coordinate_bounds.txt'")
