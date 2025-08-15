import pyart
import matplotlib.pyplot as plt
import glob
import os

# Directories
raw_dir = 'data/raw/'
output_dir = 'data/processed/images/'
os.makedirs(output_dir, exist_ok=True)

# Process files directly from raw_dir
radar_files = glob.glob(os.path.join(raw_dir, 'KBOX_SDUS81_NBXBOX_*'))
if not radar_files:
    print("No matching files found. Check the glob pattern.")
    exit(1)

processed_count = 0
for file_path in radar_files:
    try:
        radar = pyart.io.read_nexrad_level3(file_path)
        display = pyart.graph.RadarDisplay(radar)
        fig = plt.figure(figsize=(8, 8))
        product = 'reflectivity'  # Default product
        if 'reflectivity' not in radar.fields:
            product = list(radar.fields.keys())[0]  # Use first available field
        display.plot(product, 0, title=os.path.basename(file_path))  # First sweep
        display.set_limits(xlim=(-500, 500), ylim=(-500, 500))  # Adjust as needed
        output_file = os.path.join(output_dir, os.path.basename(file_path) + '.png')
        plt.savefig(output_file)
        plt.close(fig)
        processed_count += 1
        print(f"Saved image {processed_count}/{len(radar_files)}: {output_file}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"Total images processed: {processed_count}")