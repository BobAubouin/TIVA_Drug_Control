from pathlib import Path
import os
import sys


# Warn the user thath the script might take a while to run
print("This script will take a while to run. Please be patient.")

# create the results folder
data_folder = Path("Results_data")
image_folder = Path("Results_images")
data_folder.mkdir(exist_ok=True)
image_folder.mkdir(exist_ok=True)

# Run the PID study
print("------- Running the PID study --------")
os.system("python ./scripts/PID_study.py")

# Run the MHE_MPC study
print("------- Running the MHE_MPC study --------")
os.system("python ./scripts/MHE_MPC_study.py")

# Create the images
print("------- Creating the images --------")
os.system("python ./scripts/obtain_final_results.py")

print("Done!")
print("The results are saved in the Results_data and Results_images folders.")
