#!/usr/bin/env python3

"""
Generate representative PWS images for the paper by processing images in the specified 
folder using a Look-Up Table (LUT) for sigma to D conversion. It applies a correction 
factor and performs analysis based on the specified analysis and ROI names.

Author:
    Originally created by Jane Frederick (Backman Lab, Northwestern University).
    Last updated by Jane Frederick on 2024-12-15.
"""

from Generate_Red_PWS_D_Images import process_image_folder

# Define the path to the folder containing the image files
folder_path = (
    r"C:/Users/janef/OneDrive - Northwestern University/Documents - Backman Lab - Jane Frederick/Jane Frederick/Papers/CDA Paper/V3 (2021-2024)/Figures/PWS Images"
)

# Define the path to the Look-Up Table (LUT) file
lut_path = (
    r"C:/Users/janef/OneDrive/Documents/GitHub/CPMC_CDA/CPT_Paper/Codes/CPT_Paper_Figures_Version_3/Sigma to D/SigmaToD_LUT_LCPWS1.csv"
)

# Define the directory where processed images will be saved (None to use default)
save_path = None

# Define the name of the analysis to be performed
second_analysis_name = 'p01'

# Define the name of the region of interest (ROI) to be processed
second_roi_name = 'nucleus'

# Define the correction factor for image processing
PWS1_correction = 3

# Process the image folder with the specified parameters
process_image_folder(
    folder_path=folder_path,
    lut_path=lut_path,
    save_path=save_path,
    correction_factor=PWS1_correction,
    analysis_name=second_analysis_name,
    roi_name=second_roi_name,
)