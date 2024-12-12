#!/usr/bin/env python3

"""
This module processes raw Partial Wave Spectroscopic (PWS) Microscopy image cubes by:
- Converting Sigma values to D values using a MATLAB-generated Look-Up Table (LUT).
- Applying Region of Interest (ROI) masks to identify nuclei.
- Generating pseudocolored images.

Author:
    Originally created by Jane Frederick (Backman Lab, Northwestern University).
    Last updated by Jane Frederick on 2024-12-15.

Functions:
    - configure_logging:
        Sets up the logging configuration for the application, defining the logging 
        level and message format.
    - configure_matplotlib_style:
        Sets global aesthetics for Matplotlib plots, including resolution, font 
        settings, and color schemes.
    - initialize_save_path:
        Ensures the existence of the directory where processed images will be saved.
    - generate_red_colormap:
        Creates a custom red colormap with adjustable brightness for pseudocolor imaging.
    - save_colorbars:
        Generates and saves horizontal and vertical D colorbars based on the provided 
        red colormap.
    - generate_gray_colormap:
        Creates a grayscale colormap with adjustable brightness for image visualization.
    - generate_and_save_colormaps:
        Produces both red and grayscale colormaps and saves their corresponding colorbars.
    - load_and_fit_lut:
        Loads a Sigma-to-D conversion LUT, fits a polynomial to the data, and saves 
        the function as a plot.
    - get_image_files:
        Retrieves all HDF5 files that match a specific analysis name pattern within a 
        folder and its subdirectories.
    - apply_roi_mask:
        Applies ROI masks from an HDF5 file to the D data, combining multiple masks if 
        present.
    - save_pseudocolor_image:
        Overlays masked D data onto the original D data to create and save a 
        pseudocolor image with red nuclei and gray background.
    - find_brightfield_image:
        Searches for the associated brightfield image ("image_bd.tif") in the parent 
        directories of a given file path.
    - save_brightfield_image:
        Reads, applies a grayscale colormap to, and saves the brightfield image.
    - process_file:
        Processes a single HDF5 file by converting sigma data, applying corrections 
        and ROI masks, and generating images.
    - process_image_folder:
        Orchestrates the processing of PWS image cubes, including setting up logging, 
        initializing directories, and handling each image file.
    - main:
        Serves as the entry point for setting up parameters and initiating the image 
        processing workflow.

Usage Example:
    To process PWS image cubes, use the `process_image_folder` function as follows:
    ```
    process_image_folder(
        folder_path="path/to/image/folder",
        lut_path="path/to/lut/file.csv",
        save_path="path/to/save/directory",
        red_brightness=1.0,
        gray_brightness=0.5,
        min_D=1.5,
        max_D=3.0,
        colorbar_label=r'$D_{pixel}$',
        correction_factor=1.0,
        analysis_name='p0',
        roi_name='nuc',
        image_save_type='tif'
    )
    ```
"""

import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, List, Tuple

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Constants for image processing
# Minimum value for D parameter in images
MIN_IMAGE_D = 1.5
# Maximum value for D parameter in images 
MAX_IMAGE_D = 3
# Default brightness level for red colormap (1.0 = full brightness)
RED_BRIGHTNESS_DEFAULT = 1
# Default brightness level for grayscale colormap (0.25 = 25% brightness)
GRAY_BRIGHTNESS_DEFAULT = 0.25
# Default file format for saving images
IMAGE_SAVE_TYPE_DEFAULT = "tif"
# Default label for colorbar (using LaTeX formatting)
COLORBAR_LABEL_DEFAULT = r'$D_{pixel}$'
# Default correction factor applied to sigma values (1 means data is already corrected)
CORRECTION_FACTOR_DEFAULT = 1.0
# Default name pattern for analysis results
ANALYSIS_NAME_DEFAULT = 'p0'
# Default name for region of interest (ROI)
ROI_NAME_DEFAULT = 'nuc'


def configure_logging() -> None:
    """
    Configure the logging settings for the application.

    This function initializes the logging configuration with the following parameters:
    
    - **Level**: Sets the logging level to INFO to capture informational messages and above.
    - **Format**: Defines the log message format to include timestamp, log level, and message.
    - **Date Format**: Specifies the format for the date and time in log entries.
    """
    # Initialize basic logging configuration 
    logging.basicConfig(
        level=logging.INFO,  # Set logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # Define message format
        datefmt='%Y-%m-%d %H:%M:%S'  # Specify date/time format
    )


def configure_matplotlib_style(dpi: float = 600, font_weight: str = 'bold', 
                             cap_size: float = 2.5) -> None:
    """
    Configure global plotting aesthetics for Matplotlib.

    This function updates Matplotlib's rcParams to set the overall style for plots, including 
    resolution, font settings, colors, and other visual parameters to ensure consistency across
    all figures generated by the application.

    Parameters:
        dpi (float): Dots per inch for figures. Higher values = higher resolution. Default 600.
        font_weight (str): Weight of font used in plot texts. Default 'bold'.
        cap_size (float): Length of error bar caps in points. Default 2.5.

    Raises:
        Exception: If an error occurs while updating Matplotlib settings.
    """
    try:
        # Update Matplotlib's runtime configuration with specified styles
        mpl.rcParams.update({
            "figure.dpi": dpi,  # Set figure resolution in DPI
            "axes.edgecolor": "k",  # Set axes edges to black
            "text.color": "k",  # Set default text color to black 
            "axes.labelcolor": "k",  # Set axes label color to black
            "xtick.color": "k",  # Set x-axis tick color to black
            "ytick.color": "k",  # Set y-axis tick color to black
            "font.weight": font_weight,  # Set font weight
            "axes.labelweight": font_weight,  # Set axes label weight
            "axes.titleweight": font_weight,  # Set axes title weight
            "errorbar.capsize": cap_size,  # Set error bar cap size
            "legend.frameon": False,  # Remove legend frame
            "font.family": "sans-serif",  # Set font family to sans-serif
            "font.sans-serif": ["Arial", "Helvetica"],  # Set preferred fonts
            "axes.titlesize": 14,  # Set title font size
            "axes.labelsize": 12,  # Set label font size 
            "xtick.labelsize": 10,  # Set x-tick label size
            "ytick.labelsize": 10,  # Set y-tick label size
            "legend.fontsize": 10  # Set legend font size
        })
    except Exception as e:
        # Log error if updating rcParams fails
        logging.error(f"Error configuring Matplotlib style: {e}")
        raise


def initialize_save_path(folder_path: str, save_path: str = None) -> Path:
    """
    Initialize and create the directory for saving processed images.

    This function verifies that the provided folder_path exists and is a directory. If a 
    save_path is specified, it uses that path for saving the processed images. Otherwise, it 
    defaults to creating a subdirectory named "PWS_Red_Colormap_Images" within the provided 
    folder_path.

    Args:
        folder_path (str): The path to the main folder containing image data.
        save_path (str, optional): The desired path to save processed images. Defaults to None.

    Returns:
        Path: A Path object pointing to the directory where processed images will be saved.

    Raises:
        NotADirectoryError: If folder_path does not exist or is not a directory.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        msg = f"Folder path {folder_path} does not exist or is not a directory."
        raise NotADirectoryError(msg)
    
    # Use provided save_path or default to specific subdirectory
    save_dir = Path(save_path) if save_path else folder / "PWS_Red_Colormap_Images"
    
    # Create save directory if needed, including parent directories
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return save_dir

def generate_red_colormap(
    red_brightness: float = RED_BRIGHTNESS_DEFAULT
) -> mpl.colors.LinearSegmentedColormap:
    """
    Generate a custom red colormap for pseudocolor images.

    This function creates a linear segmented colormap that transitions from black to a 
    specified shade of red. The brightness of the red can be adjusted by the 
    `red_brightness` parameter.

    Parameters:
        red_brightness (float): 
            The intensity of the red color in the colormap. Must be between 0 (black) and 1 
            (full red). Default is RED_BRIGHTNESS_DEFAULT.

    Returns:
        mpl.colors.LinearSegmentedColormap: 
            A Matplotlib colormap object representing the red colormap.

    Raises:
        ValueError: 
            If `red_brightness` is not between 0 and 1.
    """
    # Validate that red_brightness is within the allowed range
    if not 0 <= red_brightness <= 1:
        raise ValueError("red_brightness must be between 0 and 1")
    
    # Create a linear segmented colormap from black to the specified red
    return mpl.colors.LinearSegmentedColormap.from_list(
        "red_colormap", 
        [(0, 0, 0), (red_brightness, 0, 0)], 
        N=256
    )


def save_colorbars(red_cmap: mpl.colors.LinearSegmentedColormap, save_path: Path,
                    min_D: float = MIN_IMAGE_D, max_D: float = MAX_IMAGE_D,
                    colorbar_label: str = COLORBAR_LABEL_DEFAULT,
                    image_save_type: str = IMAGE_SAVE_TYPE_DEFAULT) -> None:
    """
    Save horizontal and vertical colorbars for the provided red colormap.

    This function generates and saves both horizontal and vertical colorbars based on the 
    specified red colormap. It normalizes the color scale using the provided minimum and 
    maximum D values and labels the colorbars accordingly.

    Parameters:
        red_cmap (mpl.colors.LinearSegmentedColormap): The red colormap for colorbars.
        save_path (Path): The directory path where the colorbar images will be saved.
        min_D (float, optional): The minimum D value for normalization. Default MIN_IMAGE_D.
        max_D (float, optional): The maximum D value for normalization. Default MAX_IMAGE_D.
        colorbar_label (str, optional): Label for colorbars. Default COLORBAR_LABEL_DEFAULT.
        image_save_type (str, optional): File format for saving. Default IMAGE_SAVE_TYPE_DEFAULT.

    Raises:
        Exception: If there is an error during the colorbar creation or saving process.
    """
    # Normalize the color scale based on minimum and maximum D values
    norm = mpl.colors.Normalize(vmin=min_D, vmax=max_D)

    # Define the orientations and corresponding figure sizes for the colorbars
    orientations = [("Horizontal", (2, 0.5)), ("Vertical", (0.5, 2))]

    for orientation, size in orientations:
        # Create a new figure and axis for the colorbar with specified size
        fig, ax = plt.subplots(figsize=size)

        # Create the colorbar based on the red colormap and normalization
        mpl.colorbar.ColorbarBase(ax, cmap=red_cmap, norm=norm, 
                                    orientation=orientation.lower())

        # Set the appropriate label depending on the orientation
        if orientation == "Horizontal":
            ax.set_xlabel(colorbar_label)
        else:
            ax.set_ylabel(colorbar_label)

        # Define the filename for the colorbar image
        save_name = f"PWS_D_Colorbar_{orientation}.{image_save_type}"

        # Save the colorbar image with high DPI and transparent background
        plt.savefig(save_path / save_name, dpi=600, transparent=True, 
                    bbox_inches="tight")

        # Close the figure to free up memory
        plt.close(fig)

def generate_gray_colormap(
    gray_brightness: float = GRAY_BRIGHTNESS_DEFAULT
) -> mpl.colors.ListedColormap:
    """
    Generate a grayscale colormap for visualization purposes.

    Parameters:
        gray_brightness (float, optional):
            A value between 0 and 1 that determines the brightness of the gray colormap.
            If set to `None`, the colormap will consist entirely of black.
            Defaults to `GRAY_BRIGHTNESS_DEFAULT`.

    Returns:
        mpl.colors.ListedColormap:
            A Matplotlib ListedColormap object representing the grayscale colormap.

    Raises:
        ValueError:
            If `gray_brightness` is not between 0 and 1.
    """
    # If gray_brightness is None, return a colormap of all black
    if gray_brightness is None:
        return mpl.colors.ListedColormap(['black'] * 256)
    # Ensure gray_brightness is within the valid range
    if not 0 <= gray_brightness <= 1:
        raise ValueError("gray_brightness must be between 0 and 1")
    # Generate a grayscale array with the specified brightness
    gray = mpl.colormaps["gray"](np.linspace(0, gray_brightness, 256))
    # Create and return the ListedColormap from the grayscale array
    return mpl.colors.ListedColormap(gray)


def generate_and_save_colormaps(
    save_path: Path,
    red_brightness: float = RED_BRIGHTNESS_DEFAULT,
    gray_brightness: float = GRAY_BRIGHTNESS_DEFAULT,
    min_D: float = MIN_IMAGE_D,
    max_D: float = MAX_IMAGE_D,
    colorbar_label: str = COLORBAR_LABEL_DEFAULT,
    image_save_type: str = IMAGE_SAVE_TYPE_DEFAULT
) -> Tuple[mpl.colors.LinearSegmentedColormap, mpl.colors.ListedColormap]:
    """
    Generate and save custom colormaps for pseudocolor images.

    This function creates a red colormap and a grayscale colormap with specified brightness
    levels. It also saves horizontal and vertical colorbars for the red colormap to the 
    provided save path.

    Parameters:
        save_path (Path): Directory path where colorbar images will be saved.
        red_brightness (float, optional): Red color intensity. Default RED_BRIGHTNESS_DEFAULT.
        gray_brightness (float, optional): Gray colormap brightness. Default GRAY_BRIGHTNESS_DEFAULT.
        min_D (float, optional): Minimum D for normalization. Default MIN_IMAGE_D.
        max_D (float, optional): Maximum D for normalization. Default MAX_IMAGE_D.
        colorbar_label (str, optional): Colorbar label. Default COLORBAR_LABEL_DEFAULT.
        image_save_type (str, optional): File format for saving. Default IMAGE_SAVE_TYPE_DEFAULT.

    Returns:
        Tuple[mpl.colors.LinearSegmentedColormap, mpl.colors.ListedColormap]: 
            A tuple containing the red colormap and the grayscale colormap.

    Raises:
        ValueError: If the brightness values are not within the valid range.
    """
    # Generate the red colormap with the specified brightness
    red_cmap = generate_red_colormap(red_brightness)
    
    # Save the horizontal and vertical colorbars for the red colormap
    save_colorbars(red_cmap, save_path, min_D, max_D, colorbar_label, image_save_type)
    
    # Generate the grayscale colormap with the specified brightness
    gray_cmap = generate_gray_colormap(gray_brightness)
    
    # Return both the red and grayscale colormaps
    return red_cmap, gray_cmap


def load_and_fit_lut(
    lut_path: str, 
    save_path: Path,
    image_save_type: str = IMAGE_SAVE_TYPE_DEFAULT
) -> Tuple[np.ndarray, np.poly1d]:
    """
    Load Sigma-to-D conversion LUT and fit a polynomial.

    This function reads a lookup table (LUT) from a CSV file that maps Sigma values to D values.
    It then fits a polynomial function to the LUT data for conversion purposes. The function also
    generates and saves a plot comparing the original LUT data with the fitted polynomial.

    Parameters:
        lut_path (str): The file path to the LUT CSV file.
        save_path (Path): The directory path where the plot will be saved.
        image_save_type (str, optional): The file format for saving the plot. 
            Default is IMAGE_SAVE_TYPE_DEFAULT.

    Returns:
        Tuple[np.ndarray, np.poly1d]: A tuple containing the LUT data as a structured numpy 
        array and the fitted polynomial function.

    Raises:
        FileNotFoundError: If the LUT file is not found at the specified path.
        ValueError: If the LUT CSV does not contain 'Sigma' and 'D' columns.
    """
    # Check if the LUT file exists
    if not Path(lut_path).is_file():
        raise FileNotFoundError(f"LUT file not found at {lut_path}")
    
    # Load the LUT data from the CSV file
    convlut = np.genfromtxt(lut_path, delimiter=',', names=True)
    
    # Validate that the necessary columns are present in the LUT data
    if "Sigma" not in convlut.dtype.names or "D" not in convlut.dtype.names:
        raise ValueError("LUT CSV must contain 'Sigma' and 'D' columns.")
    
    # Fit a polynomial function to the LUT data
    convfunc = np.poly1d(np.polyfit(convlut["Sigma"], convlut["D"], 10))
    
    # Define a range of Sigma values for plotting the fitted polynomial
    sigma_range = np.linspace(convlut["Sigma"].min(), convlut["Sigma"].max(), 100)
    
    # Create a plot to compare the original LUT data with the fitted polynomial
    plt.figure()
    plt.plot(convlut["Sigma"], convlut["D"], 'o', label='MATLAB Data')
    plt.plot(sigma_range, convfunc(sigma_range), '-', label='Python Fit')
    plt.title(r'Sigma to $D$ Conversion')
    plt.xlabel('Sigma')
    plt.ylabel(r'$D$')
    plt.legend()
    
    # Define the file path for saving the plot
    plot_path = save_path / f"Sigma_to_D_Conversion_Function.{image_save_type}"
    
    # Save the plot with high resolution and transparent background
    plt.savefig(plot_path, dpi=600, transparent=True, bbox_inches='tight')
    
    # Close the plot to free up memory
    plt.close()
    
    return convlut, convfunc


def get_image_files(
    folder: Path, 
    analysis_name: str = ANALYSIS_NAME_DEFAULT
) -> List[Path]:
    """
    Retrieve all HDF5 files matching the analysis name pattern.

    Parameters:
        analysis_name (str): The analysis name pattern to match. Default ANALYSIS_NAME_DEFAULT.
        folder (Path): The root directory to search for HDF5 files.

    Returns:
        List[Path]: A list of Path objects pointing to the matching HDF5 files.
    """
    # Use glob to search for HDF5 files matching the analysis name pattern
    return list(folder.glob(f"**/analysisResults_{analysis_name}.h5"))


def apply_roi_mask(roi_path: Path, D_data: np.ndarray) -> np.ma.array:
    """
    Apply ROI mask to the D data.

    Parameters:
        roi_path (Path): The path to the HDF5 file containing ROI masks.
        D_data (np.ndarray): The D data to which the ROI mask will be applied.

    Returns:
        np.ma.array: The masked D data if ROI exists, otherwise the original D data.
    """
    try:
        with h5py.File(roi_path, 'r') as roifile:
            # Load all ROI masks from the HDF5 file
            roi_masks = [roifile[key + '/mask'][()] for key in roifile]
            # Combine all masks into a single mask
            allrois = np.any(roi_masks, axis=0)
            # Apply the combined mask to the D data
            return np.ma.array(D_data, mask=~allrois)
    except Exception as e:
        # Log an error message if applying the ROI mask fails
        logging.error(f"Error applying ROI mask from {roi_path}: {e}")
        # Return the original D data if no ROI mask is applied
        return D_data


def save_pseudocolor_image(
    save_path: Path, cell_folder: str, D_data: np.ndarray,
    masked_D_data: np.ma.array, red_cmap: mpl.colors.LinearSegmentedColormap,
    gray_cmap: mpl.colors.ListedColormap, min_D: float = MIN_IMAGE_D,
    max_D: float = MAX_IMAGE_D, image_save_type: str = IMAGE_SAVE_TYPE_DEFAULT
) -> None:
    """
    Save the pseudocolor image.

    Parameters:
        save_path (Path): Directory path where the pseudocolor image will be saved.
        cell_folder (str): Name of the cell folder used in the filename.
        D_data (np.ndarray): Original D data to be displayed in grayscale.
        masked_D_data (np.ma.array): Masked D data to be overlaid in red.
        red_cmap (mpl.colors.LinearSegmentedColormap): Colormap for masked D data.
        gray_cmap (mpl.colors.ListedColormap): Colormap for original D data.
        min_D (float): Minimum D value for normalization. Default MIN_IMAGE_D.
        max_D (float): Maximum D value for normalization. Default MAX_IMAGE_D.
        image_save_type (str): File format for saving. Default IMAGE_SAVE_TYPE_DEFAULT.
    """
    try:
        # Define the filename for saving the pseudocolor image
        save_name = f"{cell_folder}_PWS.{image_save_type}"
        # Display the original D data in grayscale
        plt.imshow(D_data, cmap=gray_cmap, vmin=min_D, vmax=max_D)
        # Overlay the masked D data in red
        plt.imshow(masked_D_data, cmap=red_cmap, vmin=min_D, vmax=max_D)
        plt.axis("off") # Hide the axes for a cleaner image
        # Save the pseudocolor image with transparent background
        plt.savefig(save_path / save_name, transparent=True,
                   bbox_inches="tight", pad_inches=0)
        plt.close() # Close the plot to free up memory
    except Exception as e:
        # Log an error message if saving the pseudocolor image fails
        logging.error(f"Error saving pseudocolor image for {cell_folder}: {e}")


def find_brightfield_image(file_path: Path) -> Path:
    """
    Find the brightfield image associated with the given file.

    This function searches for a brightfield image named "image_bd.tif" in the parent 
    directories of the provided file path. It returns the path to the brightfield image if 
    found, otherwise it returns None.

    Parameters:
        file_path (Path): The path to the file for which the brightfield image is needed.

    Returns:
        Path: The path to the brightfield image if found, otherwise None.
    """
    # Iterate through the parent directories of the given file path
    for parent in file_path.parents:
        # Define the expected path for the brightfield image
        bf_image = parent / "image_bd.tif"
        # Check if the brightfield image exists at the defined path
        if bf_image.exists():
            return bf_image  # Return the path if the image is found
    return None  # Return None if the brightfield image is not found


def save_brightfield_image(
    save_path: Path, 
    cell_folder: str, 
    bf_path: Path,
    image_save_type: str = IMAGE_SAVE_TYPE_DEFAULT
) -> None:
    """
    Save the brightfield image.

    This function reads a brightfield image from the specified path, applies a grayscale 
    colormap, and saves the image to the specified save path with the given file format.

    Parameters:
        save_path (Path): The directory path where the brightfield image will be saved.
        cell_folder (str): The name of the cell folder used in the filename.
        bf_path (Path): The path to the brightfield image file.
        image_save_type (str): The file format for saving. Defaults to IMAGE_SAVE_TYPE_DEFAULT.

    Raises:
        Exception: If there is an error during the image reading or saving process.
    """
    try:
        # Read the brightfield image from the specified path
        bf_image = mpl.image.imread(bf_path)
        
        # Display the brightfield image using the grayscale colormap
        plt.imshow(bf_image, cmap='gray')
        
        # Hide the axes for a cleaner image
        plt.axis("off")
        
        # Save the brightfield image with high DPI and transparent background
        save_name = f"{cell_folder}_BF.{image_save_type}"
        plt.savefig(save_path / save_name, transparent=True,
                   bbox_inches="tight", pad_inches=0)
        
        # Close the plot to free up memory
        plt.close()
    except Exception as e:
        # Log an error message if saving the brightfield image fails
        logging.error(f"Error saving brightfield image for {bf_path}: {e}")


def process_file(
    file_path: Path,
    folder: Path,
    save_dir: Path,
    conversion_function: Callable[[np.ndarray], np.ndarray],
    correction_factor: float = CORRECTION_FACTOR_DEFAULT,
    roi_name: str = ROI_NAME_DEFAULT,
    red_colormap: mpl.colors.LinearSegmentedColormap = generate_red_colormap(),
    gray_colormap: mpl.colors.ListedColormap = generate_gray_colormap(),
    min_D: float = MIN_IMAGE_D,
    max_D: float = MAX_IMAGE_D,
    image_save_type: str = IMAGE_SAVE_TYPE_DEFAULT,
) -> None:
    """
    Process a single HDF5 file to generate pseudocolor and brightfield images.

    This function reads sigma data from an HDF5 file, applies a correction factor, converts 
    the sigma data to D values, applies an ROI mask, and generates pseudocolor and 
    brightfield images. The images are saved to the specified directory.

    Parameters:
        file_path (Path): Path to the HDF5 file to be processed.
        folder (Path): Base folder containing the HDF5 files.
        save_dir (Path): Directory where the processed images will be saved.
        conversion_function (Callable[[np.ndarray], np.ndarray]): Function to convert sigma 
            data to D values.
        correction_factor (float, optional): Factor to correct sigma data. 
            Defaults to CORRECTION_FACTOR_DEFAULT.
        roi_name (str, optional): Name of the ROI file. Defaults to ROI_NAME_DEFAULT.
        red_colormap (mpl.colors.LinearSegmentedColormap, optional): Colormap for 
            nucleus. Defaults to generate_red_colormap().
        gray_colormap (mpl.colors.ListedColormap, optional): Colormap for cytoplasm.
            Defaults to generate_gray_colormap().
        min_D (float, optional): Minimum D value. Defaults to MIN_IMAGE_D.
        max_D (float, optional): Maximum D value. Defaults to MAX_IMAGE_D.
        image_save_type (str, optional): File format to save images. 
            Defaults to IMAGE_SAVE_TYPE_DEFAULT.
    """
    # Get the relative path of the file with respect to the base folder
    relative_path = file_path.relative_to(folder)

    # Extract the cell folder name from the relative path
    cell_folder = next((part for part in relative_path.parts 
                       if part.startswith("Cell")), None)
    if not cell_folder:
        logging.warning(f"Cell folder not found in {file_path}. Skipping.")
        return

    # Get the relative folder path up to the cell folder
    relative_folder = Path(*relative_path.parts[:relative_path.parts.index(cell_folder)])

    # Define the full save path for the processed images
    full_save_path = save_dir / relative_folder

    # Create the save directory if it does not exist
    full_save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Open the HDF5 file and read the sigma data, applying the correction factor
        with h5py.File(file_path, "r") as h5_file:
            sigma_data = np.clip(h5_file["rms"][()] * correction_factor, 0, 0.5)
    except Exception as e:
        logging.error(f"Error reading 'rms' from {file_path}: {e}")
        return

    # Convert the sigma data to D values using the conversion function
    D_data = conversion_function(sigma_data)

    # Define the path to the ROI file
    roi_path = folder / relative_folder / cell_folder / f'ROI_{roi_name}.h5'

    # Apply the ROI mask to the D data
    masked_D_data = apply_roi_mask(roi_path, D_data)

    # Save the pseudocolor image
    save_pseudocolor_image(full_save_path, cell_folder, D_data, masked_D_data,
                          red_colormap, gray_colormap, min_D, max_D, image_save_type)

    # Find the associated brightfield image
    bf_path = find_brightfield_image(file_path)
    if bf_path:
        # Save the brightfield image if found
        save_brightfield_image(full_save_path, cell_folder, bf_path, 
                             image_save_type)
    else:
        logging.warning(f"Brightfield image not found for {file_path}")

def process_image_folder(
    folder_path: str,
    lut_path: str,
    save_path: str,
    red_brightness: float = RED_BRIGHTNESS_DEFAULT,
    gray_brightness: float = GRAY_BRIGHTNESS_DEFAULT,
    min_D: float = MIN_IMAGE_D,
    max_D: float = MAX_IMAGE_D,
    colorbar_label: str = COLORBAR_LABEL_DEFAULT,
    correction_factor: float = CORRECTION_FACTOR_DEFAULT,
    analysis_name: str = ANALYSIS_NAME_DEFAULT,
    roi_name: str = ROI_NAME_DEFAULT,
    image_save_type: str = IMAGE_SAVE_TYPE_DEFAULT
) -> None:
    """
    Process PWS image cubes to generate pseudocolor and brightfield images.

    This function performs the following steps:
    1. Configures logging and Matplotlib style.
    2. Validates the folder path.
    3. Initializes the save directory.
    4. Generates and saves colormaps.
    5. Loads and fits the LUT.
    6. Retrieves the list of image files to process.
    7. Processes each image file and saves the results.

    Parameters:
        folder_path (str): Path to the folder containing the image files.
        lut_path (str): Path to the Look-Up Table (LUT) file.
        save_path (str): Directory where processed images will be saved.
        red_brightness (float, optional): Brightness level for the red channel.
            Defaults to RED_BRIGHTNESS_DEFAULT.
        gray_brightness (float, optional): Brightness level for the gray channel.
            Defaults to GRAY_BRIGHTNESS_DEFAULT.
        min_D (float, optional): Minimum D value for image processing.
            Defaults to MIN_IMAGE_D.
        max_D (float, optional): Maximum D value for image processing.
            Defaults to MAX_IMAGE_D.
        colorbar_label (str, optional): Label for the colorbar in the images.
            Defaults to COLORBAR_LABEL_DEFAULT.
        correction_factor (float, optional): Correction factor for image processing.
            Defaults to CORRECTION_FACTOR_DEFAULT.
        analysis_name (str, optional): Name of the analysis to be performed.
            Defaults to ANALYSIS_NAME_DEFAULT.
        roi_name (str, optional): Name of the region of interest (ROI) to be processed.
            Defaults to ROI_NAME_DEFAULT.
        image_save_type (str, optional): File type for saving processed images.
            Defaults to IMAGE_SAVE_TYPE_DEFAULT.

    Raises:
        Exception: If there is an error during the image processing.
    """
    try:
        # Step 1: Configure logging and Matplotlib style
        configure_logging()
        configure_matplotlib_style()
        
        # Step 2: Validate the folder path
        folder = Path(folder_path)
        if not folder.is_dir():
            logging.error(
                f"Folder path {folder_path} does not exist or is not a directory."
            )
            return
        
        # Step 3: Initialize the save directory
        save_dir = initialize_save_path(folder_path, save_path)
        
        # Step 4: Generate and save colormaps
        red_cmap, gray_cmap = generate_and_save_colormaps(
            save_path=save_dir,
            red_brightness=red_brightness,
            gray_brightness=gray_brightness,
            min_D=min_D,
            max_D=max_D,
            colorbar_label=colorbar_label,
            image_save_type=image_save_type,
        )
        
        # Step 5: Load and fit the LUT
        _, conversion_function = load_and_fit_lut(
            lut_path, save_dir, image_save_type
        )
        
        # Step 6: Retrieve the list of image files to process
        image_files = get_image_files(folder, analysis_name)
        logging.info(f"Found {len(image_files)} files to process.")
        
        # Step 7: Process each image file and save the results
        for i, file_path in enumerate(image_files, start=1):
            logging.info(f"Processing file {i}/{len(image_files)}: {file_path}")
            process_file(
                file_path=file_path,
                folder=folder,
                save_dir=save_dir,
                conversion_function=conversion_function,
                correction_factor=correction_factor,
                roi_name=roi_name,
                red_colormap=red_cmap,
                gray_colormap=gray_cmap,
                min_D=min_D,
                max_D=max_D,
                image_save_type=image_save_type,
            )
    except Exception as e:
        logging.error(f"Error processing image folder: {e}")
        raise


def main() -> None:
    """
    Launches the graphical user interface for processing PWS image cubes.

    This main function sets up a Tkinter-based GUI that allows users to:
        - Select the image folder containing PWS data.
        - Choose the LUT (Look-Up Table) file for Sigma-to-D conversion.
        - Specify the directory where processed images will be saved.
        - Adjust various processing parameters such as brightness levels, D value ranges,
          correction factors, and more.

    Upon configuring the necessary inputs and parameters, the user can initiate the
    processing workflow, which will generate pseudocolor and brightfield images based
    on the provided settings.

    The GUI ensures user-friendly interaction with the processing pipeline, providing
    error messages for invalid inputs and confirmation upon successful completion.
    """

    def browse_folder():
        """Opens a dialog for the user to select the image folder."""
        folder_selected = filedialog.askdirectory(title="Select Image Folder")
        if folder_selected:
            folder_path_var.set(folder_selected)

    def browse_lut():
        """Opens a dialog for the user to select the LUT (CSV) file."""
        lut_selected = filedialog.askopenfilename(
            title="Select LUT File", filetypes=[("CSV", "*.csv")]
        )
        if lut_selected:
            lut_path_var.set(lut_selected)

    def browse_save():
        """Opens a dialog for the user to select the save directory."""
        save_selected = filedialog.askdirectory(title="Select Save Directory")
        if save_selected:
            save_path_var.set(save_selected)

    def start_processing():
        """
        Initiates the image processing workflow with the configured parameters.

        This function retrieves user inputs from the GUI, validates them, and calls the
        `process_image_folder` function to start processing the PWS image cubes. It provides
        error messages for invalid inputs and notifies the user upon successful completion.
        """
        folder = folder_path_var.get()
        lut = lut_path_var.get()
        save = save_path_var.get()

        # Retrieve and validate additional processing parameters
        try:
            red_brightness = float(red_brightness_var.get())
            gray_brightness = float(gray_brightness_var.get())
            min_D = float(min_D_var.get())
            max_D = float(max_D_var.get())
            correction_factor = float(correction_factor_var.get())
            colorbar_label = colorbar_label_var.get()
            image_save_type = image_save_type_var.get()
            analysis_name = analysis_name_var.get()
            roi_name = roi_name_var.get()
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Please enter valid numerical values for brightness, D values, and correction factor.",
            )
            return

        # Ensure that mandatory fields are selected
        if not folder or not lut:
            messagebox.showerror(
                "Selection Error",
                "Please select both the Image Folder and LUT File to proceed.",
            )
            return

        # Start processing the image folder with the provided parameters
        process_image_folder(
            folder_path=folder,
            lut_path=lut,
            save_path=save if save else folder,
            red_brightness=red_brightness,
            gray_brightness=gray_brightness,
            min_D=min_D,
            max_D=max_D,
            colorbar_label=colorbar_label,
            correction_factor=correction_factor,
            analysis_name=analysis_name,
            roi_name=roi_name,
            image_save_type=image_save_type,
        )
        messagebox.showinfo(
            "Processing Complete", "PWS image processing completed successfully."
        )

    # Initialize the main Tkinter window
    root = tk.Tk()
    root.title("Generate Red PWS D Images")
    root.geometry("600x600")
    root.resizable(False, False)

    # Create the main frame within the window for layout management
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Image Folder Selection Section
    ttk.Label(main_frame, text="Image Folder:").grid(
        row=0, column=0, sticky=tk.W, pady=5
    )
    folder_path_var = tk.StringVar()
    ttk.Entry(
        main_frame, width=50, textvariable=folder_path_var
    ).grid(row=0, column=1, pady=5, padx=5)
    ttk.Button(
        main_frame, text="Browse", command=browse_folder
    ).grid(row=0, column=2, padx=5, pady=5)

    # LUT File Selection Section
    ttk.Label(main_frame, text="LUT File:").grid(
        row=1, column=0, sticky=tk.W, pady=5
    )
    lut_path_var = tk.StringVar()
    ttk.Entry(
        main_frame, width=50, textvariable=lut_path_var
    ).grid(row=1, column=1, pady=5, padx=5)
    ttk.Button(main_frame, text="Browse", command=browse_lut).grid(
        row=1, column=2, padx=5, pady=5
    )

    # Save Directory Selection Section
    ttk.Label(main_frame, text="Save Directory:").grid(
        row=2, column=0, sticky=tk.W, pady=5
    )
    save_path_var = tk.StringVar()
    ttk.Entry(
        main_frame, width=50, textvariable=save_path_var
    ).grid(row=2, column=1, pady=5, padx=5)
    ttk.Button(main_frame, text="Browse", command=browse_save).grid(
        row=2, column=2, padx=5, pady=5
    )

    # Additional Processing Parameters Section
    params_frame = ttk.LabelFrame(
        main_frame, text="Processing Parameters", padding="10"
    )
    params_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=tk.EW)

    # Red Brightness Configuration
    ttk.Label(
        params_frame, text="Red Channel Brightness (1.0 = 100%):"
    ).grid(row=0, column=0, sticky=tk.W, pady=5)
    red_brightness_var = tk.StringVar(value=str(RED_BRIGHTNESS_DEFAULT))
    ttk.Entry(
        params_frame, width=10, textvariable=red_brightness_var
    ).grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

    # Gray Brightness Configuration
    ttk.Label(
        params_frame, text="Gray Channel Brightness (1.0 = 100%):"
    ).grid(row=1, column=0, sticky=tk.W, pady=5)
    gray_brightness_var = tk.StringVar(value=str(GRAY_BRIGHTNESS_DEFAULT))
    ttk.Entry(
        params_frame, width=10, textvariable=gray_brightness_var
    ).grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

    # Minimum D Value Configuration
    ttk.Label(
        params_frame, text="Minimum D Value in Image:"
    ).grid(row=2, column=0, sticky=tk.W, pady=5)
    min_D_var = tk.StringVar(value=str(MIN_IMAGE_D))
    ttk.Entry(
        params_frame, width=10, textvariable=min_D_var
    ).grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

    # Maximum D Value Configuration
    ttk.Label(
        params_frame, text="Maximum D Value in Image:"
    ).grid(row=3, column=0, sticky=tk.W, pady=5)
    max_D_var = tk.StringVar(value=str(MAX_IMAGE_D))
    ttk.Entry(
        params_frame, width=10, textvariable=max_D_var
    ).grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)

    # Correction Factor Configuration
    ttk.Label(
        params_frame,
        text="Extra Reflection Correction (1.0 = data already corrected):",
    ).grid(row=4, column=0, sticky=tk.W, pady=5)
    correction_factor_var = tk.StringVar(value=str(CORRECTION_FACTOR_DEFAULT))
    ttk.Entry(
        params_frame, width=10, textvariable=correction_factor_var
    ).grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)

    # Image Save Type Configuration
    ttk.Label(
        params_frame, text="Image Save Format (e.g., tif, png):"
    ).grid(row=5, column=0, sticky=tk.W, pady=5)
    image_save_type_var = tk.StringVar(value=IMAGE_SAVE_TYPE_DEFAULT)
    ttk.Entry(
        params_frame, width=10, textvariable=image_save_type_var
    ).grid(row=5, column=1, sticky=tk.W, pady=5, padx=5)

    # Colorbar Label Configuration
    ttk.Label(
        params_frame, text="Colorbar Label (LaTeX Supported):"
    ).grid(row=6, column=0, sticky=tk.W, pady=5)
    colorbar_label_var = tk.StringVar(value=COLORBAR_LABEL_DEFAULT)
    ttk.Entry(
        params_frame, width=20, textvariable=colorbar_label_var
    ).grid(row=6, column=1, sticky=tk.W, pady=5, padx=5)

    # Analysis Name Configuration
    ttk.Label(
        params_frame, text="PWS Analysis Identifier:"
    ).grid(row=7, column=0, sticky=tk.W, pady=5)
    analysis_name_var = tk.StringVar(value=ANALYSIS_NAME_DEFAULT)
    ttk.Entry(
        params_frame, width=10, textvariable=analysis_name_var
    ).grid(row=7, column=1, sticky=tk.W, pady=5, padx=5)

    # ROI Name Configuration
    ttk.Label(
        params_frame, text="Region of Interest (ROI) Name:"
    ).grid(row=8, column=0, sticky=tk.W, pady=5)
    roi_name_var = tk.StringVar(value=ROI_NAME_DEFAULT)
    ttk.Entry(
        params_frame, width=10, textvariable=roi_name_var
    ).grid(row=8, column=1, sticky=tk.W, pady=5, padx=5)

    # Start Processing Button
    ttk.Button(
        main_frame, text="Start Processing", command=start_processing
    ).grid(row=4, column=0, columnspan=3, pady=20)

    # Run the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
