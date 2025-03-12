#!/usr/bin/env python3

"""
This module provides functions to analyze chromatin packing domains using ChromSTEM analysis.

It includes functions to calculate the voxel mass and the genomic size of a packing domain 
in base pairs based on the ChromSTEM analysis published by Y. Li, A. Eshein, R.K.A. Virk, 
et al. (Science Advances, 2021, 7, eabe4310) and updated by Y. Li, V. Agrawal, R.K.A. Virk, 
et al. (Scientific Reports, 2022, 12, 12198).

Author:
    Originally created by Jane Frederick (Backman Lab, Northwestern University).
    Last updated by Jane Frederick on 2024-12-15.
    
Functions:
    - calculate_bp_per_voxel:
        Calculates the number of base pairs in one voxel.
    - calculate_genomic_size_of_PD: 
        Calculates the genomic size of a packing domain in base pairs.
    - initialize_PD_parameters: 
        Initializes packing domain parameters from a CSV file or returns default values.
    - get_default_HCT116_PD_values:
        Returns default packing domain parameters for HCT116 cells.
    - compute_statistics_from_csv: 
        Reads experimental data from a CSV file and calculates statistics.

Usage Example:
    This module can be used to analyze chromatin packing domains by providing experimental  
    data in a CSV file or using default values for HCT116 cells:
    ```
    file_path = 'path/to/your/csvfile.csv'
    params = initialize_PD_parameters(file_path)
    print(params)
    ```
"""

import logging
import os
import pandas as pd

# Constants
DEFAULT_CHROMSTEM_VOXEL_SIZE_NM = 2  # ChromSTEM voxel size in nm
# Assuming highest intensity in tomogram represents 100% unhydrated DNA
DEFAULT_UNHYDRATED_DNA_DENSITY_G_PER_CM3 = 2  # Unhydrated DNA density in g/cm³
DEFAULT_NUCLEOTIDE_MW_DA = 325  # Average molecular weight of a nucleotide in Daltons
DEFAULT_AMU_TO_G = 1.660e-24  # Conversion factor from atomic mass units to grams

def calculate_bp_per_voxel(
    voxel_size_nm: float = DEFAULT_CHROMSTEM_VOXEL_SIZE_NM,
    dna_density_g_per_cm3: float = DEFAULT_UNHYDRATED_DNA_DENSITY_G_PER_CM3,
    nucleotide_mw_Da: float = DEFAULT_NUCLEOTIDE_MW_DA,
    amu_to_grams: float = DEFAULT_AMU_TO_G
) -> int:
    """
    Calculate the number of base pairs (bp) in one voxel based on the provided parameters.

    Parameters:
        - voxel_size_nm (float, optional): The size of the voxel in nanometers (nm). Default is
            DEFAULT_CHROMSTEM_VOXEL_SIZE_NM (2 nm).
        - dna_density_g_per_cm3 (float, optional): The density of unhydrated DNA in grams per cubic
            centimeter (g/cm³). Default is DEFAULT_UNHYDRATED_DNA_DENSITY_G_PER_CM3 (2 g/cm³).
        - nucleotide_mw_Da (float, optional): The molecular weight of a nucleotide in Dalton (Da).
            Default is DEFAULT_NUCLEOTIDE_MW_DA (325 Da).
        - amu_to_grams (float, optional): Conversion factor from atomic mass units (AMU) to grams(g).
            Default is DEFAULT_AMU_TO_G (1.660e-24 g/AMU).

    Returns:
        - int: The number of base pairs (bp) in one voxel, rounded to the nearest integer.

    References:
        - Y. Li, A. Eshein, R.K.A. Virk, et al. (2021). Nanoscale chromatin imaging and
          analysis platform bridges 4D chromatin organization with molecular function.
          Science Advances.
    """
    # Collect parameters into a dictionary for validation
    parameters = {
        'voxel_size_nm': voxel_size_nm,
        'dna_density_g_per_cm3': dna_density_g_per_cm3,
        'nucleotide_mw_Da': nucleotide_mw_Da,
        'amu_to_grams': amu_to_grams
    }
    # Iterate over each parameter to validate its type and value
    for param_name, param_value in parameters.items():
        # Check if the parameter is a number (int or float)
        if not isinstance(param_value, (int, float)):
            raise TypeError(f"{param_name} must be a number.")
        # Ensure the parameter is greater than zero
        if param_value <= 0:
            raise ValueError(f"{param_name} must be greater than zero.")

    try:
        # Calculate the volume of the voxel in cubic centimeters (cm³)
        voxel_volume_cm3 = (voxel_size_nm * 1e-7) ** 3
        # Calculate the mass of DNA in one voxel in grams (g)
        voxel_dna_mass_g = voxel_volume_cm3 * dna_density_g_per_cm3
        # Calculate the number of nucleotides in one voxel
        nucleotides_per_voxel = voxel_dna_mass_g / (nucleotide_mw_Da * amu_to_grams)
        # Calculate the number of base pairs (bp) in one voxel
        bp_per_voxel = round(nucleotides_per_voxel / 2)
        return bp_per_voxel
    except Exception as e:
        # Log any exceptions that occur during calculation
        logging.error("Error calculating DNA base pairs per voxel: %s", e)
        return 0

def calculate_genomic_size_of_PD(
    D_PD: float,
    A_v: float,
    r_PD: float,
    voxel_size_nm: float = DEFAULT_CHROMSTEM_VOXEL_SIZE_NM,
    **kwargs
) -> int:
    """
    Calculate the genomic size (N_PD) of a packing domain in base pairs (bp).

    Parameters:
        - D_PD (float): Scaling exponent of the packing domain.
        - A_v (float): Volume packing efficiency of the packing domain.
        - r_PD (float): Radius of the packing domain in nanometers (nm).
        - voxel_size_nm (float, optional): The size of the voxel in nanometers (nm).
            Default is DEFAULT_CHROMSTEM_VOXEL_SIZE_NM (2 nm).
        - kwargs (dict): Additional keyword arguments for calculate_bp_per_voxel().

    Returns:
        - int: Genomic size N_PD in base pairs, or 0 if parameters are invalid.

    References:
        - Y. Li, A. Eshein, R.K.A. Virk, et al. (2021). Nanoscale chromatin imaging and
          analysis platform bridges 4D chromatin organization with molecular function.
          Science Advances.
        - Y. Li, V. Agrawal, R.K.A. Virk, et al. (2022). Analysis of three-dimensional 
          chromatin packing domains by ChromSTEM. Scientific Reports.
    """
    # Validate required parameters
    if D_PD is None or A_v is None or r_PD is None:
        logging.error("Missing parameters: D_PD, A_v, and r_PD are required.")
        return 0

    # Calculate the number of base pairs per voxel
    bp_per_voxel = calculate_bp_per_voxel(voxel_size_nm=voxel_size_nm, **kwargs)

    try:
        # Effective domain size if the voxel is the minimum unit of the domain
        r_eff = r_PD / voxel_size_nm
        # Number of voxels in the domain based on the packing efficiency
        M_PD = A_v * (r_eff ** D_PD)
        # Convert domain size in voxels to base pairs
        N_PD = round(M_PD * bp_per_voxel)
        return N_PD
    except Exception as e:
        # Log any exceptions that occur during calculation
        logging.error("Error calculating genomic size of packing domain: %s", e)
        return 0

def compute_statistics_from_csv(file_path: str, **kwargs) -> dict:
    """
    Reads the CSV file and calculates the statistics.

    The CSV file is expected to have the following columns:
        - 'D': Scaling exponent of the packing domain.
        - 'CVC': Chromatin volume concentration.
        - 'A_v': Volume packing efficiency.
        - 'r_min': Radius of chromatin fiber in nanometers (nm).
        - 'r_max': Radius of packing domain in nanometers (nm).

    Parameters:
        - file_path (str): The path to the CSV file containing experimental data.
        - kwargs (dict): Additional keyword arguments for calculate_bp_per_voxel().

    Returns:
        dict: A dictionary containing the calculated statistics:
            - 'Average_D_PD' (float): Mean scaling exponent (D_PD) of packing domains.
            - 'Average_CVC_PD' (float): Mean chromatin volume concentration (CVC_PD) of 
                packing domains.
            - 'Average_A_v_PD' (float): Mean volume packing efficiency (A_v_PD) of packing 
                domains.
            - 'Average_N_PD_bp' (int): Mean genomic size (N_PD) of packing domains in base 
                pairs (bp).
            - 'Median_r_fiber_nm' (float): Median radius of chromatin fiber (r_fiber) in 
                nanometers (nm).
            - 'Median_r_PD_nm' (float): Median radius (r_PD) of packing domains in nanometers 
                (nm).
    """
    required_columns = {'D', 'CVC', 'A_v', 'r_min', 'r_max'}

    try:
        # Read the CSV file into a DataFrame
        PD_prop = pd.read_csv(file_path)
        
        # Check for required columns in the DataFrame
        missing_columns = required_columns - set(PD_prop.columns)
        if missing_columns:
            logging.error("Missing required columns: %s", ", ".join(missing_columns))
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Calculate the genomic size of each packing domain using provided parameters
        genomic_size_pd = calculate_genomic_size_of_PD(
            D_PD=PD_prop['D'],
            A_v=PD_prop['A_v'],
            r_PD=PD_prop['r_max'],
            **kwargs
        )
        
        # Compile the statistics into a dictionary
        PD_params = {
            'Average_D_PD': PD_prop['D'].mean(),  # Mean scaling exponent
            'Average_CVC_PD': PD_prop['CVC'].mean(),  # Mean chromatin volume concentration
            'Average_A_v_PD': PD_prop['A_v'].mean(),  # Mean volume packing efficiency
            'Average_N_PD_bp': int(genomic_size_pd.mean()),  # Mean genomic size in bp
            'Median_r_fiber_nm': PD_prop['r_min'].median(),  # Median radius of chromatin fiber
            'Median_r_PD_nm': PD_prop['r_max'].median()  # Median radius of packing domains
        }

        return PD_params
    except FileNotFoundError as error:
        logging.error("The CSV file '%s' was not found. Error: %s", file_path, error)
        raise
    except pd.errors.EmptyDataError as error:
        logging.error("The CSV file '%s' is empty. Error: %s", file_path, error)
        raise
    except pd.errors.ParserError as error:
        logging.error("Error parsing the CSV file '%s'. Error: %s", file_path, error)
        raise
    except PermissionError as error:
        logging.error("Permission denied when accessing the CSV file '%s'. Error: %s", 
                      file_path, error)
        raise
    except Exception as error:
        logging.error("An unexpected error occurred while processing the CSV file '%s'. Error: %s", 
                      file_path, error)
        raise

def get_default_HCT116_PD_values() -> dict:
    """
    Returns default values for HCT116 cells.

    Returns:
        dict: Default packing domain parameters for HCT116 cells.
            - 'Average_D_PD' (float): 2.6
            - 'Average_CVC_PD' (float): 0.275
            - 'Average_A_v_PD' (float): 0.6
            - 'Average_N_PD_bp' (int): 380000
            - 'Median_r_fiber_nm' (float): 10.0
            - 'Median_r_PD_nm' (float): 110.0
    """
    return {
        'Average_D_PD': 2.6,  # Default mean scaling exponent
        'Average_CVC_PD': 0.275,  # Default mean chromatin volume concentration
        'Average_A_v_PD': 0.6,  # Default mean volume packing efficiency
        'Average_N_PD_bp': 380000,  # Default mean genomic size in base pairs
        'Median_r_fiber_nm': 10.0,  # Default median radius of chromatin fiber in nm
        'Median_r_PD_nm': 110.0  # Default median radius of packing domains in nm
    }

def initialize_PD_parameters(file_path: str = None, **kwargs) -> dict:
    """
    Initializes packing domain parameters based on experimental data from a CSV file.
    Returns default values if the file does not exist or file_path is not provided.

    Parameters:
        - file_path (str, optional): Path to the CSV file containing experimental data. Defaults 
            to None.
        - kwargs (dict): Additional keyword arguments for calculate_bp_per_voxel().

    Returns:
        dict: A dictionary containing the calculated statistics:
            - 'Average_D_PD' (float): Mean scaling exponent (D_PD) of packing domains.
            - 'Average_CVC_PD' (float): Mean chromatin volume concentration (CVC_PD) of 
                packing domains.
            - 'Average_A_v_PD' (float): Mean volume packing efficiency (A_v_PD) of packing 
                domains.
            - 'Average_N_PD_bp' (int): Mean genomic size (N_PD) of packing domains in base 
                pairs (bp).
            - 'Median_r_fiber_nm' (float): Median radius of chromatin fiber (r_fiber) in 
                nanometers (nm).
            - 'Median_r_PD_nm' (float): Median radius (r_PD) of packing domains in nanometers 
                (nm).
    """
    if file_path and os.path.isfile(file_path):
        # CSV file exists, compute statistics from the file
        logging.info("CSV file '%s' found. Computing statistics from the file.", file_path)
        return compute_statistics_from_csv(file_path, **kwargs)
    else:
        if file_path:
            # File path provided but file not found, use default values
            logging.warning("CSV file '%s' not found. Using default values from HCT116 PDs.", 
                            file_path)
        else:
            # File path not provided, use default values
            logging.info("Using default values from HCT116 PDs.")
        return get_default_HCT116_PD_values()
