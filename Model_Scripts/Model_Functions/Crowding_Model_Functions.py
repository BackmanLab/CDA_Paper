#!/usr/bin/env python3

"""
This module simulates the effects of macromolecular crowding on gene
transcription and mRNA synthesis by adapting the Macromolecular Crowding
(MC) model published by H. Matsuda et al. (Biophysical Journal, 2014)
and A.R. Shim et al. (Biophysical Journal, 2020).

The code was adapted from the software published by L.M. Almassalha et al.
(Nature Biomedical Engineering, 2017) and updated by R.K.A. Virk et al.
(Science Advances, 2020).

Authors:
    Originally created by Wenli Wu and edited by Ranya K. A. Virk and 
    Jane Frederick (Backman Lab, Northwestern University).
    Last updated by Jane Frederick on 2025-03-12.

Functions:
    - calculate_3d_to_1d_diffusion_conversion_factors:
        Calculates conversion factors from 3D to 1D diffusion rates.
    - calculate_1d_diffusion_rates:
        Calculates the 1D diffusion rates for transcription factors and
        RNA polymerase II.
    - calculate_binding_rate_constants:
        Calculates the rate constants for transcription factor (TF) and
        RNA polymerase (RNAP) binding.
    - calculate_promoter_dissociation_rates:
        Calculates the dissociation rates for transcription factors and
        RNA polymerase.
    - calculate_phi_influenced_free_energies:
        Calculates phi-influenced free energies for transcription factors
        and RNA polymerase.
    - calculate_phi_dependent_parameters:
        Calculates phi-dependent parameters for transcription factors and
        RNA polymerase.
    - transcription_reactions:
        Defines the system of equations for the macromolecular crowding
        model.
    - solve_transcription_equations:
        Solves the system of equations to determine mRNA concentration.
    - simulate_mRNA_synthesis_under_crowding:
        Simulates mRNA dynamics influenced by transcription factors and
        RNA polymerase.
    - find_optimal_transcription_rate:
        Uses binary search to find the optimal transcription rate matching
        target phi_in.
    - run_crowding_model_simulation:
        Initializes constants based on mode, runs the simulation, and
        saves the results in csv files.
    - load_data_from_text_file:
        Loads numerical data from a text file, handling NaN values.
    - load_wenli_crowding_data:
        Loads crowding model outputs from Wenli's data.
    - load_saved_data:
        Loads saved crowding model outputs from files.
    - load_mc_model_outputs:
        Retrieves crowding model outputs from Wenli's data or saved
        files.

Usage Example:
    This module can be used to simulate the effects of macromolecular
    crowding on gene transcription and mRNA synthesis:
    ```
    results = run_crowding_model_simulation()
    print(results)
    ```
"""

import logging
import os

import numpy as np
import scipy.constants as const
import sympy as sp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CELL_VOLUME_UM3 = 500        # Volume of a typical HeLa cell in cubic micrometers
DNA_LENGTH_BP = 6e9          # Number of DNA base pairs in a diploid human cell before S phase
DNA_LENGTH_M = 1             # One-half of the total length of genomic DNA in meters
NUCLEAR_EXPORT_RATE = 8e-4   # Nuclear export rate of mRNA in s⁻¹ (gamma)
MRNA_DEGRADATION_RATE = 3e-4 # mRNA degradation rate in s⁻¹ (nu)

# Constants for DNA and molecules
BASE_PAIR_LENGTH_NM = 0.34 # Length of one base pair in nanometers
DNA_TURN_BP = 10           # Number of base pairs per turn of DNA
DNA_RADIUS_NM = 1          # Radius of the DNA molecule in nanometers
TF_RADIUS_NM = 4           # Radius of the sphere for transcription factors in nanometers
RNAP_RADIUS_NM = 5.4       # Radius of the sphere for RNA polymerase II in nanometers

# 3D Diffusion rates
TF_3D_DIFFUSION_RATE = 8.7e-11   # TF 3D diffusion rate (for a 4 nm sphere) in m²/s
RNAP_3D_DIFFUSION_RATE = 6.4e-11 # RNAP 3D diffusion rate (for a 5.4 nm sphere) in m²/s

# Conversion factors for 3D to 1D diffusion rates
TF_DIFFUSION_CONVERSION_FACTOR = 157.3
RNAP_DIFFUSION_CONVERSION_FACTOR = 270.4

# Coefficients for cubic fit to f(phi) from Brownian Dynamics simulations
TF_COEFFICIENTS = (-2.83, 3.87, -4.11)   # Alpha, Beta, Gamma for TF (for 4 nm sphere)
RNAP_COEFFICIENTS = (-3.89, 7.72, -7.72) # Alpha, Beta, Gamma for RNAP (for 5.4 nm sphere)

# Conversion constants
UM3_TO_M3 = 1e-18 # 1 cubic micrometer (um³) is 1e-18 cubic meters (m³)
UM3_TO_L = 1e-15  # 1 cubic micrometer (um³) is 1e-15 liters (L)

# Convert cell volume to SI units
cell_volume_m3 = CELL_VOLUME_UM3 * UM3_TO_M3 # Convert cell volume to cubic meters
cell_volume_l = CELL_VOLUME_UM3 * UM3_TO_L   # Volume in liters

# Calculate initial DNA concentration
dna_moles = DNA_LENGTH_BP / const.Avogadro # Number of moles of base pairs in a diploid cell
total_dna_concentration_M = dna_moles / cell_volume_l # Total concentration of DNA in a cell in M
total_dna_concentration_mM = round(total_dna_concentration_M, 3) * 1e3 # Total DNA concentration in mM

# Association rate constants and dissociation constants
k_ns_t = 4.9e4                   # Nonspecific TF-DNA binding association rate constant in mM⁻¹s⁻¹
k_ns_f = 3.6e4                   # Nonspecific RNAP-DNA binding association rate constant in mM⁻¹s⁻¹
K_ns_D_TF_0 = 1                  # Nonspecific TF-DNA binding dissociation constant in mM
K_ns_D_RNAP_0 = 1                # Nonspecific RNAP-DNA binding dissociation constant in mM
k_ns_o = k_ns_t * K_ns_D_TF_0    # Nonspecific TF-DNA dissociation rate in s⁻¹
k_ns_b = k_ns_f * K_ns_D_RNAP_0  # Nonspecific RNAP-DNA dissociation rate in s⁻¹
K_D_TF = 1                       # Dissociation constant for transcription factors in nM
K_D_RNAP = 1                     # Dissociation constant for RNA polymerase II in nM

# Define default parameters used for the 2014 and 2020 papers
default_params = {"mc": {"k_m": 0.02, "total_reactant_concentrations": np.array([0.3, 3, 30])},
                  "cpmc": {"k_m": 0.001, "total_reactant_concentrations": np.logspace(1, 4, 30)}}

# Generate the phi array
MINIMUM_PHI = 0
MAXIMUM_PHI = 0.505
phi_vector = np.linspace(MINIMUM_PHI, MAXIMUM_PHI, num=201, endpoint=False)

def calculate_3d_to_1d_diffusion_conversion_factors(
    dna_base_pairs_per_turn: int = DNA_TURN_BP,
    bp_length_nm: float = BASE_PAIR_LENGTH_NM,
    dna_radius_nm: float = DNA_RADIUS_NM,
    tf_radius_nm: float = TF_RADIUS_NM,
    rnap_radius_nm: float = RNAP_RADIUS_NM
) -> tuple:
    """
    Calculate the conversion factors for 3D to 1D diffusion rates for transcription factors
    and RNA polymerase II.

    Parameters:
        - dna_base_pairs_per_turn (int, optional): Number of base pairs per DNA helix 
            turn. Defaults to DNA_TURN_BP (10 bp).
        - bp_length_nm (float, optional): Length of one base pair in nanometers (nm).
            Defaults to BASE_PAIR_LENGTH_NM (0.34 nm).
        - dna_radius_nm (float, optional): Radius of the DNA molecule in nanometers (nm).
            Defaults to DNA_RADIUS_NM (1 nm).
        - tf_radius_nm (float, optional): Radius of the transcription factor sphere in 
            nanometers (nm). Defaults to TF_RADIUS_NM (4 nm).
        - rnap_radius_nm (float, optional): Radius of the RNA polymerase II sphere in 
            nanometers (nm). Defaults to RNAP_RADIUS_NM (5.4 nm).

    Returns:
        tuple: A tuple containing:
            - tf_conv_factor (float): Conversion factor for transcription factors 
                (unitless).
            - rnap_conv_factor (float): Conversion factor for RNA polymerase II 
                (unitless).

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
    """
    # Pitch of the DNA helix
    helix_pitch = dna_base_pairs_per_turn * bp_length_nm

    # Wave number of the DNA helix squared
    squared_wave_number = ((2 * np.pi) / helix_pitch) ** 2

    # Distance from the DNA center to the protein surface squared
    squared_tf_dist = (dna_radius_nm + tf_radius_nm) ** 2
    squared_rnap_dist = (dna_radius_nm + rnap_radius_nm) ** 2

    # Cross-sectional distance of proteins
    tf_area = (4 / 3) * (tf_radius_nm ** 2)
    rnap_area = (4 / 3) * (rnap_radius_nm ** 2)

    # Calculate conversion factor for proteins from 3D to 1D diffusion
    tf_conv_factor = 1 + squared_wave_number * (tf_area + squared_tf_dist)
    rnap_conv_factor = 1 + squared_wave_number * (rnap_area + squared_rnap_dist)

    return tf_conv_factor, rnap_conv_factor

def calculate_1d_diffusion_rates(
    tf_3d_diffusion_rate: float = TF_3D_DIFFUSION_RATE,
    rnap_3d_diffusion_rate: float = RNAP_3D_DIFFUSION_RATE,
    calculate_conversion_factors: bool = False,
    tf_conversion_factor: float = TF_DIFFUSION_CONVERSION_FACTOR,
    rnap_conversion_factor: float = RNAP_DIFFUSION_CONVERSION_FACTOR,
    **kwargs
) -> tuple:
    """
    Calculate the 1D diffusion rates for transcription factors and RNA polymerase II.
    This function uses global constants by default or recalculates conversion factors
    when specified.

    Parameters:
        - tf_3d_diffusion_rate (float, optional): The 3D diffusion rate of 
            transcription factors. Defaults to TF_3D_DIFFUSION_RATE (8.7e-11 m²/s).
        - rnap_3d_diffusion_rate (float, optional): The 3D diffusion rate of RNA 
            polymerase II. Defaults to RNAP_3D_DIFFUSION_RATE (6.4e-11 m²/s).
        - calculate_conversion_factors (bool, optional): Flag to recalculate 
            conversion factors. Defaults to False.
        - tf_conversion_factor (float, optional): Conversion factor for transcription 
            factors. Defaults to TF_DIFFUSION_CONVERSION_FACTOR (157.3).
        - rnap_conversion_factor (float, optional): Conversion factor for RNA 
            polymerase II. Defaults to RNAP_DIFFUSION_CONVERSION_FACTOR (270.4).
        - kwargs (dict): Additional keyword arguments for 
            calculate_3d_to_1d_diffusion_conversion_factors().

    Returns:
        tuple: A tuple containing:
            - tf_1d_diff_rate (float): 1D diffusion rate for transcription factors (m²/s).
            - rnap_1d_diff_rate (float): 1D diffusion rate for RNA polymerase II (m²/s).

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
    """
    if calculate_conversion_factors:
        # Recalculate conversion factors if specified
        tf_conv_fac, rnap_conv_fac = calculate_3d_to_1d_diffusion_conversion_factors(**kwargs)
    else:
        # Use provided or default conversion factors
        tf_conv_fac = tf_conversion_factor
        rnap_conv_fac = rnap_conversion_factor

    # Calculate the 1D diffusion rates by dividing the 3D diffusion rates by the conversion factors
    tf_1d_diff_rate = tf_3d_diffusion_rate / tf_conv_fac
    rnap_1d_diff_rate = rnap_3d_diffusion_rate / rnap_conv_fac

    return tf_1d_diff_rate, rnap_1d_diff_rate

def calculate_binding_rate_constants(
    k_ns_o: float = k_ns_o,
    k_ns_b: float = k_ns_b,
    cell_volume_m3: float = cell_volume_m3,
    dna_length_m: float = DNA_LENGTH_M,
    **kwargs
) -> tuple:
    """
    Calculate the rate constants for transcription factor and RNA polymerase II binding.

    Parameters:
        - k_ns_o (float, optional): Nonspecific TF-DNA dissociation rate in s⁻¹.
            Defaults to k_ns_o (4.9e4 s⁻¹).
        - k_ns_b (float, optional): Nonspecific RNAP-DNA dissociation rate in s⁻¹.
            Defaults to k_ns_b (3.6e4 s⁻¹).
        - cell_volume_m3 (float, optional): Cell volume in cubic meters (m³).
            Defaults to cell_volume_m3 (5e-16 m³).
        - dna_length_m (float, optional): Total length of genomic DNA in meters (m).
            Defaults to DNA_LENGTH_M (1 m).
        - kwargs (dict): Additional keyword arguments for calculate_1d_diffusion_rates().

    Returns:
        tuple: A tuple containing:
            - k_t (float): TF-promoter association rate constant in nM⁻¹s⁻¹.
            - k_f (float): RNAP-complex association rate constant in nM⁻¹s⁻¹.

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
    """
    # Calculate 1D diffusion rates for TF and RNAP
    tf_1d_diff_rate, rnap_1d_diff_rate = calculate_1d_diffusion_rates(**kwargs)

    # Calculate sqrt of diffusion rate * dissociation constant 
    sqrt_tf_diff_dissoc = np.sqrt(tf_1d_diff_rate * k_ns_o)
    sqrt_rnap_diff_dissoc = np.sqrt(rnap_1d_diff_rate * k_ns_b)

    # Calculate binding rate constants by scaling with cell volume and DNA length
    k_t_m = (cell_volume_m3 * sqrt_tf_diff_dissoc) / dna_length_m 
    k_f_m = (cell_volume_m3 * sqrt_rnap_diff_dissoc) / dna_length_m

    # Convert rate constants to nM⁻¹s⁻¹
    k_t = k_t_m * const.Avogadro * 1e-6
    k_f = k_f_m * const.Avogadro * 1e-6

    return k_t, k_f

def calculate_promoter_dissociation_rates(
    K_D_TF: float = K_D_TF,
    K_D_RNAP: float = K_D_RNAP,
    total_dna_concentration_mM: float = total_dna_concentration_mM,
    K_ns_D_TF_0: float = K_ns_D_TF_0,
    K_ns_D_RNAP_0: float = K_ns_D_RNAP_0,
    **kwargs
) -> tuple:
    """
    Calculate the promoter dissociation rates for transcription factors and RNA polymerase II.

    Parameters:
        - K_D_TF (float, optional): Dissociation constant for transcription factors (nM).
            Defaults to K_D_TF (1 nM).
        - K_D_RNAP (float, optional): Dissociation constant for RNA polymerase II (nM).
            Defaults to K_D_RNAP (1 nM).
        - total_dna_concentration_mM (float, optional): Total concentration of DNA (mM).
            Defaults to total_dna_concentration_mM (20.0 mM).
        - K_ns_D_TF_0 (float, optional): Nonspecific TF-DNA binding dissociation 
            constant (mM). Defaults to K_ns_D_TF_0 (1 mM).
        - K_ns_D_RNAP_0 (float, optional): Nonspecific RNAP-DNA binding dissociation 
            constant (mM). Defaults to K_ns_D_RNAP_0 (1 mM).
        - kwargs (dict): Additional keyword arguments for calculate_binding_rate_constants().

    Returns:
        tuple: A tuple containing:
            - k_o (float): TF-promoter dissociation rate in s⁻¹.
            - k_b (float): RNAP-promoter dissociation rate in s⁻¹.

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
    """
    # Calculate binding rate constants for TF and RNA polymerase
    k_t, k_f = calculate_binding_rate_constants(**kwargs)
    
    # Calculate protein-promoter dissociation rate
    k_o = total_dna_concentration_mM * (K_D_TF / K_ns_D_TF_0) * k_t
    k_b = total_dna_concentration_mM * (K_D_RNAP / K_ns_D_RNAP_0) * k_f

    return k_o, k_b

def calculate_phi_influenced_free_energies(
    phi: np.ndarray = phi_vector,
    tf_coefficients: tuple = TF_COEFFICIENTS,
    rnap_coefficients: tuple = RNAP_COEFFICIENTS,
    **kwargs
) -> tuple:
    """
    Calculate phi-influenced free energies for transcription factors and RNA polymerase.

    Parameters:
        - phi (np.ndarray, optional): Array of phi values. Defaults to phi_vector 
            (array from 0 to 0.505 with 201 elements).
        - tf_coefficients (tuple, optional): Coefficients for transcription factors
            (alpha, beta, gamma). Defaults to TF_COEFFICIENTS (-2.83, 3.87, -4.11).
        - rnap_coefficients (tuple, optional): Coefficients for RNA pol II
            (alpha, beta, gamma). Defaults to RNAP_COEFFICIENTS (-3.89, 7.72, -7.72).

    Returns:
        tuple: Contains the following phi-influenced free energies (in units of k_BT):
            - f_TF_phi (np.ndarray): Free energy for transcription factors.
            - f_RNAP_phi (np.ndarray): Free energy for RNA pol II.
            - f_cro_TF_phi (np.ndarray): Crowding free energy for transcription factors.
            - f_cro_RNAP_phi (np.ndarray): Crowding free energy for RNA pol II.
            - f_cro_RNAPs_phi (np.ndarray): Crowding free energy for sliding RNA pol II.
            - f_bar_TF_phi (np.ndarray): Barrier free energy for transcription factors.
            - f_bar_RNAP_phi (np.ndarray): Barrier free energy for RNA pol II.
            - f_bar_RNAPs_phi (np.ndarray): Barrier free energy for sliding RNA pol II.

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
    """
    # Unpack coefficients
    alpha_TF, beta_TF, gamma_TF = tf_coefficients
    alpha_RNAP, beta_RNAP, gamma_RNAP = rnap_coefficients

    # Calculate free energy for proteins based on phi
    f_TF_phi = 1 + alpha_TF * phi + beta_TF * phi ** 2 + gamma_TF * phi ** 3
    f_RNAP_phi = 1 + alpha_RNAP * phi + beta_RNAP * phi ** 2 + gamma_RNAP * phi ** 3

    # Calculate crowding free energies from simulations
    f_cro_TF_phi = -3.2 * phi + -2.0 * phi ** 2
    f_cro_RNAP_phi = -3.7 * phi + -2.7 * phi ** 2
    f_cro_RNAPs_phi = -2.6 * phi + -4.6 * phi ** 2

    # Calculate phi-influenced barrier free energies from simulations
    f_bar_TF_phi = 2.5 * phi ** 2
    f_bar_RNAP_phi = 3.1 * phi ** 2
    f_bar_RNAPs_phi = 0.1 * phi ** 2 + 9.2 * phi ** 3
    
    return (f_TF_phi, f_RNAP_phi, f_cro_TF_phi, f_cro_RNAP_phi, f_cro_RNAPs_phi, 
            f_bar_TF_phi, f_bar_RNAP_phi, f_bar_RNAPs_phi)

def calculate_phi_dependent_parameters(
    k_ns_t: float = k_ns_t,
    k_ns_f: float = k_ns_f,
    k_ns_o: float = k_ns_o,
    k_ns_b: float = k_ns_b,
    **kwargs
) -> tuple:
    """
    Calculate phi-dependent parameters for transcription factors and RNA polymerase.

    Parameters:
        - k_ns_t (float, optional): Nonspecific TF-DNA binding association rate 
            constant in mM⁻¹s⁻¹. Defaults to k_ns_t (4.9e4 mM⁻¹s⁻¹).
        - k_ns_f (float, optional): Nonspecific RNAP-DNA binding association rate 
            constant in mM⁻¹s⁻¹. Defaults to k_ns_f (3.6e4 mM⁻¹s⁻¹).
        - k_ns_o (float, optional): Nonspecific TF-DNA dissociation rate in s⁻¹.
            Defaults to k_ns_o (4.9e4 s⁻¹).
        - k_ns_b (float, optional): Nonspecific RNAP-DNA dissociation rate in s⁻¹.
            Defaults to k_ns_b (3.6e4 s⁻¹).
        - kwargs (dict): Additional keyword arguments for 
            calculate_binding_rate_constants() and 
            calculate_promoter_dissociation_rates().

    Returns:
        tuple: Contains the following phi-dependent parameters (all in units of s⁻¹):
            - k_ns_t_phi (np.ndarray): Nonspecific TF-DNA association rate constants.
            - k_ns_f_phi (np.ndarray): Nonspecific RNAP-DNA association rate constants.
            - k_ns_o_phi (np.ndarray): Nonspecific TF-DNA dissociation rates.
            - k_ns_b_phi (np.ndarray): Nonspecific RNAP-DNA dissociation rates.
            - k_t_phi (np.ndarray): TF-promoter association rate constants.
            - k_f_phi (np.ndarray): RNAP-complex association rate constants.
            - k_o_phi (np.ndarray): TF-promoter dissociation rates.
            - k_b_phi (np.ndarray): RNAP-promoter dissociation rates.

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
    """    
    # Calculate binding rate constants for TF and RNA polymerase
    k_t, k_f = calculate_binding_rate_constants(**kwargs)

    # Calculate the dissociation rates for TF and RNA polymerase
    k_o, k_b = calculate_promoter_dissociation_rates(**kwargs)

    # Calculate phi-influenced free energies
    (f_TF_phi, f_RNAP_phi, f_cro_TF_phi, f_cro_RNAP_phi, f_cro_RNAPs_phi, f_bar_TF_phi, 
     f_bar_RNAP_phi, f_bar_RNAPs_phi) = calculate_phi_influenced_free_energies(**kwargs)
    
    # Calculate nonspecific protein-DNA binding association rate constants influenced by phi
    k_ns_t_phi = k_ns_t * (f_TF_phi * np.exp(-f_bar_TF_phi))
    k_ns_f_phi = k_ns_f * (f_RNAP_phi * np.exp(-f_bar_RNAP_phi))

    # Calculate nonspecific protein-DNA dissociation rates influenced by phi
    k_ns_o_phi = k_ns_o * np.exp(f_cro_TF_phi) * (f_TF_phi * np.exp(-f_bar_TF_phi))
    k_ns_b_phi = k_ns_b * np.exp(f_cro_RNAP_phi) * (f_RNAP_phi * np.exp(-f_bar_RNAP_phi))

    # Calculate protein-promoter association rate constants influenced by phi
    k_t_phi = k_t * f_TF_phi * np.exp(f_cro_TF_phi / 2) * np.exp(-f_bar_TF_phi / 2)
    k_f_phi = (k_f * f_RNAP_phi * np.exp(f_cro_RNAP_phi / 2) * np.exp(-f_bar_RNAP_phi / 2)
               * np.exp(-f_bar_RNAPs_phi))

    # Calculate protein-promoter dissociation rates influenced by phi
    k_o_phi = k_o * f_TF_phi * np.exp(f_cro_TF_phi / 2) * np.exp(-f_bar_TF_phi / 2)
    k_b_phi = (k_b * np.exp(f_cro_RNAPs_phi) * f_RNAP_phi * np.exp(f_cro_RNAP_phi / 2)
               * np.exp(-f_bar_RNAPs_phi) * np.exp(-f_bar_RNAP_phi / 2))

    return (k_ns_t_phi, k_ns_f_phi, k_ns_o_phi, k_ns_b_phi, k_t_phi, k_f_phi, k_o_phi, k_b_phi)

def transcription_reactions(
    COMPLEX_I: float,
    COMPLEX_II: float,
    BINDING_SITE_CONC: float,
    BOUND_TF_CONC: float,
    BOUND_RNAP_CONC: float,
    k_ns_t_phi: float,
    k_ns_o_phi: float,
    k_ns_f_phi: float,
    k_ns_b_phi: float,
    k_t_phi: float,
    k_o_phi: float,
    k_f_phi: float,
    k_b_phi: float,
    k_m: float,
    total_tf_concentration_nM: float,
    total_rnap_concentration_nM: float,
    total_promoter_concentration_nM: float,
    total_dna_concentration_mM: float = total_dna_concentration_mM
) -> tuple:
    """
    Define the system of equations to solve for the macromolecular crowding model.

    Parameters:
        - COMPLEX_I (float): Concentration of intermediate complex I (nM).
        - COMPLEX_II (float): Concentration of intermediate complex II (nM).
        - BINDING_SITE_CONC (float): Concentration of promoters (nM).
        - BOUND_TF_CONC (float): Concentration of DNA bound transcription factor (nM).
        - BOUND_RNAP_CONC (float): Concentration of DNA bound RNA polymerase (nM).
        - k_ns_t_phi (float): Phi-influenced nonspecific TF-DNA binding association rate 
            constant (mM⁻¹s⁻¹).
        - k_ns_o_phi (float): Phi-influenced nonspecific TF-DNA dissociation rate 
            constant (mM⁻¹s⁻¹).
        - k_ns_f_phi (float): Phi-influenced nonspecific RNAP-DNA binding association rate 
            constant (s⁻¹).
        - k_ns_b_phi (float): Phi-influenced nonspecific RNAP-DNA dissociation rate 
            constant (s⁻¹).
        - k_t_phi (float): Phi-influenced TF-promoter association rate constant (nM⁻¹s⁻¹).
        - k_o_phi (float): Phi-influenced TF-promoter dissociation rate constant (nM⁻¹s⁻¹).
        - k_f_phi (float): Phi-influenced RNAP-promoter association rate constant (s⁻¹).
        - k_b_phi (float): Phi-influenced RNAP-promoter dissociation rate constant (s⁻¹).
        - k_m (float): Rate constant for mRNA synthesis (s⁻¹).
        - total_tf_concentration_nM (float): Total concentration of transcription factors (nM).
        - total_rnap_concentration_nM (float): Total concentration of RNA polymerase (nM).
        - total_promoter_concentration_nM (float): Total concentration of promoters (nM).
        - total_dna_concentration_mM (float, optional): Total concentration of DNA (mM).
            Defaults to total_dna_concentration_mM (20.0 mM).

    Returns:
        - tuple: A tuple containing the system of equations, each in units of nM.

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
        - A.R. Shim, et al. (2020). Dynamic Crowding Regulates Transcription. 
          Biophysical Journal.
    """
    # Define differential equations for the intermediate transcriptional complexes
    dCI_eq = (k_t_phi * BOUND_TF_CONC * BINDING_SITE_CONC - k_o_phi * COMPLEX_I 
              - k_f_phi * BOUND_RNAP_CONC * COMPLEX_I + k_b_phi * COMPLEX_II)

    dCII_eq = k_f_phi * BOUND_RNAP_CONC * COMPLEX_I - (k_b_phi + k_m) * COMPLEX_II

    # Define the equation for promoters
    O_eq = total_promoter_concentration_nM - COMPLEX_I - COMPLEX_II - BINDING_SITE_CONC

    # Define the equation for transcription factors bound to DNA
    TF_D_eq = (
        ((total_dna_concentration_mM * (k_ns_t_phi / k_ns_o_phi)) 
         * (total_tf_concentration_nM - COMPLEX_I - COMPLEX_II)
        - (k_m / k_ns_o_phi) * COMPLEX_II)
        / (1 + total_dna_concentration_mM * (k_ns_t_phi / k_ns_o_phi))
        - BOUND_TF_CONC
    )

    # Define the equation for RNA polymerase bound to DNA
    RNAP_D_eq = (
        ((total_dna_concentration_mM * (k_ns_f_phi / k_ns_b_phi)) 
         * (total_rnap_concentration_nM - COMPLEX_II)
        - (k_m / k_ns_b_phi) * COMPLEX_II)
        / (1 + total_dna_concentration_mM * (k_ns_f_phi / k_ns_b_phi))
        - BOUND_RNAP_CONC
    )

    return (dCI_eq, dCII_eq, O_eq, TF_D_eq, RNAP_D_eq)

def solve_transcription_equations(
    k_m: float,
    phi: np.ndarray = phi_vector,
    mrna_deg_rate: float = MRNA_DEGRADATION_RATE,
    **kwargs
) -> np.ndarray:
    """
    Solve the system of equations to determine mRNA concentration using sympy.nsolve.

    Parameters:
        - k_m (float): Rate constant for mRNA synthesis (s⁻¹).
        - phi (np.ndarray, optional): Array of phi values. Defaults to phi_vector 
            (array from 0 to 0.505 with 201 elements).
        - mrna_deg_rate (float, optional): mRNA degradation rate (s⁻¹). Defaults to
            MRNA_DEGRADATION_RATE (3e-4 s⁻¹).
        - kwargs (dict): Additional keyword arguments for 
            calculate_phi_dependent_parameters().

    Returns:
        - np.ndarray: Cytoplasmic mRNA concentration for each phi value (nM).

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
        - A.R. Shim, et al. (2020). Dynamic Crowding Regulates Transcription. 
          Biophysical Journal.
    """
    # Define symbols for the variables in the system of equations
    (COMPLEX_I, COMPLEX_II, BINDING_SITE_CONC, BOUND_TF_CONC, 
     BOUND_RNAP_CONC) = sp.symbols('COMPLEX_I, COMPLEX_II, BINDING_SITE_CONC, BOUND_TF_CONC, BOUND_RNAP_CONC')
    
    # Calculate phi-dependent parameters
    (k_ns_t_phi, k_ns_f_phi, k_ns_o_phi, k_ns_b_phi, k_t_phi, k_f_phi, k_o_phi,
     k_b_phi) = calculate_phi_dependent_parameters(**kwargs)

    # Initial guess for the solver variables
    initial_guess = (1e-6, 1e-6, 1e-6, 1e-6, 1e-6)
    
    # Initialize array to store mRNA concentrations for each phi value
    mRNA_cyto = np.empty(len(phi))

    # Iterate through each phi value to solve the system of equations
    for i, phi_val in enumerate(phi):
        try:
            # Solve the system of transcription reactions for the current phi value
            equation_solution = sp.nsolve(
                transcription_reactions(
                    COMPLEX_I=COMPLEX_I,
                    COMPLEX_II=COMPLEX_II, 
                    BINDING_SITE_CONC=BINDING_SITE_CONC,
                    BOUND_TF_CONC=BOUND_TF_CONC, 
                    BOUND_RNAP_CONC=BOUND_RNAP_CONC,
                    k_ns_t_phi=k_ns_t_phi[i],
                    k_ns_o_phi=k_ns_o_phi[i],
                    k_ns_f_phi=k_ns_f_phi[i],
                    k_ns_b_phi=k_ns_b_phi[i],
                    k_t_phi=k_t_phi[i],
                    k_o_phi=k_o_phi[i],
                    k_f_phi=k_f_phi[i],
                    k_b_phi=k_b_phi[i],
                    k_m=k_m,
                    **kwargs
                ),
                (COMPLEX_I, COMPLEX_II, BINDING_SITE_CONC, BOUND_TF_CONC, BOUND_RNAP_CONC),
                initial_guess
            )
            # Extract the concentration of COMPLEX_II from the solution
            complex_ii_conc = float(equation_solution[1])
            # Calculate mRNA concentration based on the solution
            mRNA_cyto[i] = k_m / mrna_deg_rate * complex_ii_conc
        except Exception as e:
            # Log an error message if solving fails and assign NaN to indicate failure
            logging.error(f"Error solving equations for phi={phi_val} (index {i}): {e}")
            mRNA_cyto[i] = np.nan  # Assign NaN to indicate failure for this phi value
    
    return mRNA_cyto

def simulate_mRNA_synthesis_under_crowding(
    total_reactant_concentrations: np.ndarray,
    phi: np.ndarray = phi_vector,
    **kwargs
) -> tuple:
    """
    Simulate transcription.

    Parameters:
        - total_reactant_concentrations (np.ndarray): Total concentrations of transcription 
            factors, RNA polymerase II, and promoters (nM).
        - phi (np.ndarray, optional): Array of phi values. Defaults to phi_vector (array 
            from 0 to 0.505 with 201 elements).
        - kwargs (dict): Additional keyword arguments for solve_transcription_equations().

    Returns:
        tuple: A tuple containing the following elements:
            - mRNA_expression_profiles (np.ndarray): mRNA expression profiles for all conditions 
              across phi values (nM).
            - max_mRNA_concentrations (np.ndarray): Maximum mRNA concentration at steady state for 
              each condition (nM).
            - mRNA_peak_phi_locations (np.ndarray): Phi value at maximum mRNA concentration for each 
              condition (unitless).
            - mRNA_peak_second_derivatives (np.ndarray): Second derivative of the normalized mRNA 
              expression curve at the peak for each condition (unitless).

    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
        - A.R. Shim, et al. (2020). Dynamic Crowding Regulates Transcription. 
          Biophysical Journal.
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin 
          packing regulates phenotypic plasticity. Science Advances.
    """
    # Log start of simulation
    logging.info("Starting simulation...")

    # Find number of different concentration conditions
    num_conditions = len(total_reactant_concentrations)
    # Initialize mRNA expression profiles array to store simulation results
    mRNA_expression_profiles = np.zeros((num_conditions, len(phi)))
    # Initialize array for max mRNA concentrations to store results
    max_mRNA_concentrations = np.zeros(num_conditions)
    # Initialize array for phi peak locations to store results
    mRNA_peak_phi_locations = np.zeros(num_conditions)
    # Initialize mRNA expression second derivative array to store results
    mRNA_peak_second_derivatives = np.zeros(num_conditions)

    # Iterate through each concentration condition
    for idx, conc in enumerate(total_reactant_concentrations):
        # Log current condition
        logging.info(f"Processing condition {idx + 1}/{num_conditions} ({conc:.2f} nM)")

        try:
            # Solve transcription equations for the current condition
            mRNA_cyto = solve_transcription_equations(
                total_tf_concentration_nM=conc, 
                total_rnap_concentration_nM=conc, 
                total_promoter_concentration_nM=conc,
                **kwargs
            )
        except Exception as e:
            # Log error and assign NaN if solving fails
            logging.error(f"Failed to solve equations for condition {idx + 1}: {e}")
            mRNA_expression_profiles[idx, :] = np.nan
            max_mRNA_concentrations[idx] = np.nan
            mRNA_peak_phi_locations[idx] = np.nan
            mRNA_peak_second_derivatives[idx] = np.nan
            continue  # Skip to the next condition

        # Store mRNA full expression profile as a function of phi
        mRNA_expression_profiles[idx, :] = mRNA_cyto
        # Find index of maximum mRNA concentration
        max_idx = np.nanargmax(mRNA_cyto)
        # Store maximum mRNA concentration
        max_mRNA_concentrations[idx] = mRNA_cyto[max_idx]
        # Store phi value at peak mRNA concentration
        mRNA_peak_phi_locations[idx] = phi[max_idx]

        # Normalize mRNA concentration by maximum value
        mRNA_cyto_normalized = mRNA_cyto / mRNA_cyto[max_idx]
        # Calculate first derivative (rate of change) of mRNA production using numpy's gradient
        first_deriv = np.gradient(mRNA_cyto_normalized, phi)
        # Calculate second derivative (acceleration) of mRNA production using numpy's gradient
        second_deriv = np.gradient(first_deriv, phi)
        # Find the second derivative at the maximum mRNA concentration
        mRNA_peak_second_derivatives[idx] = second_deriv[max_idx]

    # Log completion of simulation
    logging.info("Simulation completed.")
    return mRNA_expression_profiles, max_mRNA_concentrations, mRNA_peak_phi_locations, mRNA_peak_second_derivatives

def find_optimal_transcription_rate(
    target_phi_in: float,
    error_margin: float = 0.001,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    k_m_bounds: tuple = (0.001, 0.021),
    test_concentrations: np.ndarray = np.array([0.3, 3, 30]),
    **kwargs
) -> float:
    """
    Uses a binary search algorithm to find the optimal transcription rate where the 
    mean phi_in is within the specified error margin of the target_phi_in.

    Parameters:
        - target_phi_in (float): The desired target value for phi_in (unitless).
        - error_margin (float, optional): Acceptable deviation from the target_phi_in. 
            Defaults to 0.001.
        - max_iterations (int, optional): Maximum number of iterations for the search. 
            Defaults to 20.
        - tolerance (float, optional): Tolerance for convergence. Defaults to 1e-6.
        - k_m_bounds (tuple, optional): Lower and upper bounds for k_m search. 
            Defaults to (0.001, 0.021).
        - test_concentrations (np.ndarray, optional): Total concentrations 
            of transcription factors, RNA polymerase II, and promoters (nM). 
            Defaults to np.array([0.3, 3, 30]).
        - kwargs (dict): Additional keyword arguments for 
            simulate_mRNA_synthesis_under_crowding().

    Returns:
        - float: Optimal k_m value within the error margin, or None if not found.
    """
    # Initialize lower and upper bounds for k_m
    lower, upper = k_m_bounds
    previous_diff = None  # Initialize previous difference
    previous_k_m = None   # Initialize previous k_m value

    # Iterate up to the maximum number of iterations
    for iteration in range(1, max_iterations + 1):
        # Calculate the midpoint of the current bounds
        k_m_mid = (lower + upper) / 2

        # Run the simulation with the current k_m value
        _, _, phi_peaks, _ = simulate_mRNA_synthesis_under_crowding(
            k_m=round(k_m_mid, 6),
            total_reactant_concentrations=test_concentrations,
            **kwargs
        )

        # Compute the mean phi value from the simulation results
        mean_phi = np.mean(phi_peaks)
        # Compute the standard deviation of the phi value
        std_phi = np.std(phi_peaks)
        # Calculate the difference between the mean phi and the target
        current_diff = mean_phi - target_phi_in
        # Log the current iteration details
        logging.info(
            f"Completed iteration {iteration}/{max_iterations}: k_m = {k_m_mid:.6f}, "
            f"phi = {mean_phi:.6f} ± {std_phi:.6f}, diff = {current_diff:.6f}"
        )

        # Check if the previous difference is the same as the current difference
        if previous_diff is not None and abs(current_diff - previous_diff) < tolerance:
            logging.info(
                "Difference did not change from previous iteration. "
                "Returning average of previous k_m and current k_m."
            )
            average_k_m = (previous_k_m + k_m_mid) / 2
            return average_k_m  # Return the average of the two k_m values

        # Check if the current difference is within the acceptable error margin
        if abs(current_diff) <= error_margin:
            logging.info(
                f"Optimal k_m found: {k_m_mid:.6f} with phi = {mean_phi:.6f} ± {std_phi:.6f}"
            )
            return k_m_mid  # Return the optimal k_m value

        # Adjust the search bounds based on whether the mean phi is too high or too low
        if current_diff > 0:
            lower = k_m_mid  # Increase k_m to decrease mean_phi
        else:
            upper = k_m_mid  # Decrease k_m to increase mean_phi

        previous_diff = current_diff      # Update previous difference
        previous_k_m = k_m_mid            # Update previous k_m value

        # Check if the bounds have converged within the specified tolerance
        if upper - lower < tolerance:
            logging.warning("Search converged within tolerance but did not meet error margin.")
            return k_m_mid  # Return the current k_m_mid as the best approximation

    # If the optimal k_m was not found within the maximum iterations, log a warning
    logging.warning("Optimal k_m not found within the specified error margin and iterations.")
    return None  # Indicate that no optimal k_m was found

def run_crowding_model_simulation(
    mode: str = None,
    save_directory: str = "MC_Model_Outputs/",
    k_m: float = None,
    total_reactant_concentrations: np.ndarray = None,
    **kwargs
) -> tuple:
    """
    Initialize macromolecular crowding model constants based on the specified mode or 
    input values and run the simulation.
    
    Parameters:
        - mode (str, optional): Specify "mc" for Matsuda et al. 2014 or "cpmc" for 
            Virk et al. 2020 constants. Defaults to None to allow input from user.
        - save_directory (str, optional): Directory to save the output files. Defaults 
            to "MC_Model_Outputs/".
        - k_m (float, optional): Transcription rate of Pol-II in s⁻¹. Defaults to 0.02 for 'mc' 
            mode and 0.001 for 'cpmc' mode.
        - total_reactant_concentrations (np.ndarray, optional): Array of total concentrations 
            [TF], [RNAP], and [O] in nM. If None, it uses default based on the mode.
        - kwargs (dict): Additional keyword arguments for simulation.

    Returns:
        - tuple: Results from simulate_mRNA_synthesis_under_crowding
            (mRNA_expression_profiles, max_mRNA_concentrations, mRNA_peak_phi_locations, 
            mRNA_peak_second_derivatives)
               
    References:
        - H. Matsuda, et al. (2014). Macromolecular Crowding as a Regulator of Gene
          Transcription. Biophysical Journal.
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin 
          packing regulates phenotypic plasticity. Science Advances.
    """
    # Retrieve parameters based on mode or user input
    if mode is None:
        # Validate that essential parameters are provided
        if k_m is None or total_reactant_concentrations is None:
            raise ValueError(
                "k_m and total_reactant_concentrations must be provided if mode is not specified."
            )
    elif mode.lower() in default_params:
        # Update parameters with default values based on the specified mode ('mc' or 'cpmc')
        k_m = default_params[mode.lower()]['k_m']
        total_reactant_concentrations = default_params[mode.lower()]['total_reactant_concentrations']
    else:
        # Raise an error if an invalid mode is provided
        raise ValueError("Invalid mode. Mode must be 'mc' or 'cpmc'.")
    
    # Ensure the save directory exists; create it if it does not
    os.makedirs(save_directory, exist_ok=True)
    
    # Run the mRNA synthesis simulation with the specified parameters
    results = simulate_mRNA_synthesis_under_crowding(
        k_m=k_m,
        total_reactant_concentrations=total_reactant_concentrations,
        **kwargs
    )

    # Create a filename suffix based on concentration range and transcription rate
    start_conc, end_conc = total_reactant_concentrations[[0, -1]]
    suffix = (
        f"{start_conc:.1f}nM_to_{end_conc:.1f}nM_km_{k_m:.4f}s-1"
    )

    np.savetxt(
        os.path.join(save_directory, f"total_reactant_concentrations_{suffix}.csv"), 
        total_reactant_concentrations, 
        delimiter=',', 
        header='', 
        comments=''
    )

    # Define result filenames and corresponding data arrays
    result_names = [
        "mRNA_expression_profiles", 
        "max_mRNA_concentrations", 
        "mRNA_peak_phi_locations", 
        "mRNA_peak_second_derivatives"
    ]
    result_files = [
        os.path.join(save_directory, f"{name}_{suffix}.csv") 
        for name in result_names
    ]
    
    # Save each simulation result to a CSV file with appropriate headers
    for file_path, data in zip(result_files, results):
        np.savetxt(
            file_path, 
            data, 
            delimiter=',', 
            header='', 
            comments=''
        )

    return results

def load_data_from_text_file(
        file_path: str, 
        delimiter: str = ',', 
        skip_header: int = 0
        ) -> np.ndarray:
    """
    Load numerical data from a text file and handle potential NaN values.
    
    Parameters:
        - file_path (str): Path to the file. The file should be a text file with 
          numerical data, where each row represents a data point and columns are 
          separated by the specified delimiter.
        - delimiter (str, optional): Delimiter used in the file. Defaults to ','.
        - skip_header (int, optional): Number of lines to skip at the beginning 
          of the file. Defaults to 0.

    Returns:
        - np.ndarray: A numpy array containing the data loaded from the file 
          with NaN values filled with 0.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Attempt to load the data from the file
        loaded_data = np.genfromtxt(file_path, delimiter=delimiter, 
                                    skip_header=skip_header)
    except Exception as e:
        # Log an error message if there is an issue reading the file
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise
    
    # Validate loaded data
    if loaded_data.size == 0:
        logging.warning(f"No data found in {file_path}. Returning an empty array.")
        return loaded_data
    
    # Check if the loaded data contains any NaN values
    nan_mask = np.isnan(loaded_data)
    if nan_mask.any():
        logging.warning(f"Data from {file_path} contains NaN values. Filling NaN values with 0.")
        # Replace NaN values with 0
        loaded_data[nan_mask] = 0.0
    
    return loaded_data

def load_wenli_crowding_data(wenli_file_path: str, phi_in: float) -> tuple:
    """
    Load macromolecular crowding model outputs from Wenli's data.
    
    Parameters:
        wenli_file_path (str): Directory containing Wenli's data files.
        phi_in (float): Experimental phi_in value to match against Wenli's data.

    Returns:
        dict: A dictionary containing the following elements:
            - total_reactant_concentrations (np.ndarray): Total concentrations of 
              transcription factors and RNA polymerase (nM).
            - mRNA_expression_profiles (None): mRNA concentrations, which are not 
              included in Wenli's data.
            - max_mRNA_concentrations (np.ndarray): Maximum mRNA concentration at 
              steady state for each condition (nM).
            - mRNA_peak_phi_locations (np.ndarray): Phi value at maximum mRNA concentration 
              for each condition (unitless).
            - mRNA_peak_second_derivatives (np.ndarray): Second derivative of the normalized 
              mRNA expression curve at the peak for each condition (unitless).
    
    Raises:
        ValueError: If phi_in is not provided.
        FileNotFoundError: If required files are missing.

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin 
          packing regulates phenotypic plasticity. Science Advances.
    """
    required_files = {
        "total_reactant_concentration": 'tot_con_fix_phi_cpmc_A549_1e-5_1e-2.txt',
        "mRNA_peak_phi_locations": 'phi_initial_fix_phi_cpmc_A549_1e-5_1e-2.txt',
        "max_mRNA_concentrations": 
            'max_mRNA_initial_fix_phi_cpmc_A549_1e-5_1e-2.txt',
        "mRNA_peak_second_derivatives": 
            'second_derivative_TF_norm_fix_phi_cpmc_A549_1e-5_1e-2.txt',
    }
    
    # Construct full file paths by joining the directory path with filenames
    file_paths = {
        key: os.path.join(wenli_file_path, fname) 
        for key, fname in required_files.items()
    }
    
    # Check if all required files exist in the specified directory
    missing_files = [
        fname for key, fname in required_files.items() 
        if not os.path.exists(file_paths[key])
    ]
    if missing_files:
        # Raise an error if any required files are missing
        raise FileNotFoundError(
            f"Missing required files: {', '.join(missing_files)} in {wenli_file_path}"
        )
    
    # Load data from each required file using the previously defined loader function
    data = {
        key: load_data_from_text_file(path, delimiter = None) 
        for key, path in file_paths.items()
    }
    
    # Extract individual datasets from the loaded data
    tot_conc = data.get("total_reactant_concentration")
    all_phi_peaks_locs = data.get("mRNA_peak_phi_locations")
    all_max_mRNA_concs = data.get("max_mRNA_concentrations")
    all_mRNA_second_derivs = data.get("mRNA_peak_second_derivatives")
    
    # Ensure that all necessary data was successfully loaded
    if (
        tot_conc is None or 
        all_phi_peaks_locs is None or 
        all_max_mRNA_concs is None or 
        all_mRNA_second_derivs is None
    ):
        raise ValueError(
            "One or more required data files could not be loaded properly."
        )
    
    # Find the index of the phi value closest to the provided phi_in
    index_phi_in = np.argmin(np.abs(all_phi_peaks_locs - phi_in))
    
    # Retrieve the phi peak location corresponding to the closest index
    phi_peak_locs = all_phi_peaks_locs[index_phi_in]
    
    # Extract the maximum mRNA concentrations and second derivatives at the identified index
    max_mRNA_concs = all_max_mRNA_concs[index_phi_in, :]
    mRNA_second_derivs = all_mRNA_second_derivs[index_phi_in, :]

    # Convert concentrations to nM
    tot_conc *= 1e6
    max_mRNA_concs *= 1e6

    # Compile the results into a dictionary
    result = {
        'total_reactant_concentrations_nM': tot_conc,
        'mRNA_expression_profiles_nM': None,  # Not available from Wenli's data
        'max_mRNA_concentrations_nM': max_mRNA_concs,
        'mRNA_peak_phi_locations': phi_peak_locs,
        'mRNA_peak_second_derivatives': mRNA_second_derivs
    }

    return result

def load_saved_data(file_path: str) -> dict:
    """
    Load macromolecular crowding model outputs from saved files.
    
    Parameters:
        file_path (str): Directory containing the data files.

    Returns:
        dict: A dictionary containing the following elements:
            - total_reactant_concentrations (np.ndarray): Total concentrations of 
                transcription factors and RNA polymerase (nM).
            - mRNA_expression_profiles (np.ndarray): mRNA expression profiles for 
                all conditions across phi values (nM).
            - max_mRNA_concentrations (np.ndarray): Maximum mRNA concentration at 
                steady state for each condition (nM).
            - mRNA_peak_phi_locations (np.ndarray): Phi value at maximum mRNA concentration 
                for each condition (unitless).
            - mRNA_peak_second_derivatives (np.ndarray): Second derivative of the normalized
                mRNA expression curve at the peak for each condition (unitless).
    """
    # Define patterns to match required data files.
    required_file_patterns = [
        'total_reactant_concentrations', 
        'mRNA_expression_profiles', 
        'max_mRNA_concentrations', 
        'mRNA_peak_phi_locations', 
        'mRNA_peak_second_derivatives'
    ]
    
    # List all files in the specified directory.
    all_files = os.listdir(file_path)
    
    # Filter files matching each pattern.
    matched_files = {
        pattern: [os.path.join(file_path, f) for f in all_files if f.startswith(pattern)] 
        for pattern in required_file_patterns
    }
    
    # Identify any patterns that did not match any files.
    missing_files = [
        pattern for pattern, files in matched_files.items() 
        if not files
    ]
    if missing_files:
        # Log a warning and raise an error if any required files are missing.
        logging.warning(
            f"No files matching the following patterns were found in {file_path}: "
            f"{', '.join(missing_files)}."
        )
        raise FileNotFoundError(
            f"No files matching patterns: {', '.join(missing_files)} found in "
            f"{file_path}"
        )

    try:
        # Load data from each matched file using the data loader function.
        tot_conc = load_data_from_text_file(
            matched_files['total_reactant_concentrations'][0]
        )
        mRNA_expr_profiles = load_data_from_text_file(
            matched_files['mRNA_expression_profiles'][0]
        )
        max_mRNA_concs = load_data_from_text_file(
            matched_files['max_mRNA_concentrations'][0]
        )
        phi_peak_locs = load_data_from_text_file(
            matched_files['mRNA_peak_phi_locations'][0]
        )
        mRNA_second_derivs = load_data_from_text_file(
            matched_files['mRNA_peak_second_derivatives'][0]
        )
    except IndexError as e:
        # Log an error if there is an issue accessing the matched files.
        logging.error(f"Error accessing matched files: {e}")
        raise
    except Exception as e:
        # Log any other errors that occur during data loading.
        logging.error(f"Error loading data from files: {e}")
        raise
    
    # Compile the loaded data and regression result into a dictionary.
    result = {
        'total_reactant_concentrations_nM': tot_conc,
        'mRNA_expression_profiles_nM': mRNA_expr_profiles,
        'max_mRNA_concentrations_nM': max_mRNA_concs,
        'mRNA_peak_phi_locations': phi_peak_locs,
        'mRNA_peak_second_derivatives': mRNA_second_derivs
    }

    return result

def load_mc_model_outputs(
    source: str = "saved",
    file_path: str = "Saved_Model_Outputs/Crowding_Model_Outputs/",
    phi_in: float = None,
    wenli_file_path: str = "Saved_Model_Outputs/Wu_2017_MC_Model_Outputs/"
) -> dict:
    """
    Retrieve macromolecular crowding model outputs from either Wenli's data or saved files.
    
    Parameters:
        source (str, optional): Data source ('wenli' or 'saved'). Defaults to 'saved'.
        file_path (str, optional): Directory containing the saved data files.
            Defaults to 'Saved_Model_Outputs/Crowding_Model_Outputs/'.
        phi_in (float, optional): Experimental phi_in value to match against Wenli's
            data. Required if source is 'wenli'. Defaults to None.
        wenli_file_path (str, optional): Directory containing Wenli's data files.
            Defaults to 'Saved_Model_Outputs/Wu_2017_MC_Model_Outputs/'.
        
    Returns:
        dict: A dictionary containing the following elements:
            - total_reactant_concentrations (np.ndarray): Total concentrations of
              transcription factors and RNA polymerase (nM).
            - mRNA_conc (np.ndarray or None): mRNA concentrations (nM), or None if
              source is 'wenli'.
            - max_mRNA_concentrations (np.ndarray): Maximum mRNA concentration at
              steady state for each condition (nM).
            - mRNA_peak_phi_locations (np.ndarray): Phi value at maximum mRNA concentration
              for each condition (unitless).
            - mRNA_peak_second_derivatives (np.ndarray): Second derivative of the normalized
              mRNA expression curve for each condition (unitless).
    
    Raises:
        ValueError: If source is invalid or if phi_in is not provided when source is
            'wenli'.
        FileNotFoundError: If necessary files are not found.
    """
    # Convert the source string to lowercase to ensure case-insensitive matching
    source = source.lower()
    
    if source == "wenli":
        # Check if phi_in is provided when the source is 'wenli'
        if phi_in is None:
            logging.error("phi_in must be provided when source is 'wenli'.")
            raise ValueError("phi_in must be provided when source is 'wenli'.")
        # Load crowding data from Wenli's dataset
        return load_wenli_crowding_data(wenli_file_path, phi_in)
    elif source == "saved":
        # Load previously saved crowding data
        return load_saved_data(file_path)
    else:
        # Log an error for invalid source input and raise an exception
        logging.error(
            f"Invalid source: {source}. Choose either 'wenli' or 'saved'."
        )
        raise ValueError("Invalid source. Choose either 'wenli' or 'saved'.")