#!/usr/bin/env python3

"""
This module adapts the Chromatin Packing Macromolecular Crowding (CPMC) model 
published by L.M. Almassalha et al. (Nature Biomedical Engineering, 2017) 
and updated by R.K.A. Virk et al. (Science Advances, 2020).

The code was adapted from the software published by R.K.A. Virk et al. 
(Science Advances, 2020).

Authors:
    Originally created by Wenli Wu and Ranya K. A. Virk and edited by Jane 
    Frederick (Backman Lab, Northwestern University).
    Last updated by Jane Frederick on 2025-03-12.

Functions:
    - calculate_phi_interaction_volume:
        Calculates the phi of the interaction volume, assuming mobile crowders fill 
        the remaining space not occupied by chromatin.
    - calculate_interaction_volume_radius:
        Calculates the interaction volume radius in nanometers using the volume 
        packing efficiency.
    - calculate_interaction_volume_variance_phi:
        Calculates the variance of crowding within an interaction volume.
    - calculate_bar_epsilon:
        Calculates the average expression (bar_epsilon) using either the exact method or 
        an approximation based on a Taylor series expansion.
    - calculate_kappa:
        Calculates kappa (the critical expression rate beyond which crowding has little 
        effect) for values where the Taylor Series approximation is valid.
    - initialize_normalized_expression_rate_vector:
        Initializes normalized expression vectors for sensitivity analysis.
    - calculate_G_bar_epsilon:
        Calculates the change of variables function G(bar_epsilon).
    - calculate_phi_c:
        Calculates the chromatin volume fraction (phi_c) based on the scaling exponent (D),
        volume packing efficiency (A_v), and genomic size (N_PD).
    - calculate_Se_D:
        Calculates the sensitivity of the expression rate to the scaling exponent (D).
    - calculate_Se_N_PD:
        Calculates the sensitivity of the expression rate to the genomic size (N_PD).
    - calculate_Se_A_v:
        Calculates the sensitivity of the expression rate to the volume packing efficiency (A_v).

Usage Example:
    This module can be used to calculate the sensitivity of the expression rate to the scaling exponent (D):
    ```
    D = 2.6
    sensitivity_D = calculate_Se_D(D)
    print(sensitivity_D)
    ```
"""

import logging
import numpy as np
from scipy.optimize import curve_fit
import Model_Functions.Crowding_Model_Functions as mcfunc
import Model_Functions.ChromSTEM_PD_Functions as pdfunc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
# Average gene length in base pairs
AVERAGE_GENE_LENGTH_BP = 6000
# Interaction volume radius in nm for one base pair
BASEPAIR_INTERACTION_VOL_RADIUS_NM = 15
# Radius of a base pair in nm
BASEPAIR_RADIUS_NM = 1  
# Volume fraction of mobile crowders in an interaction volume
PHI_MOBILE_CROWDERS = 0.05  
# Range of normalized expression rates (fold difference from average)
NORMALIZED_EXPRESSION_RANGE = (-2, 3)  

# Get packing domain parameters
PD_OUT = pdfunc.initialize_PD_parameters()
# Mean scaling exponent (D)
AVERAGE_D_PD = PD_OUT['Average_D_PD']
# Mean chromatin volume concentration (CVC)
AVERAGE_CVC_PD = PD_OUT['Average_CVC_PD']
# Mean volume packing efficiency (A_v)
AVERAGE_A_V_PD = PD_OUT['Average_A_v_PD']
# Mean genomic size (N_d) in base pairs
AVERAGE_N_PD_BP = PD_OUT['Average_N_PD_bp']
# Median radius of chromatin fiber (r_fiber) in nm
MEDIAN_R_FIBER_NM = PD_OUT['Median_r_fiber_nm']
# Median radius of packing domains (r_PD) in nm
MEDIAN_R_PD_NM = PD_OUT['Median_r_PD_nm']

# Get crowding model outputs as vectors of input reactant concentrations
MC_OUT = mcfunc.load_mc_model_outputs(source="saved", file_path="Saved_Model_Outputs/Crowding_Model_Outputs/")
# Input vector of total transcriptional reactant concentrations in nM
REACTANT_CONCENTRATIONS = MC_OUT['total_reactant_concentrations_nM']
# Maximum mRNA concentration in nM
MRNA_PEAK_CONCENTRATIONS = MC_OUT['max_mRNA_concentrations_nM']
# Convert concentration to expression rate
MRNA_PEAK_EXPRESSION_RATES = MRNA_PEAK_CONCENTRATIONS * mcfunc.MRNA_DEGRADATION_RATE
# Second derivative of the mRNA expression curve (unitless)
SECOND_DERIVATIVE_AT_MRNA_PEAKS = MC_OUT['mRNA_peak_second_derivatives']

def calculate_phi_interaction_volume(
    phi_chromatin: float = AVERAGE_CVC_PD,
    phi_mobile_crowders: float = PHI_MOBILE_CROWDERS
) -> float:
    """
    Calculate the phi of the interaction volume, assuming mobile crowders fill 
    the remaining space not occupied by chromatin.

    Based on new equation defined in Vadim's derivations.
    
    Parameters:
        - phi_chromatin (float, optional): Chromatin volume fraction (unitless). 
            Defaults to AVERAGE_CVC_PD (0.275).
        - phi_mobile_crowders (float, optional): Volume fraction of mobile 
            crowders (unitless). Defaults to PHI_MOBILE_CROWDERS (0.05).
    
    Returns:
        - float: The total volume fraction of the interaction volume (unitless).
    """
    total_volume_fraction = phi_chromatin + ((1 - phi_chromatin) * phi_mobile_crowders)
    return total_volume_fraction

def calculate_interaction_volume_radius(
    D: float,
    A_v: float = AVERAGE_A_V_PD,
    gene_length: float = AVERAGE_GENE_LENGTH_BP,
    r_min_in: float = BASEPAIR_INTERACTION_VOL_RADIUS_NM,
    r_min: float = BASEPAIR_RADIUS_NM
) -> float:
    """
    Calculate the interaction volume radius in nanometers using the volume 
    packing efficiency.

    Based on new equation defined in Vadim's derivations.
    
    Parameters:
        - D (float): Scaling of the nuclear chromatin ACF (unitless).
        - A_v (float, optional): Mean volume packing efficiency of packing 
            domains (unitless). Defaults to AVERAGE_A_V_PD (0.6).
        - gene_length (float, optional): Gene length in base pairs (bp).
            Defaults to AVERAGE_GENE_LENGTH_BP (6000 bp).
        - r_min_in (float, optional): Minimum interaction volume radius (nm).
            Defaults to BASEPAIR_INTERACTION_VOL_RADIUS_NM (15 nm).
        - r_min (float, optional): Minimum radius of a base pair (nm).
            Defaults to BASEPAIR_RADIUS_NM (1 nm).

    Returns:
        - float: The calculated interaction volume radius in nanometers (nm).

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin 
          packing regulates phenotypic plasticity. Science Advances.
    """
    interaction_volume_radius = r_min_in + ((gene_length / A_v) ** (1 / D)) * r_min
    return interaction_volume_radius

def calculate_interaction_volume_variance_phi(
    D: float,
    r_min: float = BASEPAIR_RADIUS_NM,
    phi_chromatin: float = AVERAGE_CVC_PD,
    phi_mobile_crowders: float = PHI_MOBILE_CROWDERS,
    **kwargs
) -> float:
    """
    Calculate the variance of crowding within an interaction volume.

    Based on new equation defined in Vadim's derivations.

    Parameters:
        - D (float): Scaling of the nuclear chromatin ACF (unitless).
        - r_min (float, optional): Minimum radius of a base pair (nm).
            Defaults to BASEPAIR_RADIUS_NM (1 nm).
        - phi_chromatin (float, optional): Chromatin volume fraction (unitless). 
            Defaults to AVERAGE_CVC_PD (0.275).
        - phi_mobile_crowders (float, optional): Volume fraction of mobile 
            crowders (unitless). Defaults to PHI_MOBILE_CROWDERS (0.05).
        - kwargs (dict): Additional keyword arguments for 
            calculate_interaction_volume_radius().

    Returns:
        - float: The variance of crowding in the interaction volume (unitless).

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin 
          packing regulates phenotypic plasticity. Science Advances.
    """
    # Calculate the interaction volume radius using the previously defined function
    r_in = calculate_interaction_volume_radius(D=D, r_min=r_min, **kwargs)
    # Initial variance based on chromatin and mobile crowders
    variance_o = phi_chromatin * (1 - phi_chromatin) * ((1 - phi_mobile_crowders) ** 2)
    # Calculate the variance of crowding within the interaction volume
    var_phi_in = ((r_min / r_in) ** (3 - D)) * variance_o
    return var_phi_in

def calculate_bar_epsilon(
    mRNA_peak_expression_rates: np.ndarray,
    mRNA_peak_second_derivatives: np.ndarray = None,
    kappa: float = None,
    D: float = AVERAGE_D_PD,
    **kwargs
) -> np.ndarray:
    """
    Calculate the average expression (bar_epsilon) using either the exact method or 
    an approximation based on a Taylor series expansion.

    Parameters:
        - mRNA_peak_expression_rates (np.ndarray): Maximum mRNA expression rate at 
            steady state for each reactant concentration (nM/s).
        - mRNA_peak_second_derivatives (np.ndarray, optional): Second derivative of the 
            normalized mRNA expression curve at the peak for each reactant concentration 
            (unitless). If not provided, the approximation method will be used.
        - kappa (float, optional): The threshold value used to calculate bar_epsilon. 
            Defaults to KAPPA.
        - D (float, optional): Scaling of the nuclear chromatin ACF (unitless). Defaults to
            AVERAGE_D_PD (2.6).
        - kwargs (dict): Additional keyword arguments for 
            calculate_interaction_volume_variance_phi().

    Returns:
        - np.ndarray: The calculated average expression (bar_epsilon).

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin 
          packing regulates phenotypic plasticity. Science Advances.
    """
    # Calculate the variance of phi_in
    var_phi_in = calculate_interaction_volume_variance_phi(D=D, **kwargs)
    
    if mRNA_peak_second_derivatives is not None:
        # Calculate exact value of bar_epsilon using the second derivative
        bar_epsilon = mRNA_peak_expression_rates + (0.5 * var_phi_in * mRNA_peak_second_derivatives)
    else:
        # Approximate the second derivative of the mRNA expression curve
        approx_second_deriv = -np.sqrt(kappa / mRNA_peak_expression_rates)
        # Calculate bar_epsilon using the kappa approximation for the second derivative
        bar_epsilon = mRNA_peak_expression_rates + (0.5 * var_phi_in * approx_second_deriv)
    
    return bar_epsilon

def calculate_kappa(
    reactant_concentrations: np.ndarray = REACTANT_CONCENTRATIONS,
    mRNA_peak_expression_rates: np.ndarray = MRNA_PEAK_EXPRESSION_RATES,
    mRNA_peak_second_derivatives: np.ndarray = SECOND_DERIVATIVE_AT_MRNA_PEAKS,
    **kwargs
) -> float:
    """
    Calculate kappa (the critical expression rate beyond which crowding has little 
    effect) for values where the Taylor Series approximation is valid (output expression 
    is positive). The minimum reactant concentration where the Taylor series expansion
    is valid is also calculated.

    Parameters:
        - reactant_concentrations (np.ndarray): Total transcription factors, RNA 
            polymerase, and promoters (nM). Defaults to REACTANT_CONCENTRATIONS.
        - mRNA_peak_expression_rates (np.ndarray): Maximum mRNA expression rate at 
            steady state for each reactant concentration (nM/s). Defaults to 
            MRNA_PEAK_EXPRESSION_RATES.
        - mRNA_peak_second_derivatives (np.ndarray): Second derivative of the normalized 
            mRNA expression curve at the peak for each reactant concentration (unitless).
            Defaults to SECOND_DERIVATIVE_AT_MRNA_PEAKS.

    Returns:
        - float: Calculated kappa in nM/s.
        - float: Average bar_epsilon value using positive values only in nM/s.
        - float: Minimum input reactant concentration where the Taylor series expansion is valid in nM.

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
    """
    try:
        exact_bar_epsilon = calculate_bar_epsilon(
            mRNA_peak_expression_rates=mRNA_peak_expression_rates,
            mRNA_peak_second_derivatives=mRNA_peak_second_derivatives,
            **kwargs
        )

        # Filter out negative values of exact_bar_epsilon
        positive_indices = np.where(exact_bar_epsilon > 0)
        bar_eps_pos = exact_bar_epsilon[positive_indices]
        react_pos = reactant_concentrations[positive_indices]
        epsilon_pos = mRNA_peak_expression_rates[positive_indices]
        second_deriv_pos = mRNA_peak_second_derivatives[positive_indices]

        # Define the fitting function for kappa
        kappa_fitting_func = lambda epsilon, sec_deriv: sec_deriv / np.sqrt(epsilon)
        # Perform curve fitting to estimate kappa
        popt, pcov = curve_fit(kappa_fitting_func, epsilon_pos, second_deriv_pos)
        # Extract the kappa value from the curve fitting
        kappa = popt[0] ** 2

        # Calculate the average bar_epsilon using positive values only
        average_bar_epsilon = np.mean(bar_eps_pos)
        # Find the minimum reactant concentration where the Taylor series expansion is valid
        min_reactant_concentration = react_pos[0]

        return kappa, average_bar_epsilon, min_reactant_concentration
    except Exception as e:
        logging.error(f"Error in calculate_kappa: {e}")
        raise

def initialize_normalized_expression_rate_vector(
    expression_range: tuple = NORMALIZED_EXPRESSION_RANGE
) -> tuple:
    """
    Initialize normalized expression vectors for sensitivity analysis.

    This function generates a linear space of logarithmic normalized expression rates
    and computes their exponential values.

    Parameters:
        - expression_range (tuple, optional): Range of normalized expression rates (fold difference from average).
            Defaults to NORMALIZED_EXPRESSION_RANGE (-2, 3).

    Returns:
        - tuple: 
            - np.ndarray: Array of logarithmic normalized expression rates.
            - np.ndarray: Array of normalized expression rates.

    References:
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin packing  
          regulates phenotypic plasticity. Science Advances, 6, eaax6232. 
    """
    # Retrieve parameters from arguments or use default values
    E_low, E_high = expression_range
    # Relative expression rate is ln(E/<E>)
    # Sensitivity curve range (x-axis)
    ln_norm_E_vec = np.linspace(E_low, E_high, 100)
    # Compute normalized expression rates
    norm_E_vec = np.exp(ln_norm_E_vec)
    return ln_norm_E_vec, norm_E_vec

# Calculate the kappa value and average bar_epsilon
kappa, average_bar_epsilon, min_reactant_concentration = calculate_kappa()
# Initialize normalized expression rate vectors
ln_norm_E_vec, norm_E_vec = initialize_normalized_expression_rate_vector()
# Scale the average bar_epsilon with the normalized expression rate vector
E_vec = average_bar_epsilon * norm_E_vec

def calculate_G_bar_epsilon(
    D: float,
    E_vec: np.ndarray = E_vec,
    kappa: float = kappa,
    **kwargs
) -> float:
    """
    Calculate the change of variables function G(bar_epsilon).
    
    Parameters:
        - D (float): Scaling of the nuclear chromatin ACF (unitless).
        - E_vec (np.ndarray, optional): Normalized expression rate vector. Defaults to E_vec.
        - kappa (float, optional): Kappa value from crowding model. Defaults to kappa.
        - kwargs (dict): Additional keyword arguments for calculate_interaction_volume_variance_phi().
    
    Returns:
        - float: Result of the G(bar_epsilon) function.
        
    References:
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin packing  
          regulates phenotypic plasticity. Science Advances, 6, eaax6232. 
    """
    try:
        # Calculate the variance of phi_in
        var_phi_in = calculate_interaction_volume_variance_phi(D=D, **kwargs)
        # Square the variance of phi_in
        var_squared = var_phi_in ** 2
        # Calculate the dimensionless term
        dimensionless_term = E_vec / kappa
        # Calculate the square root term
        sqrt_term = 1 + np.sqrt(1 + (16 * dimensionless_term) / var_squared)
        # Calculate G_bar_epsilon using the derived formula
        G_bar_epsilon = (1 / (8 * dimensionless_term)) * var_squared * sqrt_term
        return G_bar_epsilon
    except Exception as e:
        logging.error(f"Error in calculate_G_bar_epsilon: {e}")
        raise

def calculate_phi_c(
    D: float,
    A_v: float,
    N_PD: int
) -> float:
    """
    Calculate the chromatin volume fraction (phi_c) based on the scaling exponent (D),
    volume packing efficiency (A_v), and genomic size (N_PD).

    Parameters:
        - D (float): Scaling of the nuclear chromatin ACF (unitless).
        - A_v (float, optional): Mean volume packing efficiency of packing domains (unitless).
        - N_PD (int, optional): Genomic size in base pairs.

    Returns:
        - float: The calculated chromatin volume fraction (phi_c).

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via 
          modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin 
          packing regulates phenotypic plasticity. Science Advances.
    """
    phi_c = A_v * (N_PD / A_v) ** (1 - 3 / D)
    return phi_c

def calculate_Se_D(
    D: float,
    E_vec: np.ndarray = E_vec,
    kappa: float = kappa,
    A_v: float = AVERAGE_A_V_PD,
    N_PD: int = AVERAGE_N_PD_BP,
    r_min: float = MEDIAN_R_FIBER_NM,
    gene_length: float = AVERAGE_GENE_LENGTH_BP,
    **kwargs
) -> np.ndarray:
    """
    Calculate the sensitivity of the expression rate to the scaling exponent (D).

    Parameters:
        - D (float): Scaling of the nuclear chromatin ACF (unitless).
        - E_vec (np.ndarray, optional): Normalized expression rate vector. Defaults to E_vec.
        - kappa (float, optional): Kappa value from crowding model. Defaults to kappa.
        - A_v (float, optional): Mean volume packing efficiency of packing domains (unitless). Defaults to AVERAGE_A_V_PD (0.6).
        - N_PD (int, optional): Genomic size in base pairs. Defaults to AVERAGE_N_PD_BP (380000 bp).
        - r_min (float, optional): Minimum radius of a base pair (nm). Defaults to MEDIAN_R_FIBER_NM (10 nm).
        - gene_length (float, optional): Gene length in base pairs (bp). Defaults to AVERAGE_GENE_LENGTH_BP (6000 bp).
        - kwargs (dict): Additional keyword arguments for calculate_interaction_volume_radius() and calculate_G_bar_epsilon().

    Returns:
        - np.ndarray: The sensitivity of the expression rate to the scaling exponent (D).

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin packing regulates phenotypic plasticity. Science Advances.
    """
    try:
        # Calculate the interaction volume radius
        r_in = calculate_interaction_volume_radius(D=D, r_min=r_min, gene_length=gene_length, A_v=A_v)
        # Calculate the values of the change of variables function G(bar_epsilon)
        G_bar_epsilon = calculate_G_bar_epsilon(D=D, E_vec=E_vec, kappa=kappa, **kwargs)
        # Calculate the chromatin volume fraction (phi_c)
        phi_c = calculate_phi_c(D=D, A_v=A_v, N_PD=N_PD)
        
        # Calculate the derivative of the variance in phi within the packing domain
        d_var_o = (3 / D) * np.log(N_PD / A_v) * ((1 - 2 * phi_c) / (1 - phi_c))
        # Calculate the derivative of the variance of phi_in
        d_var_phi_in = D * np.log(r_in / r_min) + ((3 - D) / D) * (r_min / r_in) * (
            (gene_length / A_v) ** (1 / D)) * np.log(gene_length / A_v) + d_var_o
        # Calculate the derivative of the exposure ratio
        d_ER = (1 / D) * np.log(N_PD / A_v)
        return d_ER + -G_bar_epsilon * d_var_phi_in
    except Exception as e:
        logging.error(f"Error in calculate_Se_D: {e}")
        raise

def calculate_Se_N_PD(
    N_PD: int,
    E_vec: np.ndarray = E_vec,
    kappa: float = kappa,
    D: float = AVERAGE_D_PD,
    A_v: float = AVERAGE_A_V_PD,
    **kwargs
) -> np.ndarray:
    """
    Calculate the sensitivity of the expression rate to the genomic size (N_PD).

    Parameters:
        - N_PD (int): Genomic size in base pairs.
        - E_vec (np.ndarray, optional): Normalized expression rate vector. Defaults to E_vec.
        - kappa (float, optional): Kappa value from crowding model. Defaults to kappa.
        - D (float, optional): Scaling of the nuclear chromatin ACF (unitless). Defaults to AVERAGE_D_PD (2.6).
        - A_v (float, optional): Mean volume packing efficiency of packing domains (unitless). Defaults to AVERAGE_A_V_PD (0.6).
        - kwargs (dict): Additional keyword arguments for calculate_G_bar_epsilon() and calculate_phi_c().

    Returns:
        - np.ndarray: The sensitivity of the expression rate to the genomic size (N_PD).

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin packing regulates phenotypic plasticity. Science Advances.
    """
    try:
        # Calculate the values of the change of variables function G(bar_epsilon)
        G_bar_epsilon = calculate_G_bar_epsilon(D=D, E_vec=E_vec, kappa=kappa, **kwargs)
        # Calculate the chromatin volume fraction (phi_c)
        phi_c = calculate_phi_c(D=D, A_v=A_v, N_PD=N_PD)
        
        # Calculate the derivative of the variance of phi_in
        d_var_phi_in = (1 - (3 / D)) * ((1 - 2 * phi_c) / (1 - phi_c))
        # Calculate the derivative of the exposure ratio
        d_ER = -(1 / D)
        return d_ER + -G_bar_epsilon * d_var_phi_in
    except Exception as e:
        logging.error(f"Error in calculate_Se_N_PD: {e}")
        raise

def calculate_Se_A_v(
    A_v: float,
    E_vec: np.ndarray = E_vec,
    kappa: float = kappa,
    D: float = AVERAGE_D_PD,
    N_PD: int = AVERAGE_N_PD_BP,
    r_min: float = MEDIAN_R_FIBER_NM,
    gene_length: float = AVERAGE_GENE_LENGTH_BP,
    **kwargs
) -> np.ndarray:
    """
    Calculate the sensitivity of the expression rate to the volume packing efficiency (A_v).

    Parameters:
        - A_v (float): Mean volume packing efficiency of packing domains (unitless).
        - E_vec (np.ndarray, optional): Normalized expression rate vector. Defaults to E_vec.
        - kappa (float, optional): Kappa value from crowding model. Defaults to kappa.
        - D (float, optional): Scaling of the nuclear chromatin ACF (unitless). Defaults to AVERAGE_D_PD (2.6).
        - N_PD (int, optional): Genomic size in base pairs. Defaults to AVERAGE_N_PD_BP (380000 bp).
        - r_min (float, optional): Minimum radius of a base pair (nm). Defaults to MEDIAN_R_FIBER_NM (10 nm).
        - gene_length (float, optional): Gene length in base pairs (bp). Defaults to AVERAGE_GENE_LENGTH_BP (6000 bp).
        - kwargs (dict): Additional keyword arguments for calculate_interaction_volume_radius() and calculate_G_bar_epsilon().

    Returns:
        - np.ndarray: The sensitivity of the expression rate to the volume packing efficiency (A_v).

    References:
        - L.M. Almassalha, G.M. Bauer, W. Wu, et al. (2017). Macrogenomic engineering via modulation of the scaling of chromatin packing density. Nature Biomedical Engineering.
        - R.K.A. Virk, W. Wu, L.M. Almassalha, et al. (2020). Disordered chromatin packing regulates phenotypic plasticity. Science Advances.
    """
    try:
        # Calculate the interaction volume radius
        r_in = calculate_interaction_volume_radius(D=D, A_v=A_v, r_min=r_min, gene_length=gene_length, **kwargs)
        # Calculate the values of the change of variables function G(bar_epsilon)
        G_bar_epsilon = calculate_G_bar_epsilon(D=D, E_vec=E_vec, kappa=kappa, **kwargs)
        # Calculate the chromatin volume fraction (phi_c)
        phi_c = calculate_phi_c(D=D, A_v=A_v, N_PD=N_PD)
        
        # Calculate the derivative of the variance of phi_in
        d_var_phi_in = ((3 / D) - 1) * (((1 - 2 * phi_c) / (1 - phi_c)) + (A_v / gene_length) * (r_min / r_in))
        # Calculate the derivative of the exposure ratio
        d_ER = (1 / D)
        return d_ER + -G_bar_epsilon * d_var_phi_in
    except Exception as e:
        logging.error(f"Error in calculate_Se_A_v: {e}")
        raise