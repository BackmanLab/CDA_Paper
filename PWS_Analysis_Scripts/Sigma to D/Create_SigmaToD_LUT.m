% Create_SigmaToD_LUT.m
% Description: Generates a Lookup Table (LUT) for converting Sigma values to D values.
% Author: Jane Frederick
% Last Updated: December 10, 2024

% Clear all variables from the workspace to prevent potential conflicts or errors
clear variables

%% User Inputs

% ALL USERS MUST CHANGE THIS
% Define the file path to the folder containing the sigma to D conversion scripts
functionpath = 'C:\Users\janef\OneDrive - Northwestern University\Documents - Backman Lab - Jane Frederick\Jane Frederick\Papers\CDA Paper\V3 (2021-2024)\Scripts\Sigma to D\PWS-SigmaToD';
addpath(genpath(functionpath)); % Add the specified path and its subfolders to MATLAB's search path

%% Inputs for Constants

% Define constants related to the imaged cells
CVC = 0.275; % Crowding volume fraction for HCT116 cells
N_PD = 381000; % Genomic size of a packing domain in base pairs (bp) for HCT116
thickness = 2; % Cell thickness in micrometers

% Define various constants related to the LCPWS systems
RI = 1.337; % Refractive index of the cell culture media
NAi_LCPWS1 = 0.54; % Illumination numerical aperture for LCPWS1 system
NAi_LCPWS2 = 0.52; % Illumination numerical aperture for LCPWS2 system, measured using NA calculator or hexagon size
NAc_LCPWS1 = 1.4; % Collection numerical aperture for LCPWS1 system (63x objective)
NAc_LCPWS2 = 1.49; % Collection numerical aperture for LCPWS2 system (100x objective)
lambda = 585; % Central imaging wavelength in nanometers
oil_objective = true; % Indicates if oil immersion objective is used
cell_glass = true; % Indicates if imaging was done where the cell contacts the dish

%% Determine Relevant Microscope Parameters

% Calculate the refractive index using the Gladstone-Dale equation
liveCellRI = S2D.RIDefinition.createFromGladstoneDale(RI, CVC);

% Create system configuration for LCPWS1 microscope
LCPWS1Sys = S2D.SystemConfiguration(liveCellRI, NAi_LCPWS1, NAc_LCPWS1, ...
    lambda, oil_objective, cell_glass);

% Create system configuration for LCPWS2 microscope
LCPWS2Sys = S2D.SystemConfiguration(liveCellRI, NAi_LCPWS2, NAc_LCPWS2, ...
    lambda, oil_objective, cell_glass);

%% Create the Sigma to D LUT

% Define range of sigma values
sigma = linspace(0,0.497);

% Perform conversion for LCPWS1 system
[D_B_PWS1, D_PWS1, Nf_PWS1, lmax_PWS1] = SigmaToD_AllInputs( ...
    sigma, LCPWS1Sys, N_PD, thickness);

% Perform conversion for LCPWS2 system
[D_B_PWS2, D_PWS2, Nf_PWS2, lmax_PWS2] = SigmaToD_AllInputs( ...
    sigma, LCPWS2Sys, N_PD, thickness);

% Combine conversion results into LUT for LCPWS1
convlut_LCPWS1 = [sigma; D_B_PWS1; D_PWS1; Nf_PWS1; lmax_PWS1];

% Combine conversion results into LUT for LCPWS2
convlut_LCPWS2 = [sigma; D_B_PWS2; D_PWS2; Nf_PWS2; lmax_PWS2];

% Create table for LCPWS1 conversion results with appropriate column names
conversion_LCPWS1 = array2table(convlut_LCPWS1', 'VariableNames', ...
    {'Sigma', 'D_ACF', 'D', 'Nf', 'lmax'});

% Create table for LCPWS2 conversion results with appropriate column names
conversion_LCPWS2 = array2table(convlut_LCPWS2', 'VariableNames', ...
    {'Sigma', 'D_ACF', 'D', 'Nf', 'lmax'});

% Write the LCPWS1 conversion table to a CSV file
writetable(conversion_LCPWS1,'SigmaToD_LUT_LCPWS1.csv');

% Write the LCPWS2 conversion table to a CSV file
writetable(conversion_LCPWS2,'SigmaToD_LUT_LCPWS2.csv');