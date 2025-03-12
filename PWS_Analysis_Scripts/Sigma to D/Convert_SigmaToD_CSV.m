% Create_SigmaToD_CSV.m
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

% ALL USERS MUST CHANGE THIS
% Specify the main folder containing all experiments to be converted
folderdir = 'C:\Users\janef\OneDrive - Northwestern University\Documents - Backman Lab - Jane Frederick\Jane Frederick\Papers\CDA Paper\Data\All Compiled Data';

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

%% Convert the Sigma Values in the CSV Files

% Retrieve information about all CSV files within the specified folder and its subfolders
fileinfo = dir([folderdir, '\**\*.csv']);

% Determine the number of CSV files to process
numfiles = size(struct2table(fileinfo), 1);

% Define the output directory for converted D values
outdir = [folderdir, '\..\D Files'];
if ~exist(outdir, 'dir')
    mkdir(outdir); % Create the output directory if it doesn't exist
end

% Loop through each CSV file to perform the Sigma to D conversion
for i=1:numfiles
    filepath = [fileinfo(i).folder, '\', fileinfo(i).name]; % Get the full path of the current file
    opts = detectImportOptions(filepath, 'VariableNamingRule', 'preserve'); % Detect import options
    myTable = readtable(filepath, opts); % Read the CSV file into a table

    % Check if the 'Sigma' column exists in the table
    if(isequal(any(strcmp('Sigma', myTable.Properties.VariableNames)),1))
        sigma = table2array(myTable(:, "Sigma")); % Extract Sigma values from the table

        newstr = extract(fileinfo(i).name, "20" + digitsPattern(2)); % Extract specific pattern from filename
        sigma = sigma.*3; % Apply extra reflection correction factor to Sigma values
        nuSys = LCPWS1Sys; % Assign the system configuration (modify if using LCPWS2)

        % Uncomment and modify the following lines if using PWS2 system
        % sigma_noise = 0.009; % Sigma values for background regions indicating noise
        % nuSys = LCPWS2Sys; % Assign LCPWS2 system configuration
        % sigma_noise = 0.1;
        % sigma = sqrt(sigma.^2 - sigma_noise^2); % Subtract noise from Sigma values

        % Perform the Sigma to D conversion using all inputs
        [DB, D, Nf, lmax] = SigmaToD_AllInputs(sigma, nuSys, N_PD, thickness);
        
        % Append the new D, Avg_N_PD, and Avg_r_PD columns to the table
        myTable = [myTable table(D, 'VariableNames', {'D_n'}) 
            table(Nf, 'VariableNames', {'Avg_N_PD'}) 
            table(lmax, 'VariableNames', {'Avg_r_PD'})];

        % Define the output folder path for the current file
        outputfold = [outdir, extractAfter(fileinfo(i).folder, folderdir)];
        if ~exist(outputfold, 'dir')
            mkdir(outputfold); % Create the output subfolder if it doesn't exist
        end

        pat = "R" + digitsPattern(4) + ("a"|"b"); % Define pattern for filename extraction
        [~, name, ext] = fileparts(filepath); % Split the file into name and extension
        writetable(myTable, strcat(outputfold, '\', name, '.csv')); % Write the updated table to a new CSV file
    end

    fprintf('%1.0f out of %1.0f files processed.\n', i, numfiles); % Display progress
end
fprintf('Sigma to D conversion for all %1.0f files completed.\n', numfiles); % Completion message
