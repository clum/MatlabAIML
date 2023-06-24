%Compare 2 NeuralNetwork objects
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/27/23: Created

clear
clc
close all

tic

%% User selections
% trainedNetworkFile_A = 'TrainedNeuralNetworkScenario1_OLD.mat';
% trainedNetworkFile_B = 'TrainedNeuralNetworkScenario1.mat';

% trainedNetworkFile_A = 'TrainedNeuralNetworkScenario2_OLD.mat';
% trainedNetworkFile_B = 'TrainedNeuralNetworkScenario2.mat';

% trainedNetworkFile_A = 'TrainedNeuralNetworkScenario3_OLD.mat';
% trainedNetworkFile_B = 'TrainedNeuralNetworkScenario3.mat';

% trainedNetworkFile_A = 'TrainedNeuralNetworkScenario4_OLD.mat';
% trainedNetworkFile_B = 'TrainedNeuralNetworkScenario4.mat';

% trainedNetworkFile_A = 'TrainedNeuralNetworkScenario2.mat';
% trainedNetworkFile_B = 'TrainedNeuralNetworkScenario5.mat';

trainedNetworkFile_A = 'TrainedNeuralNetworkScenario4.mat';
trainedNetworkFile_B = 'TrainedNeuralNetworkScenario7.mat';

%% Load data
%Set A
temp_A = load(trainedNetworkFile_A);

nn_A                          = temp_A.nn;
options_A                     = temp_A.options;
trainingDataFile_A            = temp_A.trainingDataFile;
initialNeuralNetworkFile_A    = temp_A.initialNeuralNetworkFile;
E_data_A                      = temp_A.E_data;
norm_gradient_data_A          = temp_A.norm_gradient_data;

temp2_A = load(trainingDataFile_A);
U_train_A     = temp2_A.U_train;
D_train_A     = temp2_A.D_train;
U_test_A      = temp2_A.U_test;
D_test_A      = temp2_A.D_test;

%Set B
temp_B = load(trainedNetworkFile_B);

nn_B                          = temp_B.nn;
options_B                     = temp_B.options;
trainingDataFile_B            = temp_B.trainingDataFile;
initialNeuralNetworkFile_B    = temp_B.initialNeuralNetworkFile;
E_data_B                      = temp_B.E_data;
norm_gradient_data_B          = temp_B.norm_gradient_data;

temp2_B = load(trainingDataFile_B);
U_train_B     = temp2_B.U_train;
D_train_B     = temp2_B.D_train;
U_test_B      = temp2_B.U_test;
D_test_B      = temp2_B.D_test;

%Compare the two
figh_histW = nn_A.HistogramWeightsAndBiases();
nn_B.HistogramWeightsAndBiases(figh_histW);

toc
disp('DONE!')