%Split the flattened data into training and test datasets
%
%Christopher Lum
%lum@uw.edu

%Version History
%04/22/23: Created
%05/21/23: Modified
%05/28/23: Updated
%05/29/23: Updated

clear
clc
close all

tic

%% User selections
scenarioSelection = 5;

%% Load and split data
switch scenarioSelection
    case 1
        %Split using 50,000 for training and 10,000 for test (similar to
        %http://neuralnetworksanddeeplearning.com/chap1.html)
        flatDataFile = 'FlatDatasetScenario1.mat';
        temp = load(flatDataFile);
        
        U_data = temp.U_data;
        D_data = temp.D_data;
        
        numTrainingSamples  = 50000;
        numTestSamples      = 10000;
        
        idxTrainingStart    = 1;
        idxTrainingEnd      = idxTrainingStart + numTrainingSamples - 1;
        
        idxTestStart        = idxTrainingEnd + 1;
        idxTestEnd          = idxTestStart + numTestSamples - 1;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
      
    case 2
        %Similar to case 1 but randomly shuffle the samples
        flatDataFile = 'FlatDatasetScenario1.mat';        
        temp = load(flatDataFile);
        
        U_data = temp.U_data;
        D_data = temp.D_data;

        numTrainingSamples  = 50000;
        numTestSamples      = 10000;
        
        [ns,nu] = size(U_data);
        P = randperm(ns)';
        
        idxTraining = P(1:1:numTrainingSamples);
        idxTest     = P(idxTraining+1:1:idxTraining+1+numTestSamples-1);
                
    case 3
        %Similar to case 1 but randomly shuffle the samples, small set for
        flatDataFile = 'FlatDatasetScenario1.mat';
        temp = load(flatDataFile);
        
        U_data = temp.U_data;
        D_data = temp.D_data;

        numTrainingSamples  = 2000;
        numTestSamples      = 200;
        
        [ns,nu] = size(U_data);
        P = randperm(ns)';
        
        idxTraining = P(1:1:numTrainingSamples);
        idxTest     = P(idxTraining+1:1:idxTraining+1+numTestSamples-1);
        
    case 4
        %Split using 50,000 for training and 10,000 for test (similar to
        %http://neuralnetworksanddeeplearning.com/chap1.html)
        flatDataFile = 'FlatDatasetScenario2.mat';
        temp = load(flatDataFile);
        
        U_data = temp.U_data;
        D_data = temp.D_data;
        
        numTrainingSamples  = 50000;
        numTestSamples      = 10000;
        
        idxTrainingStart    = 1;
        idxTrainingEnd      = idxTrainingStart + numTrainingSamples - 1;
        
        idxTestStart        = idxTrainingEnd + 1;
        idxTestEnd          = idxTestStart + numTestSamples - 1;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 5
        %Split using 50,000 for training and 10,000 for test (similar to
        %http://neuralnetworksanddeeplearning.com/chap1.html)
        flatDataFile = 'FlatDatasetScenario3.mat';
        temp = load(flatDataFile);
        
        U_data = temp.U_data;
        D_data = temp.D_data;
        
        numTrainingSamples  = 50000;
        numTestSamples      = 10000;
        
        idxTrainingStart    = 1;
        idxTrainingEnd      = idxTrainingStart + numTrainingSamples - 1;
        
        idxTestStart        = idxTrainingEnd + 1;
        idxTestEnd          = idxTestStart + numTestSamples - 1;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    otherwise
        error('Supported scenarioSelection')
end

U_train = U_data(idxTraining,:);
D_train = D_data(idxTraining,:);

U_test  = U_data(idxTest,:);
D_test  = D_data(idxTest,:);

%Display outputs
[ns_train,nu_train] = size(U_train);
[ns_train,no_train] = size(D_train);

[ns_test,nu_test] = size(U_test);
[ns_test,no_test] = size(D_test);

disp('Training')
disp(['ns = ',num2str(ns_train)])
disp(['nu = ',num2str(nu_train)])
disp(['no = ',num2str(no_train)])

disp('Test')
disp(['ns = ',num2str(ns_test)])
disp(['nu = ',num2str(nu_test)])
disp(['no = ',num2str(no_test)])

%% Save data
outputFileName = ['TrainingAndTestDataScenario',num2str(scenarioSelection)];
save(outputFileName,'U_train','D_train','U_test','D_test')
disp(['Saved to ',outputFileName])

toc
disp('DONE!')