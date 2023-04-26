%Setup the training dataset
%
%Christopher Lum
%lum@uw.edu

%Version History
%04/22/23: Created

clear
clc
close all

tic

%% User selections
scenarioSelection = 10;

%% Load data
cwd = pwd;

step01Directory = [fileparts(cwd),'\Step01_ObtainDataset'];
cd(step01Directory);

temp = load('MNISTData.mat');
TrainingSetImages   = temp.TrainingSetImages;
TrainingSetLabels   = temp.TrainingSetLabels;
TestSetImages       = temp.TestSetImages;
TestSetLabels       = temp.TestSetLabels;

cd(cwd);

%% Generate data set
%Combine into a single data set
Images = cat(3,TrainingSetImages,TestSetImages);
Labels = [TrainingSetLabels;TestSetLabels];

switch scenarioSelection
    case 1
        %Split using 50,000 for training and 10,000 for test (similar to
        %http://neuralnetworksanddeeplearning.com/chap1.html)
        numTrainingSamples  = 50000;
        numTestSamples      = 10000;
        
        idxTrainingStart    = 1;
        idxTrainingEnd      = idxTrainingStart + numTrainingSamples - 1;
        
        idxTestStart        = idxTrainingEnd + 1;
        idxTestEnd          = idxTestStart + numTestSamples - 1;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 2
        %Training samples:  00,001 - 10,000
        %Test samples:      60,001 - 70,000
        idxTrainingStart    = 1;
        idxTrainingEnd      = 10000;
        
        idxTestStart        = 60001;
        idxTestEnd          = 70000;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 3
        %Training samples:  10,001 - 20,000
        %Test samples:      60,001 - 70,000
        idxTrainingStart    = 10001;
        idxTrainingEnd      = 20000;
        
        idxTestStart        = 60001;
        idxTestEnd          = 70000;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 4
        %Training samples:  20,001 - 30,000
        %Test samples:      60,001 - 70,000
        idxTrainingStart    = 20001;
        idxTrainingEnd      = 30000;
        
        idxTestStart        = 60001;
        idxTestEnd          = 70000;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 5
        %Training samples:  30,001 - 40,000
        %Test samples:      60,001 - 70,000
        idxTrainingStart    = 30001;
        idxTrainingEnd      = 40000;
        
        idxTestStart        = 60001;
        idxTestEnd          = 70000;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 6
        %Training samples:  40,001 - 50,000
        %Test samples:      60,001 - 70,000
        idxTrainingStart    = 20001;
        idxTrainingEnd      = 30000;
        
        idxTestStart        = 60001;
        idxTestEnd          = 70000;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];

    case 7
        %Training samples:  50,001 - 60,000
        %Test samples:      60,001 - 70,000
        idxTrainingStart    = 50001;
        idxTrainingEnd      = 60000;
        
        idxTestStart        = 60001;
        idxTestEnd          = 70000;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];

    case 8
        %Similar to case 1 but randomly shuffle the samples
        P = randperm(length(Labels));
        
        ImagesShuffled = Images(:,:,P);
        LabelsShuffled = Labels(P);
        
        Images = ImagesShuffled;
        Labels = LabelsShuffled;
        
        numTrainingSamples  = 50000;
        numTestSamples      = 10000;
        
        idxTrainingStart    = 1;
        idxTrainingEnd      = idxTrainingStart + numTrainingSamples - 1;
        
        idxTestStart        = idxTrainingEnd + 1;
        idxTestEnd          = idxTestStart + numTestSamples - 1;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 9
        %Similar to case 1 but randomly shuffle the samples twice
        P = randperm(length(Labels));
        P = randperm(length(Labels));   %shuffle #2
        
        ImagesShuffled = Images(:,:,P);
        LabelsShuffled = Labels(P);
        
        Images = ImagesShuffled;
        Labels = LabelsShuffled;
        
        numTrainingSamples  = 50000;
        numTestSamples      = 10000;
        
        idxTrainingStart    = 1;
        idxTrainingEnd      = idxTrainingStart + numTrainingSamples - 1;
        
        idxTestStart        = idxTrainingEnd + 1;
        idxTestEnd          = idxTestStart + numTestSamples - 1;
        
        idxTraining         = [idxTrainingStart:1:idxTrainingEnd];
        idxTest             = [idxTestStart:1:idxTestEnd];
        
    case 10
        %Similar to case 1 but randomly shuffle the samples three times
        P = randperm(length(Labels));
        P = randperm(length(Labels));   %shuffle #2
        P = randperm(length(Labels));   %shuffle #3
        
        ImagesShuffled = Images(:,:,P);
        LabelsShuffled = Labels(P);
        
        Images = ImagesShuffled;
        Labels = LabelsShuffled;
        
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

TrainingSetImages   = Images(:,:,idxTraining);
TrainingSetLabels   = Labels(idxTraining);

TestSetImages       = Images(:,:,idxTest);
TestSetLabels       = Labels(idxTest);

%% Save data
outputFileName = ['TrainingAndTestDataScenario',num2str(scenarioSelection)];
save(outputFileName,'TrainingSetImages','TrainingSetLabels','TestSetImages','TestSetLabels')
disp(['Saved to ',outputFileName])


toc
disp('DONE!')