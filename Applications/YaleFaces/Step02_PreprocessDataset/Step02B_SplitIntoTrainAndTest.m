%Split the data into training and test datasets
%
%Christopher Lum
%lum@uw.edu

%Version History
%06/18/23: Created
%07/19/23: Added scenarioSelection = 2

clear
clc
close all

tic

%% User selections
scenarioSelection = 2;

%% Load and split data
switch scenarioSelection
    case 1
        %Use subject 14 and 15 as test
        preprocessedDatasetFile = 'PreprocessedDatasetScenario1.mat';
        temp = load(preprocessedDatasetFile);
        
        facePreprocessed        = temp.facePreprocessed;
        subjectNumberLabels     = temp.subjectNumberLabels;
        conditionLabels         = temp.conditionLabels;
      
        idx = [1:1:length(subjectNumberLabels)]';

        idx_test = union(...
            find(subjectNumberLabels==14),...
            find(subjectNumberLabels==15));

        idx_train = setdiff(idx,idx_test);

    case 2
        %Use subject 14 and 15 as test
        preprocessedDatasetFile = 'PreprocessedDatasetScenario2.mat';
        temp = load(preprocessedDatasetFile);
        
        facePreprocessed        = temp.facePreprocessed;
        subjectNumberLabels     = temp.subjectNumberLabels;
        conditionLabels         = temp.conditionLabels;
      
        idx = [1:1:length(subjectNumberLabels)]';

        idx_test = union(...
            find(subjectNumberLabels==14),...
            find(subjectNumberLabels==15));

        idx_train = setdiff(idx,idx_test);

    otherwise
        error('Supported scenarioSelection')
end

assert(length(idx_test)+length(idx_train)==length(idx));

faces_train                 = facePreprocessed(:,:,idx_train);
subjectNumberLabels_train   = subjectNumberLabels(idx_train);
conditionLabels_train       = conditionLabels(idx_train);

faces_test                 = facePreprocessed(:,:,idx_test);
subjectNumberLabels_test   = subjectNumberLabels(idx_test);
conditionLabels_test       = conditionLabels(idx_test);

%% Save data
outputFileName = ['TrainingAndTestDataScenario',num2str(scenarioSelection)];
save(outputFileName,'faces_train','subjectNumberLabels_train','conditionLabels_train','faces_test','subjectNumberLabels_test','conditionLabels_test')
disp(['Saved to ',outputFileName])

toc
disp('DONE!')