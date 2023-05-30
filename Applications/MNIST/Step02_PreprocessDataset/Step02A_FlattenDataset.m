%Flatten the dataset by converting images and labels into flat matrices.
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/21/23: Created
%05/28/23: Added scenario = 2;
%05/29/23: Added scenario = 3;

clear
clc
close all

tic

%% User selections
%1 = 10 outputs, each output is either 0 or 1
%2 = 10 outputs, each output is either 0.1 or 0.9
%3 = 10 outputs, each output is either 0.2 or 0.8
scenarioSelection = 3;

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

%% Setup matrices
%Combine into a single data set
Images = cat(3,TrainingSetImages,TestSetImages);
Labels = [TrainingSetLabels;TestSetLabels];

%Flatten images and labels into matrices
ns = length(Labels);
switch scenarioSelection
    case {1,2,3}
        %784 inputs, 10 outputs
        A = Images(:,:,1);
        [M,N] = size(A);
        nu = M*N;
        no = 10;
    
    otherwise
        error('Supported scenarioSelection')
end

U_data = zeros(ns,nu);
D_data = zeros(ns,no);

deltaFraction = 0.1;        %gradularity in displaying progress
currentFractionThreshold = 1*deltaFraction;
for k=1:ns    
    %% Flatten inputs
    A = Images(:,:,k);

    switch scenarioSelection
        case {1,2,3}
            %Reshape into an input vector U (stack each column on top of one
            %another) and convert to double
            [M,N] = size(A);
            U = double(reshape(A,M*N,1));            
            
        otherwise
            error('Supported scenarioSelection')
    end
    
    U_data(k,:) = U';
    
    %% Flatten outputs
    label   = Labels(k);

    switch scenarioSelection
        case 1
            %labels converted to vectors w/ each element is either 0 or 1
            %Convert the label to a vector d
            D = LabelToVector(label);
            
        case 2
            %labels converted to vectors w/ each element is either 0.1 or
            %0.9
            %Convert the label to a vector d
            D = LabelToVector(label);
            
            idx0 = find(D==0);
            idx1 = find(D==1);
            
            D(idx0) = 0.1;
            D(idx1) = 0.9;

        case 3
            %labels converted to vectors w/ each element is either 0.2 or
            %0.8
            %Convert the label to a vector d
            D = LabelToVector(label);
            
            idx0 = find(D==0);
            idx1 = find(D==1);
            
            D(idx0) = 0.2;
            D(idx1) = 0.8;
    
        otherwise
            error('Supported scenarioSelection')
    end
            
    D_data(k,:) = D';
    
    %Display progress
    fractionComplete = k/ns;
    
    if(fractionComplete > currentFractionThreshold)
        disp([num2str(fractionComplete*100),'% complete'])
        currentFractionThreshold = currentFractionThreshold + deltaFraction;
    end
end

%% Save data
outputFileName = ['FlatDatasetScenario',num2str(scenarioSelection)];
save(outputFileName,'U_data','D_data')
disp(['Saved to ',outputFileName])

toc
disp('DONE!')