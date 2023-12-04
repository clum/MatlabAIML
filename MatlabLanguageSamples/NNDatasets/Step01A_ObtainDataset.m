%Familiarize with various data sets for shallow neural networks.
%
%See https://www.mathworks.com/help/deeplearning/gs/fit-data-with-a-neural-network.html
%
%Christopher Lum
%lum@uw.edu

%Version History
%11/25/23: Created
%12/01/23: Continued working.  Added bodyfat dataset

clear
clc
close all

ChangeWorkingDirectoryToThisLocation();

tic

%% User selections
% dataset = 'simpleseries';
dataset = 'bodyfat';

%% Load data
temp = load([dataset,'_dataset.mat']);

eval(['inputs = temp.',dataset,'Inputs;'])
eval(['targets = temp.',dataset,'Targets;'])

%% Analyze dataset
switch dataset
    case 'simpleseries'
        u = cell2mat(inputs);
        d = cell2mat(targets);
        
    case 'bodyfat'
        u = inputs;
        d = targets;
        
    otherwise
        error('Unsupported dataset')
end

[dimensionInputs,numExamples] = size(u);
[dimensionTargets,numExamples2] = size(d);

assert(numExamples==numExamples2)

disp(['dimensionInputs  = ',num2str(dimensionInputs)])
disp(['dimensionTargets = ',num2str(dimensionTargets)])
disp(['numExamples      = ',num2str(numExamples)])

%% Visualize the dataset
if(dimensionTargets==1)
    figure
    P = ceil(sqrt(dimensionInputs));
    for k=1:dimensionInputs
        subplot(P,P,k)
        plot(u(k,:),d,'rx')
        xlabel(['u_{',num2str(k),'}'])
        ylabel('d')
        grid on
    end
end

toc
disp('DONE!')