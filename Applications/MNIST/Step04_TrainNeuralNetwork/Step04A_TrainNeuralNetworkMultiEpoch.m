%Train the Neural Network on the the MNIST dataset using multiple epochs
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/07/23: Created

clear
clc
close all

tic

%% User selections
scenarioSelection = 301;

switch scenarioSelection
    case 301
        %Accuracy: 61.62%, 64.85% (eta = 1.0, miniBatch = 30, numEpochs = 1)
        %Accuracy: 63.39%, 65.52% (eta = 1.0, miniBatch = 30, numEpochs = 2)
        %Accuracy: 69.02%, 71.09% (eta = 1.0, miniBatch = 30, numEpochs = 10)
        
        trainingDataFile = 'TrainingAndTestDataScenario1.mat';        
        nodesPerLayer   = [28*28 50 10];
        errorFunctionID = ErrorFunctionID.SquaredError;                
        numSubSteps     = 1;  %num steps to take at a given example
        eta             = 1;   %step size (AKA learning rate)
        miniBatchSize   = 30;
        numEpochs       = 10;
        displayProgress = true;        
                
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        
        %Format as options
        options.errorFunctionID = errorFunctionID;
        options.numSubSteps     = numSubSteps;
        options.eta             = eta;
        options.miniBatchSize   = miniBatchSize;
        options.numEpochs       = numEpochs;
        options.displayProgress = displayProgress;
        
    otherwise
        error('')
end

%% Load data
temp = load(trainingDataFile);
disp(['Loading from ',trainingDataFile])

TrainingSetImages   = temp.TrainingSetImages;
TrainingSetLabels   = temp.TrainingSetLabels;
TestSetImages       = temp.TestSetImages;
TestSetLabels       = temp.TestSetLabels;

%View several sample images and labels
figh_training = figure;
for k=1:9
    subplot(3,3,k)
    A       = TrainingSetImages(:,:,k);
    label   = TrainingSetLabels(k);
    
    imshow(A)
    title(['idx ',num2str(k),',  Label ' ,num2str(label)])
end

figh_test = figure;
for k=1:9
    subplot(3,3,k)
    A       = TestSetImages(:,:,k);
    label   = TestSetLabels(k);
    
    imshow(A)
    title(['idx ',num2str(k),',  Label ' ,num2str(label)])
end

%Reformat data
disp('Formatting training data')

%Create U_data and D_data matrices
nu = nn.NumInputs;
no = nn.NumOutputs;
ns = length(TrainingSetLabels);

U_data = zeros(ns,nu);
D_data = zeros(ns,no);

deltaFraction = 0.1;        %gradularity in displaying progress
currentFractionThreshold = 1*deltaFraction;
for k=1:ns
    A       = TrainingSetImages(:,:,k);
    label   = TrainingSetLabels(k);
    
    %Reshape into an input vector U (stack each column on top of one
    %another) and convert to double
    [M,N] = size(A);
    U = double(reshape(A,M*N,1));
    
    %Convert the label to a vector d
    D = LabelToVector(label);
    
    U_data(k,:) = U';
    D_data(k,:) = D';
    
    %Display progress
    fractionComplete = k/ns;
    
    if(fractionComplete > currentFractionThreshold)
        disp([num2str(fractionComplete*100),'% complete'])
        currentFractionThreshold = currentFractionThreshold + deltaFraction;
    end
end
  
%% Train network
disp('Training network')
[E_data,norm_gradient_data] = nn.TrainNeuralNetworkMultiEpoch(U_data,D_data,options);
E_data              = E_data';
norm_gradient_data  = norm_gradient_data';

%% Visualize training results
%Average E_data
averageWindow = 20;
E_data_average = NaN(1,length(E_data));
for k=averageWindow:length(E_data)
    idxStart    = k-averageWindow+1;
    idxEnd      = idxStart+averageWindow-1;
    E_data_average(k) = sum(E_data(idxStart:idxEnd))/averageWindow;
end

figure
subplot(3,1,1)
hold on
plot(E_data,'DisplayName','E')
plot(E_data_average,'r-','DisplayName',['Averaged over last ',num2str(averageWindow),' samples'])
grid on
legend()

subplot(3,1,2)
hold on
plot(E_data_average,'r-','DisplayName',['Averaged over last ',num2str(averageWindow),' samples'])
grid on
legend()

subplot(3,1,3)
plot(norm_gradient_data)
grid on
legend(StringWithUnderscoresForPlot('norm_gradient'))

%% Save network
outputFile = ['TrainedNetwork_scenario',num2str(scenarioSelection),'.mat'];
saveVars = {
    'nn'
    'options'
    
    'TrainingSetImages'
    'TrainingSetLabels'
    'TestSetImages'
    'TestSetLabels'
    
    'E_data'
    'norm_gradient_data'
    };
s = SaveVarsString(outputFile,saveVars);
eval(s);
disp(['Saved to ',outputFile])

toc
disp('DONE!')
