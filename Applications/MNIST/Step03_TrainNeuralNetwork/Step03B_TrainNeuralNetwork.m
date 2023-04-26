%Train the Neural Network on the the MNIST dataset
%
%Christopher Lum
%lum@uw.edu

%Version History
%04/22/23: Created
%04/23/23: Continued working

clear
clc
close all

tic

%% User selections
scenarioSelection = 103;

switch scenarioSelection
    case 1
        %Accuracy: 26.76%, 26.74% (eta = 0.001)
        %Accuracy: 43.86%, 43.51% (eta = 0.002)
        %Accuracy: 56.49%, 58.30% (eta = 0.005)
        %Accuracy: 55.52%, 57.01% (eta = 0.006)
        %Accuracy: 60.40%, 61.79% (eta = 0.007)
        %Accuracy: 55.37%, 56.62% (eta = 0.008)
        %Accuracy: 65.76%, 68.43% (eta = 0.009)
        %Accuracy: 65.13%, 67.33% (eta = 0.01)
        %Accuracy: 65.39%, 67.90% (eta = 0.011)
        %Accuracy: 66.17%, 67.97% (eta = 0.012)     %use this
        %Accuracy: 62.08%, 63.21% (eta = 0.013)
        %Accuracy: 63.23%, 65.50% (eta = 0.02)
        %Accuracy: 47.82%, 48.28% (eta = 0.05)
        %Accuracy: 43.18%, 44.39% (eta = 0.1)
        %Accuracy: 22.08%, 21.80% (eta = 0.2)
        %Accuracy: 13.14%, 13.44% (eta = 0.5)
        %Accuracy: 15.77%, 16.26% (eta = 1)
        %Accuracy: 15.35%, 16.34% (eta = 2)
        trainingDataFile = 'TrainingAndTestDataScenario1.mat';
        nodesPerLayer = [28*28 50 10];
        errorFunctionID = ErrorFunctionID.SquaredError;
        
        numSteps    = 50000; %num steps of SGD
        
        numSubSteps = 1;  %num steps to take at a given example
        eta         = 0.012;   %step size (AKA learning rate)
        
        plotSubstepProgress = false;
        
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        
    case 2
        %Accuracy: 09.90%, 10.40% (numSubSteps = 3)
        %Accuracy: 10.41%, 10.91% (numSubSteps = 10)
        trainingDataFile = 'TrainingAndTestDataScenario1.mat';
        nodesPerLayer = [28*28 50 10];
        errorFunctionID = ErrorFunctionID.SquaredError;
        
        numSteps    = 50000; %num steps of SGD
        
        numSubSteps = 3;  %num steps to take at a given example
        eta         = 1;   %step size (AKA learning rate)
        
        plotSubstepProgress = true;
        
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        
    case 3
        %Accuracy: 45.09%, 46.23% (eta = 0.1)
        %Accuracy: 17.24%, 17.50% (eta = 1)
        trainingDataFile = 'TrainingAndTestDataScenario1.mat';
        plotSubstepProgress = false;
        
        %Warm start: Load network from previous model
        temp = load('TrainedNetwork_scenario1.mat');
        
        nn              = temp.nn;
        errorFunctionID = temp.errorFunctionID;
        numSteps        = temp.numSteps;
        numSubSteps     = temp.numSubSteps;
        eta             = temp.eta;
        
    case 4
        %Accuracy: 49.35%, 51.08% (eta = 0.1)
        %Accuracy: 12.01%, 12.45% (eta = 1)
        trainingDataFile = 'TrainingAndTestDataScenario1.mat';
        plotSubstepProgress = false;
        
        %Warm start: Load network from previous model
        temp = load('TrainedNetwork_scenario3.mat');
        
        nn              = temp.nn;
        errorFunctionID = temp.errorFunctionID;
        numSteps        = temp.numSteps;
        numSubSteps     = temp.numSubSteps;
        eta             = temp.eta;
        
    case 5
        %Accuracy: 34.47%, 34.95% (eta = 0.009)
        %Accuracy: 34.75%, 35.86% (eta = 0.01)
        %Accuracy: 39.46%, 37.72% (eta = 0.015)
        %Accuracy: 39.15%, 39.80% (eta = 0.016)
        %Accuracy: 42.55%, 42.36% (eta = 0.019)
        %Accuracy: 45.53%, 45.73% (eta = 0.02)      Using this
        %Accuracy: 44.59%, 45.33% (eta = 0.021)
        %Accuracy: 41.23%, 41.53% (eta = 0.025)
        %Accuracy: 38.72%, 39.02% (eta = 0.05)
        
        trainingDataFile = 'TrainingAndTestDataScenario2.mat';  %Training samples:  00,001 - 10,000
        nodesPerLayer = [28*28 50 10];
        errorFunctionID = ErrorFunctionID.SquaredError;
        
        numSteps    = 10000; %num steps of SGD
        
        numSubSteps = 1;  %num steps to take at a given example
        eta         = 0.020;   %step size (AKA learning rate)
        
        plotSubstepProgress = false;
        
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        
    case 6
        %Accuracy: 52.63%, 55.40% (eta = 0.01)
        %Accuracy: 55.96%, 57.40% (eta = 0.02)
        %Accuracy: 52.61%, 54.62% (eta = 0.03)
        %Accuracy: 37.47%, 37.96% (eta = 0.1)
        trainingDataFile = 'TrainingAndTestDataScenario3.mat';  %Training samples:  10,001 - 20,000
        plotSubstepProgress = false;
        
        %Warm start: Load network from previous model
        temp = load('TrainedNetwork_scenario5.mat');
        
        nn              = temp.nn;
        errorFunctionID = temp.errorFunctionID;
        numSteps        = temp.numSteps;
        numSubSteps     = temp.numSubSteps;
        eta             = 0.1;
        
    case 99
        trainingDataFile = 'TrainingAndTestDataScenario8.mat';
        nodesPerLayer = [28*28 50 50 10];
        errorFunctionID = ErrorFunctionID.SquaredError;
        
        numSteps    = 25000; %num steps of SGD
        
        numSubSteps = 1;  %num steps to take at a given example
        eta         = 0.012;   %step size (AKA learning rate)
        
        plotSubstepProgress = false;
        
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        
    case 100
        %Accuracy: 66.17%, 67.97%       Starting point
        
        %Accuracy: 66.91%, 66.54% (eta = 0.0001)
        %Accuracy: 68.06%, 67.79% (eta = 0.001)
        %Accuracy: 68.05%, 67.84% (eta = 0.005)
        %Accuracy: 68.07%, 68.30% (eta = 0.01)
        %Accuracy: 68.52%, 68.11% (eta = 0.02)
        %Accuracy: 63.57%, 62.97% (eta = 0.03)
        
        trainingDataFile = 'TrainingAndTestDataScenario8.mat';  %Reshuffled set
        plotSubstepProgress = false;
        
        %Warm start: Load network from previous model
        temp = load('TrainedNetwork_scenario1.mat');
        
        nn              = temp.nn;
        errorFunctionID = temp.errorFunctionID;
        numSteps        = 5000;
        numSubSteps     = 1;
        eta             = 0.03;
        
    case 101
        %Epoch #2
        %Accuracy: 66.17%, 67.97%       Starting point
        
        %Accuracy: 72.20%, 72.05% (eta = 0.01, numSteps = 25000)
        
        %Accuracy: 70.91%, 70.67% (eta = 0.001, numSteps = 50000)
        %Accuracy: 73.79%, 72.59% (eta = 0.005, numSteps = 50000)
        %Accuracy: 75.74%, 75.93% (eta = 0.006, numSteps = 50000)  use this
        %Accuracy: 74.12%, 73.41% (eta = 0.007, numSteps = 50000)
        %Accuracy: 72.48%, 71.50% (eta = 0.010, numSteps = 50000)
        
        trainingDataFile = 'TrainingAndTestDataScenario8.mat';  %Reshuffled set
        plotSubstepProgress = false;
        
        %Warm start: Load network from previous model
        temp = load('TrainedNetwork_scenario1.mat');
        
        nn              = temp.nn;
        errorFunctionID = temp.errorFunctionID;
        numSteps        = 50000;
        numSubSteps     = 1;
        eta             = 0.006;
        
    case 102
        %Epoch #3
        %Accuracy: 75.74%, 75.93%       Starting point
        
        %Accuracy: 77.09%, 77.12% (eta = 0.0005, numSteps = 50000)
        %Accuracy: 77.88%, 77.98% (eta = 0.001, numSteps = 50000)
        %Accuracy: 78.37%, 78.46% (eta = 0.0015, numSteps = 50000)    use this
        %Accuracy: 78.68%, 78.24% (eta = 0.002, numSteps = 50000)
        %Accuracy: 77.39%, 76.95% (eta = 0.005, numSteps = 50000)
        %Accuracy: 78.60%, 78.77% (eta = 0.006, numSteps = 50000)
        %Accuracy: 77.77%, 77.67% (eta = 0.007, numSteps = 50000)
        %Accuracy: 77.26%, 76.78% (eta = 0.01, numSteps = 50000)
        
        trainingDataFile = 'TrainingAndTestDataScenario9.mat';  %Reshuffled #2 set
        plotSubstepProgress = false;
        
        %Warm start: Load network from previous model
        temp = load('TrainedNetwork_scenario101.mat');
        
        nn              = temp.nn;
        errorFunctionID = temp.errorFunctionID;
        numSteps        = 50000;
        numSubSteps     = 1;
        eta             = 0.0015;
        
    case 103
        %Epoch #4
        %Accuracy: 78.37%, 78.46%       Starting point
        
        %Accuracy: 79.59%, 79.10% (eta = 0.001, numSteps = 50000)
        %Accuracy: 80.06%, 79.42% (eta = 0.002, numSteps = 50000)
        %Accuracy: 80.26%, 79.87% (eta = 0.003, numSteps = 50000)
        
        trainingDataFile = 'TrainingAndTestDataScenario10.mat';  %Reshuffled #3 set
        plotSubstepProgress = false;
        
        %Warm start: Load network from previous model
        temp = load('TrainedNetwork_scenario102.mat');
        
        nn              = temp.nn;
        errorFunctionID = temp.errorFunctionID;
        numSteps        = 50000;
        numSubSteps     = 1;
        eta             = 0.003;
        
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

%% Train network
E_data              = [];
U_data              = [];
D_data              = [];
Y_data              = [];
norm_gradient_data  = [];


deltaFraction = 0.1;        %gradularity in displaying progress
currentFractionThreshold = 1*deltaFraction;
for k=1:numSteps
    A       = TrainingSetImages(:,:,k);
    label   = TrainingSetLabels(k);
    
    %Reshape into an input vector U (stack each column on top of one
    %another) and convert to double
    [M,N] = size(A);
    U = double(reshape(A,M*N,1));
    
    %Convert the label to a vector d
    D = LabelToVector(label);
    
    Esub_data = [];
    for m=1:numSubSteps
        %Compute gradient
        [dEc_dW,dEc_db] = nn.BackPropagate(U,D,errorFunctionID);
        
        %Take optimization gradient descent step (AKA update weights and biases)
        for n=1:nn.NumLayers-1
            L = n+1;
            
            W_L = nn.GetWeightsIncomingToLayer(L);
            b_L = nn.GetBiasesAtLayer(L);
            
            dEc_dW_L = dEc_dW{n};   %output from BackPropagate is off by 1
            dEc_db_L = dEc_db{n};   %output from BackPropagate is off by 1
            
            W_L_prime = W_L - eta*dEc_dW_L;
            b_L_prime = b_L - eta*dEc_db_L;
            
            nn.SetWeightsIncomingToLayer(L,W_L_prime);
            nn.SetBiasesAtLayer(L,b_L_prime);
        end
        
        %Optional: Check if the norm of the gradient is below a threshold
        %and terminate if appropriate
        if(plotSubstepProgress)
            [norm_dEc_dW,norm_dEc_db] = NeuralNetwork.GradientNorm(dEc_dW,dEc_db);
            
            Y = nn.ForwardPropagate(U);
            E = NeuralNetwork.Error(Y,D,errorFunctionID);
            Esub_data(:,end+1) = E;
        end
    end
    
    if(plotSubstepProgress)
        figh_Esubstep = figure;
        plot(Esub_data);
        plotSubstepProgress = false;
    end
    
    %Compute various performance metrics at the end of the substeps
    [norm_dEc_dW,norm_dEc_db] = NeuralNetwork.GradientNorm(dEc_dW,dEc_db);
    
    Y = nn.ForwardPropagate(U);
    E = NeuralNetwork.Error(Y,D,errorFunctionID);
    
    E_data(:,end+1)             = E;
    U_data(:,end+1)             = U;
    D_data(:,end+1)             = D;
    Y_data(:,end+1)             = Y;
    norm_gradient_data(:,end+1) = norm_dEc_dW + norm_dEc_db;
    
    %Display progress
    fractionComplete = k/numSteps;
    
    if(fractionComplete > currentFractionThreshold)
        disp([num2str(fractionComplete*100),'% complete'])
        currentFractionThreshold = currentFractionThreshold + deltaFraction;
    end
end

%% Visualize training results
%Average E_data
averageWindow = 100;
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
    'errorFunctionID'
    'numSteps'
    'numSubSteps'
    'eta'
    
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
