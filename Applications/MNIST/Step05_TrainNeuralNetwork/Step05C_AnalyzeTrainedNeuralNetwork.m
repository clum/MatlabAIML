%Analyze the trained neural network
%
%This performs operations such as:
%   -Examine distribution of weights and biases
%   -Examine saturation at each layer
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/19/23: Created
%05/24/23: Refactored workflow
%05/27/23: Continued working
%06/13/23: Renamed file and continued working
%07/30/23: Updated to figure title

clear
clc
close all

tic

%% User selections
% trainedNetworkFile = 'TrainedNeuralNetworkScenario4.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario7.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario10.mat';
trainedNetworkFile = 'TrainedNeuralNetworkScenario11.mat';

displayProgress = true;
ns_saturation = 10; %number of samples to use in saturation analysis

%% Load data
temp = load(trainedNetworkFile);

nn                          = temp.nn;
options                     = temp.options;
trainingDataFile            = temp.trainingDataFile;
initialNeuralNetworkFile    = temp.initialNeuralNetworkFile;
E_data                      = temp.E_data;
norm_gradient_data          = temp.norm_gradient_data;

temp2 = load(trainingDataFile);
U_train     = temp2.U_train;
D_train     = temp2.D_train;
U_test      = temp2.U_test;
D_test      = temp2.D_test;

%% Distribution of Weights and Biases
figh_histW = nn.HistogramWeightsAndBiases();

%% Saturation at Each Layer
%To examine saturation at each layer, we need to forward propagate the
%layer with some data.
[ns_test,~] = size(U_test);

%Assess against testing data
Y_test  = [];
E_test  = [];
classifiedCorrectTest   = 0;
classifiedIncorrectTest = 0;
y_layer_data = {};
deltaFraction = 0.05;        %gradularity in displaying progress
currentFractionThreshold = 1*deltaFraction;
for k=1:ns_saturation
    U = U_test(k,:)';
    D = D_test(k,:)';

    Y = nn.ForwardPropagate(U);

    %What are the outputs at each layer
    for m=1:nn.NumLayers
        [y_m] = nn.GetOutputsAtLayer(m);
        
        if(k==1)
            y_layer_data{m} = [y_m];
        else
            y_layer_data{m} = [y_layer_data{m};y_m];
        end
    end
    
    E_test(end+1) = NeuralNetwork.Error(Y,D,options.errorFunctionID);
    Y_test(:,k) = Y;

    %How did network classify the digit
    idx = find(D==max(D));
    label = idx - 1;

    idx = find(Y==max(Y));
    labelClassified = idx - 1;

    if(labelClassified==label)
        classifiedCorrectTest = classifiedCorrectTest + 1;
    else
        classifiedIncorrectTest = classifiedIncorrectTest + 1;
    end
    
    %Display progress
    if(displayProgress)
        fractionComplete = k/ns_saturation;
        
        if(fractionComplete > currentFractionThreshold)
            disp([num2str(fractionComplete*100),'% complete'])
            currentFractionThreshold = currentFractionThreshold + deltaFraction;
        end
    end
end

%% Visualize results
%Histogram of layer outputs
figure
numBins = 20;
for m=1:nn.NumLayers
    subplot(nn.NumLayers,1,m)
    if(m==1)
        title('Histogram of Layer Outputs')
    end
    
    hist(y_layer_data{m},numBins)
    ylabel(['Layer ',num2str(m)'])
end

toc
disp('DONE!')