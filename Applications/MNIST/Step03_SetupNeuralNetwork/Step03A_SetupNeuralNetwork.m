%Setup the NeuralNetwork object
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/22/23: Created
%05/25/23: Adding glorot initialization

clear
clc
close all

tic

%% User selections
%1 = [784 50 10]
scenarioSelection = 1;

switch scenarioSelection
    case 1
        nodesPerLayer = [28*28 50 10];
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        
        %Initialize random number generator so weights and biases are
        %deterministically set
        seed = 1;
        rng(seed);
        
        methodID = WeightInitializationMethodID.Uniform;
        methodParams.LowerBound = -1;
        methodParams.UpperBound = 1;
        nn.InitializeWeightsAndBiases(methodID,methodParams);
        
    case {2}
        %Created by other process, this is a placeholder
        disp(['scenarioSelection = ',num2str(scenarioSelection),' is setup/created by another process'])
        
    case 3
        %Glorot initialization

        nodesPerLayer = [28*28 50 10];
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        
        %Initialize random number generator so weights and biases are
        %deterministically set
        seed = 1;
        rng(seed);
        
        %Initialize weights between random weights/biases uniformly between [-1,1]
        %Set weights (note the first element of weights is the weights
        %incoming to layer 2)
        

    otherwise
        error('Unsupported scenarioSelection')
end

%% Save network
outputFile = ['NeuralNetworkScenario',num2str(scenarioSelection),'.mat'];
saveVars = {
    'nn'
    };
s = SaveVarsString(outputFile,saveVars);
eval(s);
disp(['Saved to ',outputFile])

toc
disp('DONE!')
