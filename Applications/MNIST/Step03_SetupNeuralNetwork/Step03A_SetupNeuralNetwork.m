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
        
        %Initialize weights between random weights/biases uniformly between [-1,1]
        %Set weights (note the first element of weights is the weights
        %incoming to layer 2)
        weights = {};
        for L=1:length(nodesPerLayer)-1
            %Randomize between [-1,1]
            weights{L} = 2*(rand(nodesPerLayer(L+1),nodesPerLayer(L)) - 0.5);
        end
        
        for L=2:length(nodesPerLayer)
            nn.SetWeightsIncomingToLayer(L,weights{L-1});
        end
        
        %Set biases (note the first element of biases is the biases in
        %layer 2)
        biases = {};
        for L=1:length(nodesPerLayer)-1
            %Randomize between [-1,1]
            biases{L} = 2*(rand(nodesPerLayer(L+1),1) - 0.5);
        end
        
        for L=2:length(nodesPerLayer)
            nn.SetBiasesAtLayer(L,biases{L-1});
        end
        
    case 2
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
