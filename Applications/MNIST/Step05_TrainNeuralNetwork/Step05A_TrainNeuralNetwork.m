%Train the Neural Network
%
%Christopher Lum
%lum@uw.edu

%Version History
%04/22/23: Created
%04/23/23: Continued working
%05/06/23: Changed to use NeuralNetwork.TrainNeuralNetwork method
%05/24/23: Refactored architecture

clear
clc
close all

tic

%% User selections
scenarioSelection = 2;

switch scenarioSelection
    case 1
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario1.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.22;
        options.miniBatchSize   = 32;
        options.numEpochs       = 5;
        options.displayProgress = true;
        
    case 2
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario1.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.05;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;
        
    case 3
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario1.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.01;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;
        
    case 4
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario2.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.0757;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;
        
    otherwise
        error('')
end

%% Load data
temp = load(trainingDataFile);
disp(['Loading from ',trainingDataFile])

U_train     = temp.U_train;
D_train     = temp.D_train;
U_test      = temp.U_test;
D_test      = temp.D_test;

%% Train network
disp('Training network')
disp(['Loading neural network from ',initialNeuralNetworkFile])
temp = load(initialNeuralNetworkFile);
nn = temp.nn;

[E_data,norm_gradient_data] = nn.TrainNeuralNetworkMultiEpoch(U_train,D_train,options);

%% Save network
outputFile = ['TrainedNeuralNetworkScenario',num2str(scenarioSelection),'.mat'];
saveVars = {
    'nn'
    'options'
    
    'trainingDataFile'
    'initialNeuralNetworkFile'
    
    'E_data'
    'norm_gradient_data'
    };
s = SaveVarsString(outputFile,saveVars);
eval(s);
disp(['Saved to ',outputFile])

toc
disp('DONE!')
