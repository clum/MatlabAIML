%Train the Neural Network
%
%Christopher Lum
%lum@uw.edu

%Version History
%04/22/23: Created
%04/23/23: Continued working
%05/06/23: Changed to use NeuralNetwork.TrainNeuralNetwork method
%05/24/23: Refactored architecture
%05/28/23: Adding scenarioSelection = 5
%05/29/23: Adding scenarioSelection = 6
%06/12/23: Adding scenarioSelection = 7
%06/14/23: Adding scenarioSelection = 9
%07/29/23: Adding scenarioSelection = 10
%07/31/23: Adding scenarioSelection = 11
%08/01/23: Adding scenarioSelection = 12
%08/03/23: Adding scenarioSelection = 13
%08/04/23: Adding scenarioSelection = 14

clear
clc
close all

tic

%% User selections
% scenarioSelection = 11;
% scenarioSelection = 12;
% scenarioSelection = 13;
% scenarioSelection = 14;
% scenarioSelection = 15;
% scenarioSelection = 16;
% scenarioSelection = 17;
scenarioSelection = 18;

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
 
    case 5
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario1.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.05;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;

    case 6
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario5.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario1.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.05;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;
        
    case 7
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario3.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.006;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;
        
    case 8
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario3.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.01;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;
        
    case 9
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario5.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario3.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.01;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        options.displayProgress = true;
                
    case 10
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario3.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario4.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 1.5000e-04;
        options.miniBatchSize   = 32;
%         options.numEpochs       = 25;   %0.3595, 0.2950
%         options.numEpochs       = 100;   %0.4440, 0.3700
%         options.numEpochs       = 200;   %0.4775, 0.3800
        options.numEpochs       = 500;   %0.4990, 0.4250 (4000 sec)
        options.displayProgress = true;
        
    case 11
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario3.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario5.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.0075;
        options.miniBatchSize   = 32;
%         options.numEpochs       = 100;   %0.69, 0.59
%         options.numEpochs       = 300;   %0.767, 0.68 (2300 sec)
        options.numEpochs       = 600;   % 0.7885, 0.685 (5211 sec)
        
        options.displayProgress = true;
        
    case 12
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario4.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.0005;
        options.miniBatchSize   = 32;
%         options.numEpochs       = 10;   %0.4000,0.4034
%         options.numEpochs       = 25;   %0.6664, 0.6649 (5222 sec)
%         options.numEpochs       = 50;   %0.6830, 0.6762 (10127 sec)
        options.numEpochs       = 100;   %0.6893, 0.6791 (19812 sec)
        
        options.displayProgress = true;
        
    case 13
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario6.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.0137;
        options.miniBatchSize   = 32;
%         options.numEpochs       = 10;   %0.8646, 0.8604 (1940 sec)
%         options.numEpochs       = 25;   %0.8711, 0.8641 (4879 sec)
%         options.numEpochs       = 50;   %0.8831, 0.8731 (9512 sec)
        options.numEpochs       = 100;   %0.8983, 0.8927 (19432 sec)
        
        options.displayProgress = true;
        
    case 14
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario7.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.0005;
        options.miniBatchSize   = 32;
        options.numEpochs       = 400;
        
        options.displayProgress = true;
        
    case 15
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario4.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.00043;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;
        
        options.displayProgress = true;
      
    case 16
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario9.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.015;
        options.miniBatchSize   = 32;
%         options.numEpochs       = 10;   %0.8676, 0.8662
        options.numEpochs       = 25;   %0.8762, 0.8705
        
        options.displayProgress = true;
        
    case 17
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario10.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.0031;
        options.miniBatchSize   = 32;
        options.numEpochs       = 100;  
        
        options.displayProgress = true;
        
    case 18
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario11.mat'];
        
        options.errorFunctionID = ErrorFunctionID.SquaredError;
        options.numSubSteps     = 1;
        options.eta             = 0.029;
        options.miniBatchSize   = 32;
        options.numEpochs       = 400;  
        
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
