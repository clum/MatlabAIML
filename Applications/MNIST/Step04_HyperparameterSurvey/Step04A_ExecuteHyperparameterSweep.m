%Train the NeuralNetwork using multiple epochs and in batch mode (AKA
%varying parameters between training runs).  This is useful for examining
%which hyperparameters work well.
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/08/23: Created
%05/09/23: Loading initial condition NN (AKA warm start)
%05/22/23: Refactored workflow
%05/24/23: Continued refactor
%06/11/23: Added scenarioSelection = 4
%06/14/23: Added scenarioSelection = 7
%07/29/23: Changed scenarioSelection = 7 to use a small training set
%07/31/23: Added scenarioSelection = 9
%08/02/23: Added scenarioSelection = 10
%08/04/23: Added scenarioSelection = 11
%08/05/23: Added scenarioSelection = 12

clear
clc
close all

ChangeWorkingDirectoryToThisLocation();

tic

%% User selections
scenarioSelection = 15;

errorFunctionID_cell    = {};
numSubSteps_cell        = {};
eta_cell                = {};
miniBatchSize_cell      = {};
numEpochs_cell          = {};
displayProgress_cell    = {};
switch scenarioSelection
    case 1
        %Small case for testing
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario3.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario1.mat'];
        
        numConditions = 2;
        
        etaVec = linspace(0.5,1.5,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 1;
            displayProgress_cell{end+1} = true;
        end
        
    case 2
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario1.mat'];
        
        numConditions = 8;
        
        etaVec = linspace(0.01,0.5,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 5;
            displayProgress_cell{end+1} = true;
        end
        
    case 3
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario2.mat'];
        
        numConditions = 8;
        
        etaVec = linspace(0.005,0.5,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 25;
            displayProgress_cell{end+1} = true;
        end

    case 4
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario3.mat'];
        
        numConditions = 8;
        
        etaVec = linspace(0.0005,0.04,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 25;
            displayProgress_cell{end+1} = true;
        end
        
    case 5
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario3.mat'];
        
        numConditions = 8;
        
        etaVec = linspace(0.0005,0.04,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 25;
            displayProgress_cell{end+1} = true;
        end
        
    case 6
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario5.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario3.mat'];
        
        numConditions = 8;
        
        etaVec = linspace(0.0005,0.04,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 25;
            displayProgress_cell{end+1} = true;
        end
        
    case 7
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario3.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario4.mat'];
        
        numConditions = 8;
        
%         etaVec = linspace(0.0005,0.5,numConditions);
        etaVec = linspace(0.00001,0.0005,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 5;
            displayProgress_cell{end+1} = true;
        end
        
    case 8
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario3.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario5.mat'];
        
        numConditions = 4*4;
        
        etaVec = linspace(0.0001,0.01,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 150;
            displayProgress_cell{end+1} = true;
        end
        
    case 9
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario4.mat'];
        
        numConditions = 8;
        
        etaVec = linspace(0.00001,0.0005,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 50;
            displayProgress_cell{end+1} = true;
        end
        
    case 10
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario6.mat'];
        
        numConditions = 4*4;
        
        etaVec = linspace(0.0005,0.05,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 50;
            displayProgress_cell{end+1} = true;
        end
        
    case 11
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario7.mat'];
        
        numConditions = 2*4;
        
        etaVec = linspace(0.0005,0.005,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 25;
            displayProgress_cell{end+1} = true;
        end
        
    case 12
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario4.mat'];
        
        numConditions = 2*4;
        
        etaVec = linspace(0.00001,0.0005,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 10;
            displayProgress_cell{end+1} = true;
        end
        
    case 13
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario9.mat'];
        
        numConditions = 2*4;
        
%         etaVec = linspace(0.0001,0.005,numConditions);  %0.8772, 0.8651
        etaVec = linspace(0.001,0.05,numConditions);  %0.8672,0.8662
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 10;
            displayProgress_cell{end+1} = true;
        end        
        
    case 14
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario10.mat'];
        
        numConditions = 2*4;
        
%         etaVec = linspace(0.001,0.05,numConditions);
        etaVec = linspace(0.0003,0.02,numConditions);
        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 10;
            displayProgress_cell{end+1} = true;
        end        
        
    case 15
        %Vary eta
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario4.mat'];
        initialNeuralNetworkFile    = [ReturnPathStringNLevelsUp(1),'\Step03_SetupNeuralNetwork\NeuralNetworkScenario10.mat'];
        
        numConditions = 2*4;
        
%         etaVec = linspace(0.0001,0.005,numConditions);      %0.8841, 0.8761 (best eta = 0.0015)
        etaVec = linspace(0.001,0.05,numConditions);      % 0.9058, 0.9047 (best eta = 0.0290)

        for m=1:numConditions
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
%             numEpochs_cell{end+1}       = 50;
            numEpochs_cell{end+1}       = 25;
            displayProgress_cell{end+1} = true;
        end        
                
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

%% Train network in batch mode
disp('Training network in batch mode')

%Create an array of options so structure is compatible with parfor
options_cell = {};
for m=1:numConditions
    %Format as options
    errorFunctionID = errorFunctionID_cell{m};
    numSubSteps     = numSubSteps_cell{m};
    eta             = eta_cell{m};
    miniBatchSize   = miniBatchSize_cell{m};
    numEpochs       = numEpochs_cell{m};
    displayProgress = displayProgress_cell{m};
    
    options.errorFunctionID = errorFunctionID;
    options.numSubSteps     = numSubSteps;
    options.eta             = eta;
    options.miniBatchSize   = miniBatchSize;
    options.numEpochs       = numEpochs;
    options.displayProgress = displayProgress;
    
    options_cell{end+1} = options;
end

parfor m=1:numConditions
    disp(['Now training on condition ',num2str(m),' out of ',num2str(numConditions)])
    
    %Get hyperparameters for this condition
    errorFunctionID = errorFunctionID_cell{m};
    numSubSteps     = numSubSteps_cell{m};
    eta             = eta_cell{m};
    miniBatchSize   = miniBatchSize_cell{m};
    numEpochs       = numEpochs_cell{m};
    displayProgress = displayProgress_cell{m};
    
    %Load from initialNeuralNetworkFile
    disp(['Loading neural network from ',initialNeuralNetworkFile])
    temp = load(initialNeuralNetworkFile);
    nn = temp.nn;
    
    [E_data,norm_gradient_data] = nn.TrainNeuralNetworkMultiEpoch(U_train,D_train,options_cell{m});
    
    %% Save network
    outputFile = ['HyperparameterSweepResultsScenario',num2str(scenarioSelection),'_condition',num2str(m),'.mat'];
    
    parsave(...
        outputFile,...
        nn,options,...
        trainingDataFile,initialNeuralNetworkFile,...
        E_data,norm_gradient_data)
    
    disp(['Saved to ',outputFile])    
end

toc
disp('DONE!')
