%Train the Neural Network on the the MNIST dataset using multiple epochs
%and in batch mode (AKA varying parameters between training runs).
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/08/23: Created
%05/09/23: Loading initial condition NN (AKA warm start)

clear
clc
close all

tic

%% User selections
scenarioSelection = 4;

nodesPerLayer_cell      = {};
errorFunctionID_cell    = {};
numSubSteps_cell        = {};
eta_cell                = {};
miniBatchSize_cell      = {};
numEpochs_cell          = {};
displayProgress_cell    = {};
switch scenarioSelection
    case 1
        %Nominal case
        trainingDataFile = 'TrainingAndTestDataScenario1.mat';
        numConditions = 1;
        
%         %max accuracy 0.8718, 0.8837
%         for m=1:numConditions
%             nodesPerLayer_cell{end+1}   = [28*28 50 10];
%             errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
%             numSubSteps_cell{end+1}     = 1;
%             eta_cell{end+1}             = 0.17;
%             miniBatchSize_cell{end+1}   = 32;
%             numEpochs_cell{end+1}       = 20;
%             displayProgress_cell{end+1} = true;
%         end
        
%         %max accuracy 0.9140, 0.9168
%         for m=1:numConditions
%             nodesPerLayer_cell{end+1}   = [28*28 50 10];
%             errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
%             numSubSteps_cell{end+1}     = 1;
%             eta_cell{end+1}             = 0.05;
%             miniBatchSize_cell{end+1}   = 32;
%             numEpochs_cell{end+1}       = 100;
%             displayProgress_cell{end+1} = true;
%         end

%         %max accuracy 0.9193, 0.9231
%         for m=1:numConditions
%             nodesPerLayer_cell{end+1}   = [28*28 50 10];
%             errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
%             numSubSteps_cell{end+1}     = 1;
%             eta_cell{end+1}             = 0.035;
%             miniBatchSize_cell{end+1}   = 32;
%             numEpochs_cell{end+1}       = 20;
%             displayProgress_cell{end+1} = true;
%         end
%         
%         initialNeuralNetworkFile = 'NeuralNetworkOnly_ID01.mat';  %if non-empty this will ignore nodesPerLayer and initialize neural network based on this file

%         %max accuracy 0.5917, 0.6033
%         for m=1:numConditions
%             nodesPerLayer_cell{end+1}   = [28*28 50 25 10];
%             errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
%             numSubSteps_cell{end+1}     = 1;
%             eta_cell{end+1}             = 1.4;
%             miniBatchSize_cell{end+1}   = 32;
%             numEpochs_cell{end+1}       = 150;
%             displayProgress_cell{end+1} = true;
%         end
%         
%         initialNeuralNetworkFile = '';
        
        %max accuracy 0.6666, 0.6781
        for m=1:numConditions
            nodesPerLayer_cell{end+1}   = [28*28 50 25 10];
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = 0.4;
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 50;
            displayProgress_cell{end+1} = true;
        end
        
        initialNeuralNetworkFile = 'NeuralNetworkOnly_ID04.mat';  %if non-empty this will ignore nodesPerLayer and initialize neural network based on this file
        

    case 3
        %Vary eta
        trainingDataFile = 'TrainingAndTestDataScenario11.mat';
        numConditions = 10;
        
        etaVec = linspace(0.5,1.5,numConditions);
        for m=1:numConditions
            nodesPerLayer_cell{end+1}   = [28*28 50 10];
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 2;
            displayProgress_cell{end+1} = true;
        end
        
    case 4
        %Vary eta
        
        trainingDataFile = 'TrainingAndTestDataScenario1.mat';
        numConditions = 8;
        
        etaVec = linspace(0.1,10,numConditions);
        for m=1:numConditions
            nodesPerLayer_cell{end+1}   = [28*28 50 10];
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 2;
            displayProgress_cell{end+1} = true;
        end        
        
        initialNeuralNetworkFile = 'NeuralNetwork_784_50_10_ReLU_ReLU_Sigmoid.mat';
        
    case 101
        %Small set for testing
        trainingDataFile = 'TrainingAndTestDataScenario11.mat';
        numConditions = 8;
        
        etaVec = linspace(0.1,10,numConditions);
        for m=1:numConditions
            nodesPerLayer_cell{end+1}   = [28*28 50 25 10];
            errorFunctionID_cell{end+1} = ErrorFunctionID.SquaredError;
            numSubSteps_cell{end+1}     = 1;
            eta_cell{end+1}             = etaVec(m);
            miniBatchSize_cell{end+1}   = 32;
            numEpochs_cell{end+1}       = 5;
            displayProgress_cell{end+1} = true;
        end
        
        initialNeuralNetworkFile = '';  %if non-empty this will ignore nodesPerLayer and initialize neural network based on this file
        
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
[M,N] = size(A);
nu = M*N;
no = 10;
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

%% Train network in batch mode
disp('Training network in batch mode')

%Create an array of options so structure is compatible with parfor
options_cell = {};
for m=1:numConditions
    %Format as options
    nodesPerLayer   = nodesPerLayer_cell{m};
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
    nodesPerLayer   = nodesPerLayer_cell{m};
    errorFunctionID = errorFunctionID_cell{m};
    numSubSteps     = numSubSteps_cell{m};
    eta             = eta_cell{m};
    miniBatchSize   = miniBatchSize_cell{m};
    numEpochs       = numEpochs_cell{m};
    displayProgress = displayProgress_cell{m};
    
    if(isempty(initialNeuralNetworkFile))
        nn = NeuralNetwork(nodesPerLayer);
        nn.SetActivationFunctionAtAllLayers(ActivationFunctionID.Sigmoid);
        d=1
    else
        %Load from initialNeuralNetworkFile (AKA warm start)
        disp(['Loading neural network from ',initialNeuralNetworkFile,' and ignoring settings in nodesPerLayer_cell'])
        temp = load(initialNeuralNetworkFile);
        nn = temp.nn;        
    end
    
    %Format as options
%     options.errorFunctionID = errorFunctionID;
%     options.numSubSteps     = numSubSteps;
%     options.eta             = eta;
%     options.miniBatchSize   = miniBatchSize;
%     options.numEpochs       = numEpochs;
%     options.displayProgress = displayProgress;
        
    [E_data,norm_gradient_data] = nn.TrainNeuralNetworkMultiEpoch(U_data,D_data,options_cell{m});
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
    title(['Condition #',num2str(m)])
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
    outputFile = ['TrainedNetwork_scenario',num2str(scenarioSelection),'_condition',num2str(m),'.mat'];
%     saveVars = {
%         'nn'
%         'options'
%         
%         'TrainingSetImages'
%         'TrainingSetLabels'
%         'TestSetImages'
%         'TestSetLabels'
%         
%         'E_data'
%         'norm_gradient_data'
%         };
%     s = SaveVarsString(outputFile,saveVars);
%     eval(s);

parsave(...
    outputFile,...
    nn,options,...
    TrainingSetImages,TrainingSetLabels,TestSetImages,TestSetLabels,...
    E_data,norm_gradient_data)

    disp(['Saved to ',outputFile])
    
end

toc
disp('DONE!')
