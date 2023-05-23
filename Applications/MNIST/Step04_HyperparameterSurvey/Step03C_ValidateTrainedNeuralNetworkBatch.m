%Validate the trained the neural network
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/08/23: Created based on previous version

clear
clc
close all

ChangeWorkingDirectoryToThisLocation();

tic

%% User selections
scenarioSelection = 4;

%% Load data
%Find all files associated with this scenarioSelection
allFiles = dir2(pwd,'only_files');
expressionPrefix = ['TrainedNetwork_scenario',num2str(scenarioSelection),'_condition'];
expression = [expressionPrefix,'*'];
indices = ~cellfun(@isempty,regexp(allFiles,expression));

files = allFiles(indices);
numConditions = length(files);

%Sort in order of condition number
for m=1:numConditions
    file = files{m};
    
    idxStart = length(expressionPrefix)+1;
    idxEnd = find(file=='.')-1;
    
    conditionNumberStr = file(idxStart:idxEnd);
    conditionNumber = str2num(conditionNumberStr);
    
    conditionNumberVec(m,1) = conditionNumber;
end

T = table(files,conditionNumberVec);
T = sortrows(T,'conditionNumberVec');

filesSorted = T.files;

accuracyTrain_data  = [];
accuracyTest_data   = [];
for m=1:numConditions
    trainedNetworkFile = filesSorted{m};
    disp(['Loading ',trainedNetworkFile])
    
    temp = load(trainedNetworkFile);
    nn                  = temp.nn;
    options             = temp.options;
    
    TrainingSetImages   = temp.TrainingSetImages;
    TrainingSetLabels   = temp.TrainingSetLabels;
    TestSetImages       = temp.TestSetImages;
    TestSetLabels       = temp.TestSetLabels;
    
    E_data              = temp.E_data;
    norm_gradient_data  = temp.norm_gradient_data;
    
    %% Assess accuracy
    %Assess against training data
    Y_train = [];
    E_train = [];
    classifiedCorrectTrain      = 0;
    classifiedIncorrectTrain    = 0;
    for k=1:length(TrainingSetLabels)
        A       = TrainingSetImages(:,:,k);
        label   = TrainingSetLabels(k);
        
        %Reshape into an input vector U (stack each column on top of one
        %another) and convert to double
        [M,N] = size(A);
        U = double(reshape(A,M*N,1));
        
        %Convert the label to a vector d
        D = LabelToVector(label);
        
        Y = nn.ForwardPropagate(U);
        
        E_train(end+1) = NeuralNetwork.Error(Y,D,options.errorFunctionID);
        Y_train(:,k) = Y;
        
        %How did network classify the digit
        idx = find(Y==max(Y));
        labelClassified = idx - 1;
        
        if(labelClassified==label)
            classifiedCorrectTrain = classifiedCorrectTrain + 1;
        else
            classifiedIncorrectTrain = classifiedIncorrectTrain + 1;
        end
    end
    
    %Assess against testing data
    Y_test  = [];
    E_test  = [];
    classifiedCorrectTest   = 0;
    classifiedIncorrectTest = 0;
    for k=1:length(TestSetLabels)
        A       = TestSetImages(:,:,k);
        label   = TestSetLabels(k);
        
        %Reshape into an input vector U (stack each column on top of one
        %another) and convert to double
        [M,N] = size(A);
        U = double(reshape(A,M*N,1));
        
        %Convert the label to a vector d
        D = LabelToVector(label);
        
        Y = nn.ForwardPropagate(U);
        
        E_test(end+1) = NeuralNetwork.Error(Y,D,options.errorFunctionID);
        Y_test(:,k)  = Y;
        
        %How did network classify the digit
        idx = find(Y==max(Y));
        labelClassified = idx - 1;
        
        if(labelClassified==label)
            classifiedCorrectTest = classifiedCorrectTest + 1;
        else
            classifiedIncorrectTest = classifiedIncorrectTest + 1;
        end
    end
    
    %% Visualize training results
    accuracyTrain = classifiedCorrectTrain/length(TrainingSetLabels);
    accuracyTest = classifiedCorrectTest/length(TestSetLabels);
    
    disp(['Accuracy on training data = ',num2str(accuracyTrain)])
    disp(['Accuracy on test data     = ',num2str(accuracyTest)])
    
    accuracyTrain_data(end+1,1) = accuracyTrain;
    accuracyTest_data(end+1,1)  = accuracyTest;
    
    %Filter E_data to get a better idea of system converged during training
    averageWindow = 20;
    E_data_average = NaN(1,length(E_data));
    for k=averageWindow:length(E_data)
        idxStart    = k-averageWindow+1;
        idxEnd      = idxStart+averageWindow-1;
        E_data_average(k) = sum(E_data(idxStart:idxEnd))/averageWindow;
    end
    
    if(m==1)
        figh_E_average = figure;
    else
        figure(figh_E_average)
    end
    
    hold on
    plot(E_data_average,'DisplayName',['Condition ',num2str(m)])
    grid on
    title('E (filtered) during training process')
    legend()
    
end

conditionBestTrain = find(accuracyTrain_data==max(accuracyTrain_data));
conditionBestTest = find(accuracyTest_data==max(accuracyTest_data));

bestAccuracyTrain = accuracyTrain_data(conditionBestTrain);
bestAccuracyTest  = accuracyTest_data(conditionBestTest);

disp(['Best Train = ',num2str(bestAccuracyTrain)])
disp(['Best Test  = ',num2str(bestAccuracyTest)])

conditionNumbers = [1:1:numConditions];
figh_accuracy = figure;
hold on
plot(conditionNumbers,accuracyTrain_data,'b-','DisplayName','Train')
plot(conditionBestTrain,bestAccuracyTrain,'bx','DisplayName',['Best Train = ',num2str(bestAccuracyTrain)])
plot(conditionNumbers,accuracyTest_data,'r-','DisplayName','Test')
plot(conditionBestTest,bestAccuracyTest,'rx','DisplayName',['Best Test = ',num2str(bestAccuracyTest)])

grid on
xlabel('Condition Number')
ylabel('Accuracy')
legend()

%% Save the NeuralNetwork by itself in a file
trainedNetworkFile = filesSorted{conditionBestTest};
disp(['Best accuracy in test set taken from ',trainedNetworkFile])

temp = load(trainedNetworkFile);
nn                  = temp.nn;

outputFile = 'NeuralNetworkOnly.mat';
save(outputFile,'nn')
disp(['Saved NeuralNetwork variable only to ',outputFile])

toc
disp('DONE!')