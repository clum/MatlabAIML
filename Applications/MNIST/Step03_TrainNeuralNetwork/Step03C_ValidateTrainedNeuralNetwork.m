%Validate the trained the neural network
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
% trainedNetworkFile = 'TrainedNetwork_scenario1.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario2.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario3.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario4.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario5.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario6.mat';

% trainedNetworkFile = 'TrainedNetwork_scenario99.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario100.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario101.mat';
% trainedNetworkFile = 'TrainedNetwork_scenario102.mat';
trainedNetworkFile = 'TrainedNetwork_scenario103.mat';

%% Load data
temp = load(trainedNetworkFile);
nn                  = temp.nn;
errorFunctionID     = temp.errorFunctionID;
numSteps            = temp.numSteps;
numSubSteps         = temp.numSubSteps;
eta                 = temp.eta;

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
    
    E_train(end+1) = NeuralNetwork.Error(Y,D,errorFunctionID);
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

    E_test(end+1) = NeuralNetwork.Error(Y,D,errorFunctionID);
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
disp('Accuracy on training data')
classifiedCorrectTrain/length(TrainingSetLabels)

disp('Accuracy on test data')
classifiedCorrectTest/length(TestSetLabels)

figure
subplot(3,1,1)
plot(E_data)
grid on
legend('E (during training process)')

subplot(3,1,2)
plot(E_train)
grid on
legend('E (on training set)')

subplot(3,1,3)
plot(E_test)
grid on
legend('E (on test set)')

toc
disp('DONE!')