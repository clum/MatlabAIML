%Validate the trained the neural network
%
%Christopher Lum
%lum@uw.edu

%Version History
%04/22/23: Created
%04/23/23: Continued working
%05/06/23: Continued working
%05/24/23: Refactored workflow
%06/14/23: Adding scenarios
%07/29/23: Adding scenarios
%07/30/23: Adding gradient plot

clear
clc
close all

tic

%% User selections
% trainedNetworkFile = 'TrainedNeuralNetworkScenario1.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario2.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario3.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario4.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario5.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario6.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario7.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario8.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario9.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario10.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario11.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario12.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario13.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario14.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario15.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario16.mat';
% trainedNetworkFile = 'TrainedNeuralNetworkScenario17.mat';
trainedNetworkFile = 'TrainedNeuralNetworkScenario18.mat';

%% Load data
disp(['Loading ',trainedNetworkFile])

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

%% Assess accuracy
[ns_train,~]    = size(U_train);
[ns_test,~]     = size(U_test);

%Assess against training data
Y_train = [];
E_train = [];
classifiedCorrectTrain      = 0;
classifiedIncorrectTrain    = 0;
for k=1:ns_train
    U = U_train(k,:)';
    D = D_train(k,:)';

    Y = nn.ForwardPropagate(U);

    E_train(end+1) = NeuralNetwork.Error(Y,D,options.errorFunctionID);
    Y_train(:,k) = Y;

    %How did network classify the digit
    idx = find(D==max(D));
    label = idx - 1;

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
for k=1:ns_test
    U = U_test(k,:)';
    D = D_test(k,:)';

    Y = nn.ForwardPropagate(U);

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
end

%% Visualize training results
accuracyTrain = classifiedCorrectTrain/ns_train;
accuracyTest = classifiedCorrectTest/ns_test;

disp(['Accuracy on training data = ',num2str(accuracyTrain)])
disp(['Accuracy on test data     = ',num2str(accuracyTest)])
disp(' ')
disp(['Average error on training data: ',num2str(sum(E_train)/length(E_train))])
disp(['Average error on test data:     ',num2str(sum(E_test)/length(E_test))])
disp(' ')

%Filter E_data to get a better idea of system converged during training
averageWindow = 20;
E_data_average              = NaN(1,length(E_data));
norm_gradient_data_average  = NaN(1,length(norm_gradient_data));
for k=averageWindow:length(E_data)
    idxStart    = k-averageWindow+1;
    idxEnd      = idxStart+averageWindow-1;
    E_data_average(k)               = sum(E_data(idxStart:idxEnd))/averageWindow;
    norm_gradient_data_average(k)   = sum(norm_gradient_data(idxStart:idxEnd))/averageWindow;
end

figure
ax = [];
ax(end+1) = subplot(2,1,1);
hold on
plot(E_data,'DisplayName',StringWithUnderscoresForPlot('E_data'))
plot(E_data_average,'LineWidth',3,'DisplayName',StringWithUnderscoresForPlot('E_data_average'))
grid on
title('E (filtered) during training process')
legend()

ax(end+1) = subplot(2,1,2);
hold on
plot(norm_gradient_data,'DisplayName',StringWithUnderscoresForPlot('norm_gradient_data'))
plot(norm_gradient_data_average,'LineWidth',3,'DisplayName',StringWithUnderscoresForPlot('norm_gradient_data_average'))
grid on
title('Norm of gradient (filtered) during training process')
legend()

linkaxes(ax,'x')

%% Save the NeuralNetwork by itself in a file
outputFile = 'NeuralNetworkOnly.mat';
save(outputFile,'nn')
disp(['Saved NeuralNetwork variable only to ',outputFile])

toc
disp('DONE!')