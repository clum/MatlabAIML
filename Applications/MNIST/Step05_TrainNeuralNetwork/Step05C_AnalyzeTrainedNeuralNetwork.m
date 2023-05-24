%Analyze the trained the neural network
%
%Examine saturation at each layer
%
%Christopher Lum
%lum@uw.edu

%Version History
%05/19/23: Created
%05/24/23: Refactored workflow

clear
clc
close all

tic

%% User selections
trainedNetworkFile = 'TrainedNeuralNetworkScenario1.mat';

error('START HERE!!!')
%% Load data
temp = load(trainedNetworkFile);
nn                  = temp.nn;
options             = temp.options;

warning('TEMP: Refactor')
temp2 = load('NeuralNetworkOnly_ID03.mat')
nn = temp2.nn;

TrainingSetImages   = temp.TrainingSetImages;
TrainingSetLabels   = temp.TrainingSetLabels;
TestSetImages       = temp.TestSetImages;
TestSetLabels       = temp.TestSetLabels;

E_data              = temp.E_data;
norm_gradient_data  = temp.norm_gradient_data;

%% Assess accuracy 

%Assess against testing data
Y_test  = [];
E_test  = [];
classifiedCorrectTest   = 0;
classifiedIncorrectTest = 0;
y_layer_data = {};
% for k=1:length(TestSetLabels)
for k=1:10

    A       = TestSetImages(:,:,k);
    label   = TestSetLabels(k);
       
    %Reshape into an input vector U (stack each column on top of one
    %another) and convert to double
    [M,N] = size(A);
    U = double(reshape(A,M*N,1));
    
    %Convert the label to a vector d
    D = LabelToVector(label);
    
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

%% Visualize results
%Histogram of layer outputs to understand saturation
figure
numBins = 20;
for m=1:nn.NumLayers
    subplot(nn.NumLayers,1,m)
    hist(y_layer_data{m},numBins)
    ylabel(['Layer ',num2str(m)'])
end

toc
disp('DONE!')