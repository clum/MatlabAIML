function [] = parsave(...
    outputFile,...
    nn,options,...
    trainingDataFile,initialNeuralNetworkFile,...
    E_data,norm_gradient_data)

%Typically you cannot call 'save' inside a parfor loop.  As such, we need
%to use a separate function to enable saving within a parfor loop

%Version History
%05/??/23: Created
%05/22/23: Updated

save(outputFile,...
    'nn','options',...
    'trainingDataFile','initialNeuralNetworkFile',...
    'E_data','norm_gradient_data')