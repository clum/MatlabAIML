%Perform principal component analysis
%
%Christopher Lum
%lum@uw.edu

%Version History
%06/18/23: Created

clear
clc
close all

tic

%% User selections
scenarioSelection = 1;

switch scenarioSelection
    case 1
        trainingDataFile            = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        
%         options.errorFunctionID = ErrorFunctionID.SquaredError;
%         options.numSubSteps     = 1;
%         options.eta             = 0.22;
%         options.miniBatchSize   = 32;
%         options.numEpochs       = 5;
%         options.displayProgress = true;
                
    otherwise
        error('')
end

%% Load data
temp = load(trainingDataFile);
disp(['Loading from ',trainingDataFile])

faces_train                 = temp.faces_train;
subjectNumberLabels_train   = temp.subjectNumberLabels_train;
conditionLabels_train       = temp.conditionLabels_train;

faces_test                  = temp.faces_test;
subjectNumberLabels_test   = temp.subjectNumberLabels_test;
conditionLabels_test       = temp.conditionLabels_test;

%% Perform PCA
disp('Performing PCA')

ns_train = length(subjectNumberLabels_train);

%stack all pictures columnwise
[M,N] = size(faces_train(:,:,1));

X = zeros(M*N,ns_train);
for k=1:ns_train
    A = faces_train(:,:,k);
    X(:,k) = reshape(A,M*N,1);
end

%Compute the row average (AKA average face)
xbar = zeros(M*N,1);
for k=1:M*N
    row_ave = sum(X(k,:))/ns_train;
    xbar(k) = row_ave;
end

xbar2 = mean(X,2);
assert(max(abs(xbar-xbar2))==0)

figure
aveFaceMatrix = uint8(reshape(xbar,M,N));
imshow(aveFaceMatrix)

%Subtract xbar from each column of X.  This effectively subtacts the
%average face from each example face.
Xbar = xbar*ones(1,ns_train);
B = X - Xbar;

figure
idx = 50;

subplot(1,2,1)
imshow(uint8(reshape(X(:,idx),M,N)));
title('X')

subplot(1,2,2)
imshow(uint8(reshape(B(:,idx),M,N)));
title('B')

%Check we get the same answer if we methodically subtract off each column
B2 = zeros(size(X));
for k=1:ns_train
    B2(:,k) = X(:,k) - xbar;
end

assert(max(max(abs(B-B2)))==0);

%Compute covariance matrix of columns of B
covB = (1/(M*N))*B'*B;

figure;
plot(covB(:,1))

%perform SVD
[U,S,V] = svd(X,'econ');

%% Check results
u1 = U(:,1);


%% Save network
% outputFile = ['TrainedNeuralNetworkScenario',num2str(scenarioSelection),'.mat'];
% saveVars = {
%     'nn'
%     'options'
%     
%     'trainingDataFile'
%     'initialNeuralNetworkFile'
%     
%     'E_data'
%     'norm_gradient_data'
%     };
% s = SaveVarsString(outputFile,saveVars);
% eval(s);
% disp(['Saved to ',outputFile])

toc
disp('DONE!')
