%Attempt to classify images based on projections onto eigenfaces.
%
%Christopher Lum
%lum@uw.edu

%Version History
%07/19/23: Created

clear
clc
close all

tic

%% User selections
scenarioSelection = 2;

switch scenarioSelection
    case 1
        trainingDataFile    = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario1.mat'];
        PCAFile             = 'PCAScenario1.mat';
        r_data              = [5 25 50 75 125];     %orders of approximations
        idx_train           = 34;   %index of face from the training set to reconstruct
        idx_test            = 20;   %index of face from the test set to reconstruct

    case 2
        trainingDataFile    = [ReturnPathStringNLevelsUp(1),'\Step02_PreprocessDataset\TrainingAndTestDataScenario2.mat'];
        PCAFile             = 'PCAScenario2.mat';
        r_data              = [5 25 60 85 140];     %orders of approximations
        idx_train           = 3;   %index of face from the training set to reconstruct
        idx_test            = 3;   %index of face from the test set to reconstruct

    otherwise
        error('')
end

%% Load data
%trainingDataFile
temp = load(trainingDataFile);
disp(['Loading from ',trainingDataFile])

faces_train                 = temp.faces_train;
subjectNumberLabels_train   = temp.subjectNumberLabels_train;
conditionLabels_train       = temp.conditionLabels_train;

faces_test                  = temp.faces_test;
subjectNumberLabels_test    = temp.subjectNumberLabels_test;
conditionLabels_test        = temp.conditionLabels_test;

%PCAFile
temp = load(PCAFile);
disp(['Loading from ',PCAFile])

X       = temp.X;
xbar    = temp.xbar;
B       = temp.B;
U       = temp.U;
S       = temp.S;
V       = temp.V;

[M,N] = size(faces_train(:,:,1));

%% Reconstruct an image using the basis spanned by U
%Use an example from the training set
x_train = B(:,idx_train);

%Extract an example from the test set (remember to mean-center the data)
x_test  = reshape(double(faces_test(:,:,idx_test)),M*N,1) - xbar;


figh_faces_r_train = figure;
subplot(2,3,1)
imshow(uint8(reshape(ScaleValues(x_train,0,255),M,N)))
title(StringWithUnderscoresForPlot('B(idx_train)'))

figh_faces_r_test = figure;
subplot(2,3,1)
imshow(uint8(reshape(ScaleValues(x_test,0,255),M,N)))
title(StringWithUnderscoresForPlot('reshaped and mean centered x-test'))

for k=1:length(r_data)
    r = r_data(k);

    %Create rank-r matrix of eigenfaces
    Utilde = U(:,1:r);

    %Project x_train onto basis spanned by Utilde (AKA dot product x_train with
    %each column of Utilde).  This effectively evaluates how aligned x_train is
    %with each column (AKA eigenface) of Utilde
    alpha_train = Utilde'*x_train;
    alpha_test  = Utilde'*x_test;

    %Create vector that is a linear combination of Utilde with each column
    %weighted by alpha
    xr_train    = Utilde*alpha_train;
    xr_test     = Utilde*alpha_test;

    figure(figh_faces_r_train);
    subplot(2,3,k+1)
    imshow(uint8(reshape(ScaleValues(xr_train,0,255),M,N)))
    title(['r = ',num2str(r)])

    figure(figh_faces_r_test);
    subplot(2,3,k+1)
    imshow(uint8(reshape(ScaleValues(xr_test,0,255),M,N)))
    title(['r = ',num2str(r)])
end

toc
disp('DONE!')
