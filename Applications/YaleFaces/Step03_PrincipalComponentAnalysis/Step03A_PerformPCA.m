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
title('aveFaceMatrix')

%Subtract xbar from each column of X.  This effectively subtacts the
%average face from each example face.
Xbar = xbar*ones(1,ns_train);
B = X - Xbar;

figure
idx = 60;

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
covB = (1/(M*N))*(B')*B;

figure;
covBs = ScaleValues(covB,0,255);
imshow(uint8(covBs))
title('covB')

%perform SVD
[U,S,V] = svd(B,'econ');

%Plot singular values
figure
semilogy(diag(S))
xlabel('r')
ylabel('S_r')
grid on

%% Check results
figure
eigenfacesIndices = [1 2 5 50 100 135];

for m=1:length(eigenfacesIndices)
    k = eigenfacesIndices(m);
    u_k = U(:,k);
    u_k_s = ScaleValues(u_k,0,255);
    u_k_s_matrix = reshape(u_k_s,M,N);
    subplot(2,3,m)
    imshow(uint8(u_k_s_matrix))
    title(['u_{',num2str(eigenfacesIndices(m)),'}']);
end

%% Save data
outputFile = ['PCAScenario',num2str(scenarioSelection),'.mat'];
saveVars = {
    'X'
    'B'
    'U'
    'S'
    'V'
    };
s = SaveVarsString(outputFile,saveVars);
eval(s);
disp(['Saved to ',outputFile])

toc
disp('DONE!')
