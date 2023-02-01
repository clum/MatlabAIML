%Familiarize with the MNIST data set
%
%Christopher Lum
%lum@uw.edu

%Version History
%01/29/23: Created

clear
clc
close all

tic

%% User selections
M       = 6;            %num rows in subplot
N       = 6;            %num cols in subplot
set     = 'testing';    %'training' or 'testing'

%% Load parameters
cwd = pwd;

step01Directory = [fileparts(cwd),'\Step01_ObtainDataset'];
cd(step01Directory);

temp = load('MNISTInfo.mat');
MNISTInfo = temp.MNISTInfo;

temp2 = load('MNISTData.mat');
TrainingSetImages   = temp2.TrainingSetImages;
TrainingSetLabels   = temp2.TrainingSetLabels;
TestSetImages       = temp2.TestSetImages;
TestSetLabels       = temp2.TestSetLabels;

cd(cwd);

%View image and label
figure
k = 1;
for m=1:M
    for n=1:N
        subplot(M,N,k)
        
        %Chose a random index
        switch set
            case 'training'
                numImages = length(TrainingSetLabels);
                
            case 'testing'
                numImages = length(TestSetLabels);
                
            otherwise
                error('Unsupported set')
        end
        
        idx = floor(rand(1)*numImages);
        
        %Chose which set of data
        switch set
            case 'training'
                A       = TrainingSetImages(:,:,idx);
                label   = TrainingSetLabels(idx);
        
            case 'testing'
                A       = TestSetImages(:,:,idx);
                label   = TestSetLabels(idx);
                
            otherwise
                error('Unsupported set')
        end
        
        imshow(A)
        title(['idx ',num2str(idx),',  Label ' ,num2str(label)])
        
        k = k + 1;
    end
end

toc
disp('DONE!')