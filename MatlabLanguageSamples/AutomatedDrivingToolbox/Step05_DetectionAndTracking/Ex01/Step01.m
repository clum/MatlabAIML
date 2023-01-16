%Follow example at https://www.mathworks.com/help/driving/ug/train-a-deep-learning-vehicle-detector.html
%
%Data located at C:\Program Files\MATLAB\R2021b\examples\deeplearning_shared\data
%
%Christopher Lum
%lum@uw.edu

%Version History
%01/14/23: Created
%01/15/23: Updated

clear
clc
close all

%% Download pre-trained detector if it is not already on hard disk
doTrainingAndEval = false;
if ~doTrainingAndEval && ~exist('fasterRCNNResNet50EndToEndVehicleExample.mat','file')
    disp('Downloading pretrained detector (118 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/fasterRCNNResNet50EndToEndVehicleExample.mat';
    websave('fasterRCNNResNet50EndToEndVehicleExample.mat',pretrainedURL);
end

%% Load ground truth data
temp = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = temp.vehicleDataset;

%% Split the data set into a training set for training the detector and a test set for evaluating the detector. 
%Select 60% of the data for training. Use the rest for evaluation.
rng(0)
shuffledIdx = randperm(height(vehicleDataset));
idx = floor(0.6 * height(vehicleDataset));
trainingDataTbl = vehicleDataset(shuffledIdx(1:idx),:);
testDataTbl = vehicleDataset(shuffledIdx(idx+1:end),:);

%% Use imageDatastore and boxLabelDatastore to create datastores for loading the image and label data during training and evaluation.
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

%Combine image and box label datastores.
trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest,bldsTest);

%Display one of the training images and box labels.
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
% annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Create Faster R-CNN Detection Network 
%specify the network input size. 
inputSize = [224 224 3];

%Next, use estimateAnchorBoxes to estimate anchor boxes based on the size
%of objects in the training data. To account for the resizing of the images
%prior to training, resize the training data for estimating anchor boxes.
%Use transform to preprocess the training data, then define the number of
%anchor boxes and estimate the anchor boxes.
preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 4;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

disp('DONE!')