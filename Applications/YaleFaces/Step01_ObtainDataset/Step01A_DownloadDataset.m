%Obtain and unzip the Yale Faces data set.
%
%See https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database for
%more info
%
%Christopher Lum
%lum@uw.edu

%Version History
%06/24/23: Created

clear
clc
close all

tic

%% User selections
downloadBaseURL = 'https://faculty.washington.edu/lum/DataSets';
fileNameFaces   = 'YaleFaces.zip';

%% Download data
if(~exist(fileNameFaces))
    disp(['Downloading ',fileNameFaces]);
    websave(fileNameFaces,[downloadBaseURL,'/',fileNameFaces]);
end

%% Unzip data
disp('Unzipping data')
[outputFolderFaces,~] = SeperateFileNameAndExtension(fileNameFaces);
if(~exist(outputFolderFaces))
    unzip(fileNameFaces,outputFolderFaces)
end

toc
disp('DONE!')