%Familiarize with the Yale Faces data set
%
%Christopher Lum
%lum@uw.edu

%Version History
%06/01/23: Created

clear
clc
close all

ChangeWorkingDirectoryToThisLocation();

tic

%% User selections
expectedNumSubjects = 15;
expectedNumConditions = 11;

%% Load parameters
temp = load('YaleFacesData.mat');
faces               = temp.faces;
subjectNumberLabels = temp.subjectNumberLabels;
conditionLabels     = temp.conditionLabels;

%View image and label
for k=1:expectedNumSubjects
    idx = find(subjectNumberLabels==k);

    faces_k                 = faces(:,:,idx);
    subjectNumberLabels_k   = subjectNumberLabels(idx);
    conditionLabels_k       = conditionLabels(idx);

    figure('Name',['Subject = ',num2str(subjectNumberLabels_k(1))])
    for m=1:expectedNumConditions
       subplot(3,4,m)
       imshow(faces_k(:,:,m))
       title([conditionLabels_k(m)])
    end
end

MaximizeFigureAll();

toc
disp('DONE!')