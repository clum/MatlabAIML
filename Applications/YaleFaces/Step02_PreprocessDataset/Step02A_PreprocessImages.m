%Preprocess images.
%
%Christopher Lum
%lum@uw.edu

%Version History
%06/17/23: Created

clear
clc
close all

tic

%% User selections
%1 = no pre-processing
%2 = cropped to only include face
scenarioSelection = 1;

%% Load data
cwd = pwd;

step01Directory = [fileparts(cwd),'\Step01_ObtainDataset'];
cd(step01Directory);

temp = load('YaleFacesData.mat');
faces               = temp.faces;
subjectNumberLabels = temp.subjectNumberLabels;
conditionLabels     = temp.conditionLabels;

cd(cwd);

%% Crop
%Define cropping
switch scenarioSelection
    case 1
        %no cropping
        facesCropped = faces;

    case 2
        width = 300;
        height = 200;
        for k=1:expectedNumSubjects
            switch k
                case {3}

                otherwise
                    %Use standard cropping
                    A = faces(:,:,k);
                    imshow(A)
            end
        end
    
    otherwise
        error('Supported scenarioSelection')
end

%assemble pre-processed data set
facePreprocessed = facesCropped;

%% Save data
outputFileName = ['PreprocessedDatasetScenario',num2str(scenarioSelection)];
save(outputFileName,'facePreprocessed','subjectNumberLabels','conditionLabels')
disp(['Saved to ',outputFileName])

toc
disp('DONE!')