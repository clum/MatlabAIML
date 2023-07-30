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
scenarioSelection       = 2;

%Display results of cropping
displayCroppingIndices  = [1:11:15*11] + 1;

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
expectedNumSubjects     = length(unique(subjectNumberLabels));
expectedNumConditions   = length(unique(conditionLabels));

for n=1:length(subjectNumberLabels)
    subjectNumber = subjectNumberLabels(n);

    %Define cropping
    switch scenarioSelection
        case 1
            %no cropping
            xMin = 1;
            yMin = 1;
            [height,width,~] = size(faces);
            
        case 2
            %Crop to only face and align all subjects
            width = 180;
            height = 220;

            switch subjectNumber
                case 3
                    xMin = 93;
                    yMin = 20;

                case 4
                    xMin = 95;
                    yMin = 10;

                case 5
                    xMin = 95;
                    yMin = 10;

                case 6
                    xMin = 47;
                    yMin = 10;

                case 7
                    xMin = 87;
                    yMin = 9;

                case 8
                    xMin = 81;
                    yMin = 9;

                case 9
                    xMin = 86;
                    yMin = 10;

                case 10
                    xMin = 95;
                    yMin = 10;

                case 13
                    xMin = 87;
                    yMin = 22;

                case 14
                    xMin = 47;
                    yMin = 10;

                case 15
                    xMin = 80;
                    yMin = 20;

                otherwise
                    xMin = 100;
                    yMin = 10;

            end

        otherwise
            error('Unsupported scenarioSelection')
    end

    %Crop the image
    rect = [xMin yMin width height];
    A = faces(:,:,n);
    B = imcrop(A,rect);

    %Display this result?
    if(~isempty(find(displayCroppingIndices==n)))
        figure
        subplot(1,2,1)
        imshow(A)
        title(['n=',num2str(n),'subjectNumber=',num2str(subjectNumber)])

        subplot(1,2,2)
        imshow(B)
    end

    facesCropped(:,:,n) = B;
end

%assemble pre-processed data set
facePreprocessed = facesCropped;

%% Save data
outputFileName = ['PreprocessedDatasetScenario',num2str(scenarioSelection)];
save(outputFileName,'facePreprocessed','subjectNumberLabels','conditionLabels')
disp(['Saved to ',outputFileName])

MaximizeFigureAll();

toc
disp('DONE!')