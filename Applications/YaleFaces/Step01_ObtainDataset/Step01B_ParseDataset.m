%The Yale Faces data set is stored as gif images that should be converted
%to matrices and saved as a mat file.
%
%Christopher Lum
%lum@uw.edu

%Version History
%06/17/23: Created
%06/24/23: Updated

clear
clc
close all

tic

%% User selections
outputFileName = 'YaleFacesData.mat';
picturesFolder = 'YaleFaces\Faces';

%% Parse data
[pictures] = dir2(picturesFolder,'only_files'); %assume folder only has  pictures, ReadMe.txt is not in this folder
numPictures = length(pictures);
assert(numPictures==15*11,'There are more or less pictures in the folder than expected');

numRowsExpected = 243;
numColsExpected = 320;
faces               = uint8(zeros(numRowsExpected,numColsExpected,numPictures));
subjectNumberLabels = zeros(numPictures,1);
conditionLabels     = cell(numPictures,1);
for k=1:length(pictures)
    picture = pictures{k};

    [fileName,extension] = SeperateFileNameAndExtension(picture);

    %process picture
    A = imread([picturesFolder,filesep,picture]);
    subjectNumber = str2num(fileName(8:9));
    condition = extension;

    [M,N] = size(A);
    assert(M==numRowsExpected);
    assert(N==numColsExpected);

    faces(:,:,k)                = A;
    subjectNumberLabels(k,1)    = subjectNumber;
    conditionLabels{k,1}        = condition;
end

%% Check data consistency
disp('Checking consistency')
[M,N,P] = size(faces);
assert(P==numPictures);
assert(length(subjectNumberLabels)==numPictures);
assert(length(conditionLabels)==numPictures);

for k=1:15
    %Verify there are 15 subjects and each has exactly 11 conditions
    idx = find(subjectNumberLabels==k);
    assert(length(idx)==11,['Subject ',num2str(k),' does not appear to have 11 conditions'])

    %verify that each of the 11 conditions are represented
    conditionsExpected = sort({
        'centerlight';
        'glasses'
        'happy'
        'leftlight'
        'noglasses'
        'normal'
        'rightlight'
        'sad'
        'sleepy'
        'surprised'
        'wink'
        });

    conditionsActual = sort(conditionLabels(idx));

    assert(AreMatricesSame(conditionsActual,conditionsExpected),['Subject ',num2str(k),' does not appear to have expected conditions'])
end

%% Save data
save(outputFileName,'faces','subjectNumberLabels','conditionLabels')
disp(['Saved to ',outputFileName])

toc
disp('DONE!')