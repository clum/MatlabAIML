%Familiarize with the simplefit_dataset.
%
%More information about this dataset can be obtained by entering
%
%   help simpleseries_dataset
%
%in the command window
%
%Christopher Lum
%lum@uw.edu

%Version History
%11/25/23: Created

clear
clc
close all

ChangeWorkingDirectoryToThisLocation();

tic

%% User selections

%% Load data
temp = load('simpleseries_dataset.mat');

simpleseriesInputs      = temp.simpleseriesInputs;
simpleseriesTargets     = temp.simpleseriesTargets;

%% Visualize the dataset
u = cell2mat(simpleseriesInputs);
d = cell2mat(simpleseriesTargets);

plot(u,d,'rx')
xlabel('u')
ylabel('d')
grid on

%% Train a network
[X,T] = simpleseries_dataset;
net = narxnet(1:2,1:2,10);

%prepare time series data for network simulation or training
[Xs,Xi,Ai,Ts] = preparets(net,X,{},T);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
Y = net(Xs,Xi,Ai)
plotresponse(Ts,Y)

toc
disp('DONE!')