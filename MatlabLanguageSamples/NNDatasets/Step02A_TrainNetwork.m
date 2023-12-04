%Train a neural network
%
%Christopher Lum
%lum@uw.edu

%Version History
%11/25/23: Created
%12/01/23: Continued working

clear
clc
close all

ChangeWorkingDirectoryToThisLocation();

tic

%Use Neural Network Fitting app
nftool


%Here is how to solve to this problem at the command line, with a fitting
%neural network with 10 hidden neurons. See fitnet for more details.  
%
%   [x,t] = simplefit_dataset; 
%   plot(x,t) 
%   net = fitnet(10);
%   net = train(net,x,t);
%   view(net)
%   y = net(x);
%   plot(net,x,t);

% %% Train a network
% [X,T] = simpleseries_dataset;
% net = narxnet(1:2,1:2,10);
% 
% %prepare time series data for network simulation or training
% [Xs,Xi,Ai,Ts] = preparets(net,X,{},T);
% net = train(net,Xs,Ts,Xi,Ai);
% view(net)
% Y = net(Xs,Xi,Ai)
% plotresponse(Ts,Y)

toc
disp('DONE!')