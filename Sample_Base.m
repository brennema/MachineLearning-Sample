%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RECONSTRUCTION USING MACHINE LEARNING
% Elora C Brenneman, Anthony A. Gatti, Jacyln N. Hurley, Monica R. Maly
% 25 August 2016
% McMaster University
% Hamilton, ON CANADA
%%%%%
% This code includes sample files and examples of how the machine learning 
% approach, and the machine learning with circular solution space (CSS) 
% can reconstruct kinematic rigid body cluster data.  Only one cluster
% marker is able to be reconstructed in current code.  
%%%%%
% Lines 244 - 249 were adapted from code written by Piotr Dollar (2009) 
% [pdollar-at-caltech.edu]. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all
currentFolder = pwd;
%%%%%
% Base Program Information
%%%%%

disp('*********************************************************************')
disp('Sample Code - Waveform Reconstruction via Machine Learning')
disp('Available from https://github.com/brennema/MachineLearning-Sample')
disp('Last updated: 25 August 2016')
disp('*********************************************************************')

%%%%%
% Trial selection
%%%%%

cd(currentFolder) %if trial is not in working directory, insert file path here
sampleTrial = xlsread('SampleTrial.xlsx','B2:J142');
Sh1 = sampleTrial(:, 1:3);
Sh2 = sampleTrial(:, 4:6);
Sh3 = sampleTrial(:, 7:9);

%%%%%
% Call function
%%%%%

[RMSE] = Sample_RMSECurves(Sh1, Sh2, Sh3, sampleTrial); 

%%%%%
% Display program summary
%%%%%

clc
disp('*********************************************************************')
disp('PROGRAM SUMMARY:')
disp('RMS Error - waveform interpolation: DONE!')
disp('RMS Error - machine learning: DONE!')
disp('RMS Error - machine learning with circular solution space: DONE!')
disp('*********************************************************************')
