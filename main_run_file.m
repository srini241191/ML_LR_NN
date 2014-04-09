%Script for running the functions of logistic regression and NN - Srinivasan Rengarajan - 50097996%
%Logistic regression%
clear all;
close all;
clc;
load X_data.mat;
load test_data.mat;
%Training -LR%
[wnew]= train_lr(X_data);
%Prediction - LR%
[y predict hits r_rank]= test_lr(wnew, test_data);
%Neural network%
[w1 w2]= nn_train(X_data);
%Predict%
[yk predict1 hits1 r_rank]= test_nn(w1,w2,test_data);
