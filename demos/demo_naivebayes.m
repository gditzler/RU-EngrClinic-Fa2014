% Generate a naive Bayes classifier on Fisher's Iris data set. Demonstrate 
% someof the basic functionality that we can use with Matlab's
% implementation. 
%
% By: Gregory Ditzler (gregory.ditzler@gmail.com)

% best tree lines to start a program with!
clc;
clear;
close all;

% load the fisher isrris data set. the features are in 'meas' and the
% labels are in 'species'.
load fisheriris;

% learn a naive bayes model from the data 
nb = NaiveBayes.fit(meas, species);

% predict the labels of the training data. 
pred = nb.predict(meas);
posterior = nb.posterior(meas);