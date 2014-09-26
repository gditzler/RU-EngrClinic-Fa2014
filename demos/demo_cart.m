% Generate a CART decision tree of Fisher's Iris data set. Demonstrate some
% of the basic functionality that we can use with Matlabs CART
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

% learn a classification tree from all of the data and visualize the tree
% as a graph after it is learned. use view(tree1) to print out the logic
% statements that describe the tree. 
tree1 = ClassificationTree.fit(meas, species);
view(tree1, 'mode', 'graph');

% predict the labels of the training data. note that you would generally
% not do this in practice. the second output is optional. the first output
% is the discrete predictions and the second is the posterior probabilities
% of the data. 
[pred,posterior] = predict(tree1, meas);
