% Implement adaboost on the Ionosphere data set. This script was used to
% generate the plot shown in the presentation. 
%
% By: Gregory Ditzler (gregory.ditzler@gmail.com)

% best tree lines to start a program with!
clc;
clear;
close all;

% load the ionosphere data 
load ionosphere;

M = 100;  % number of classifiers in the ensemble
L = 10;   % number of averages to run

err = zeros(M,L);
for m = 1:L
  rng(m); % set the random seed
  
  % draw a random sample from the data 
  w = randsample(1:length(Y),length(Y),true,ones(1,length(Y))/length(Y));
  
  % fit the ensemble to the sampled data and extract the training loss
  ada = fitensemble(X(w,:),Y(w),'AdaBoostM1',M,'tree');
  err(:,m) = resubLoss(ada,'mode','cumulative');
end

% plot the results 
fs = 22;  % figure font sizes
figure;
hold on;
box on;
plot(err,'color',[0,0,0]+.6)
plot(mean(err,2),'k');
xlabel('size of the ensemble','FontSize',fs);
ylabel('error','FontSize',fs);
set(gca,'fontsize',fs)
