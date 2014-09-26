clc;
clear;
close all;

% create anon & variabls
k_folds = 15;
dataset = 'ionosphere';


data_opts.MU = {[1,1], [-1,-1]};
data_opts.SIGMA = {eye(2), [2 -1;-1 2]}; 
data_opts.classes = 2;
data_opts.samples = [10000 10000];
data_opts.plot = true;

%opts.classifier_type = 'tree'; 

% specific to naive bayes 
opts.classifier_type = 'naivebayes';

% specific to the knn
% opts.NumNeighbors = 5;
% opts.Distance = 'euclidean';
% opts.classifier_type = 'knn';

% specifc for the svm (poly)
% opts.classifier_type = 'svm';
% opts.kernel_function = 'polynomial';
% opts.polyorder = 2;
% opts.boxconstraint = 1;

% specific for the svm (rbf)
% opts.classifier_type = 'svm';
% opts.kernel_function = 'rbf';
% opts.rbf_sigma = 2;
% opts.boxconstraint = 1;

[data,labels] = load_data(dataset);

% perm the data & labels
idx = randperm(length(labels));
data = data(idx,:);
labels = labels(idx);

%%%%%%%%%%%%%%%
cv = cvpartition(length(labels), 'k', k_folds);
err = zeros(k_folds,1);

%parpool(4);
tic;
for k = 1:k_folds
  idx_train = cv.training(k);
  idx_test = cv.test(k);
  
  % compute the test error for the k'th fold
  err(k) = classifier_eval(data(idx_train,:), labels(idx_train), ...
    data(idx_test,:), labels(idx_test), opts);
end
runtime = toc;
%delete(gcp);

cv_error = mean(err);

%%%%%%%%%
save(['results/classification_',opts.classifier_type,'_cv',...
  num2str(k_folds),'_',dataset,'.mat']);