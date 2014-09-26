% Run cross validation on the ionosphere data set using the CART
% classifier. 
%
% By: Gregory Ditzler (gregory.ditzler@gmail.com)

clc;
clear;
close all;

load ionosphere;
n = size(X,1);   % number of data instances
k = 10;          % number of folds

cv = cvpartition(n,'k',k); % initialize the CV
calc_error = @(actual,prediction)(sum(~strcmp(actual,prediction))/length(prediction));

errors = zeros(k,1);
errors2 = zeros(k,1);

for i = 1:k
  % get the training and testing indices from the cv object for the ith
  % iteration. 
  i_tr = cv.training(i);
  i_te = cv.test(i);
  
  % generate a CART classifier and get the predictions on the test data 
  tree = ClassificationTree.fit(X(i_tr,:),Y(i_tr));
  pred = predict(tree, X(i_te, :));
  yhat = Y(i_te);
  
  % compute the error of the model trained on the ith round. this is not
  % the most efficent way to compute the error... but it works. 
  err = 0;
  for m = 1:length(yhat)
    if ~strcmp(yhat{m},pred{m})
      err = err+1;
    end
  end
  errors(i) = err/length(yhat);
  errors2(i) = calc_error(yhat, pred);
end

% gives the same result!
average_errors = mean(errors);
average_errors2 = mean(errors2);

