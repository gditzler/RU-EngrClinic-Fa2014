function err = classifier_eval(data_train, labels_train, data_test, labels_test, opts)
%   err = classifier_eval(type, data_train, labels_train, data_test, labels_test)
%
%   Compute the error of a classifier 
%   Written: Me 
%
% See also
% CLASSIFICATIONTREE, NAIVEBAYES, FITCKNN, SVMTRAIN, SVMCLASSIFY

% our simple anon fucntion to calculate error. 
calc_error = @(x, y) sum(x~=y)/length(y);

switch opts.classifier_type
  case 'tree'
    % build / test a decision tree
    tree = ClassificationTree.fit(data_train, labels_train);
    pred = predict(tree, data_test);
    
  case 'naivebayes'
    % build / test a naive bayes classifier
    nb = NaiveBayes.fit(data_train, labels_train);
    pred = predict(nb, data_test);
    
  case 'knn'
    % build / test a k-nn classifier
    if ~isfield(opts, 'Distance')
      opts.Distance = 'euclidean';
    end
    
    mdl = fitcknn(data_train, labels_train, 'NumNeighbors', ...
      opts.NumNeighbors, 'Distance', opts.Distance);
    pred = predict(mdl, data_test);
    
  case 'svm'
    % build / test a support vector machine
    switch opts.kernel_function
      case 'polynomial'
        % user requested a polynomial kernel
        svm = svmtrain(data_train, labels_train, 'kernel_function', ...
          opts.kernel_function, 'polyorder', opts.polyorder, ...
          'boxconstraint', opts.boxconstraint);
      case 'rbf'
        % user requested a rbf/gaussian kernel
        svm = svmtrain(data_train, labels_train, 'kernel_function', ...
          opts.kernel_function, 'rbf_sigma', opts.rbf_sigma, ...
          'boxconstraint', opts.boxconstraint);
      otherwise
        error('Unknown Kernel!');
    end
    
    pred = svmclassify(svm, data_test);
    
  case 'adaboost'
    
    if length(unique([labels_test;labels_train])) >= 3
      ens = fitensemble(data_train,labels_train, ...
        'AdaBoostM2',opts.NLearn,opts.Learners);
    else
      ens = fitensemble(data_train,labels_train, ...
        'AdaBoostM1',opts.NLearn,opts.Learners);
    end
    pred = predict(ens, data_test);
    
  otherwise
    error('Unknown classifier!');
end

err = calc_error(pred, labels_test);