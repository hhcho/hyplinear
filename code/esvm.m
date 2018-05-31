% X is a n-by-d matrix containing the coordinates of n data points in d dimensions
% y is a length-n vector containing binary class labels (0 or 1)
function [w, a, b] = esvm(y, X, cvec)
  [n, d] = size(X);
  assert(length(y) == n);

  if ~exist('cvec','var')
    cvec = [0.1, 1, 10];
  end

  nfold = min([5, sum(y > 0.5), sum(y < 0.5)]);
  opts = '-s 2 -B 1 -q';

  optc = 0;
  optacc = -inf;
  optypred = [];
  for ci = 1:length(cvec)
    optstr = sprintf('%s -c %f', opts, cvec(ci)); 

    ypred = zeros(length(y), 1);

    for fo = 1:nfold
      cv = cvpartition(y, 'KFold', nfold);
      Xtrain = X(training(cv, fo),:);
      ytrain = y(training(cv, fo));
      Xtest = X(test(cv, fo),:);
      ntest = size(Xtest, 1);

      % Invoke "train" function from the LIBLINEAR package
      optstr = sprintf('%s -c %s', opts, num2str(cvec(ci)));
      model = train(ytrain, sparse(Xtrain), optstr);
      if model.Label(1) == 0; model.w = -model.w; end

      ypred(test(cv, fo)) = [Xtest, ones(ntest,1)] * model.w(:);
    end

    [~,pr] = auc(y, ypred);

    if pr > optacc
      optacc = pr;
      optc = cvec(ci);
      optypred = ypred;
    end
  end

  optstr = sprintf('%s -c %f', opts, optc); 

  model = train(y, sparse(X), optstr);
  if model.Label(1) == 0; model.w = -model.w; end

  w = model.w(:);
  [a, b] = platt(optypred, 2 * y - 1, sum(y < 0.5), sum(y > 0.5));

end
