% X is a n-by-d matrix containing the coordinates of 
% n data points in the Poincare ball model with d dimensions
% label is either:
%   (1) a length-n vector containing class labels (can be multi-class); or
%   (2) a n-by-k binary matrix (can be multi-label)
% svmtype: (1) hyperbolic, (2) Euclidean, (3) both
%           
function acc = crossval_svm(label, X, svmtype, holdout, ntrial)
  n = size(X, 1);

  if ~exist('svmtype', 'var')
    svmtype = 3;
  end

  if ~exist('holdout', 'var')
    holdout = 0.5;
  end

  if ~exist('ntrial', 'var')
    ntrial = 5;
  end

  assert(ismember(svmtype, 1:3))

  if svmtype == 3
    svmlist = [1, 2];
  else
    svmlist = svmtype;
  end

  % Convert labels to numeric values if a label vector is given
  if size(label, 1) == 1 || size(label, 2) == 1
    ulab = unique(label);
    nclass = length(ulab);
    assert(nclass >= 2);

    map = containers.Map(ulab,1:nclass);
    label = cell2mat(values(map, num2cell(label)));
    label = label(:);
    Y = full(sparse((1:n)', label, 1, n, nclass));
  else
    Y = label;
    assert(size(Y, 1) == n)
    nclass = size(Y, 2);
  end

  probfn = @(x,a,b) 1 ./ (1 + exp(a * x + b));

  acc = zeros(ntrial, length(svmlist), 3 + 2 * (nclass>2));

  for tr = 1:ntrial

    % If a data point has multiple labels,
    % randomly choose one for stratified CV partition
    A = Y';
    T = bsxfun(@rdivide, A, max(1, sum(A)));
    TS = cumsum(T);
    T(A>0) = TS(A>0);
    chosen = 1 + sum(cumprod(1 - bsxfun(@le, rand(1, n), T)));

    cv = cvpartition(chosen(:), 'HoldOut', holdout);
    Xtrain = X(training(cv),:);
    Ytrain = Y(training(cv),:);
    Xtest = X(test(cv),:);
    Ytest = Y(test(cv),:);
    ntest = size(Ytest, 1);

    for st = 1:length(svmlist)
      score = zeros(ntest, nclass);

      for k = 1:nclass
        if nclass == 2 && k == 1
          continue
        end

        ytrain_binary = Ytrain(:,k);

        if svmlist(st) == 1
          [w, pA, pB] = hsvm(ytrain_binary, Xtrain);
          s = minkowski_innerprod(ball2loid(Xtest), w);
        elseif svmlist(st) == 2
          [w, pA, pB] = esvm(ytrain_binary, Xtrain);
          s = [Xtest, ones(ntest, 1)] * w;
        end

        if nclass == 2
          score(:,k) = s;
        else
          score(:,k) = probfn(s, pA, pB);
        end
      end

      if nclass == 2
        acc(tr,st,:) = evalperf(Ytest(:,2), score(:,2));
      else
        acc(tr,st,:) = evalperf(Ytest, score);
      end
    end
  end

  function res = evalperf(Y, S)
    [~,I] = max(S,[],2);
    J = sub2ind((1:size(Y,1))', I(:));
    res = mean(Y(J));

    % micro-average
    [roc, pr] = auc(Y(:),S(:));
    res = [res, roc, pr];

    if size(S,2) > 2
      % macro-average
      agg = [0 0];
      nc = nclass;
      for yi = 1:size(Y,2)
        if sum(Y(:,yi)) == 0 || sum(Y(:,yi)) == size(Y,1)
          nc = nc - 1; % ignore classes that are trivial
          continue
        end
        [roc, pr] = auc(Y(:,yi),S(:,yi));
        agg = agg + [roc pr];
      end
      res = [res, agg / nc];
    end
  end
end
