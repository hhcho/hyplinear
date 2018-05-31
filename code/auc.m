function [roc, pr] = auc(label, score, perturb_std, pr_thres)
  if ~exist('perturb_std', 'var')
    perturb_std = 0;
  end

  if ~exist('pr_thres', 'var')
    pr_thres = inf;
  end

  label = label(:);
  score = score(:);

  noise = sqrt(perturb_std) * randn(length(score), 1);
  score = score + noise;

  [~,ord] = sort(score, 'descend');
  label = label(ord);

  P = nnz(label);
  N = length(label) - P;

  TP = cumsum(label);
  PP = (1:length(label))';

  % ROC
  x = (PP - TP) ./ N;
  y = TP ./ P;
  roc = trapz(x, y);

  % PR
  x = y;
  y = TP ./ PP;

  if TP(end) > pr_thres
    trunc = find(TP > pr_thres, 1, 'first') - 1;
    pr = trapz(x(1:trunc), y(1:trunc)) / x(trunc);
  else
    pr = trapz(x, y);
  end
end
