addpath code
addpath /PATH/TO/LIBLINEAR/matlab % modify

for eid = 1:100 % Dataset ID

  eidstr = sprintf('%03d', eid);
  matfile = ['data/gaussian/data_', eidstr, '.mat'];
  load(matfile)

  fprintf([eidstr, ' ... ']);
  acc = crossval_svm(label, B, 3, 0.5, 5);
  m = mean(acc(:,:,end));
  fprintf('hyperbolic (%f), euclidean (%f)\n', m);

end
