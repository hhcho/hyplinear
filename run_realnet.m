addpath /PATH/TO/LIBLINEAR/matlab % modify
addpath code

nets = {'karate', 'polbooks', 'football', 'polblogs'};

for nt = 1:4 % Network ID
  for runid = 1:5 % Embedding ID (independent runs)

    fprintf('%s, run %d ... ', nets{nt}, runid);

    datafile = sprintf('data/realnet/%s_data_%d.mat', nets{nt}, runid);
    load(datafile);

    acc = crossval_svm(label, B, 3, 0.5, 5);

    m = mean(acc(:,:,end));
    fprintf('hyperbolic (%f), euclidean (%f)\n', m);

  end
end
