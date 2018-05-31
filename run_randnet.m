function run_randnet
  addpath code
  addpath /PATH/TO/LIBLINEAR/matlab % modify
  
  nclass = 10;
  range_list = {[20 50], [50 100], [100 200]};
  
  for rg = 1:length(range_list) % Size range

    valid_range = range_list{rg};
  
    for r = 1:10 % Network ID

      for t = 1:5 % Repetition

        ws = load_data(r);
        Y = zeros(ws.N, nclass);
  
        fprintf('[%d-%d] Net %d Trial %d ... ', range_list{rg}, r, t);
  
        %% Randomly generate a multi-class, multi-label dataset over the network
        for ci = 1:nclass
          y = zeros(ws.N, 1);
  
          while sum(y) < valid_range(1) || valid_range(2) < sum(y)
            % Pick a random pioneer node
            first = randi(ws.N);
  
            % Randomly activate network edges
            inheritance_prob = 0.8; 
            activated = rand(nnz(ws.A),1) < inheritance_prob;
            A = ws.A;
            A(A > 0) = activated;
  
            % Propagate label
            tagged = false(1, ws.N);
            tagged(first) = true;
            prev_count = 0;
            cur_count = 1;
  
            while prev_count ~= cur_count
              prev_count = cur_count;
              tagged = tagged | (tagged * A > 0);
              cur_count = nnz(tagged);
            end
  
            y = double(tagged(:));
          end
  
          Y(:,ci) = y;
        end
  
        %% Cross validation
        acc = crossval_svm(Y, ws.labne, 3, 0.5, 5);
  
        m = mean(acc(:,:,end));
        fprintf('hyperbolic (%f), euclidean (%f)\n', m);

      end
    end
  end

  function ws = load_data(runid)
    outprefix = sprintf('data/randnet/psmodel_deg4_gma2.25_%d_', runid);
    ws = struct;
    edgelist = dlmread([outprefix, 'net.txt']);
    ws.N = max(edgelist(:));
    ws.A = sparse(edgelist(:,1), edgelist(:,2), 1, ws.N, ws.N);
    ws.labne = dlmread([outprefix, 'labne.txt']);
  end
end
