% X is a n-by-d matrix containing the coordinates of 
% n data points in the Poincare ball model with d dimensions
% y is a length-n vector containing binary class labels (0 or 1)
function [w, a, b] = hsvm(y, X, cvec, debug)
  [n, d] = size(X);
  assert(length(y) == n);

  if ~exist('cvec','var')
    cvec = [0.1, 1, 10];
  end

  if ~exist('debug','var')
    debug = false;
  end

  learn_rate = 0.001;
  niter = 2000;
  nfold = min([5, sum(y > 0.5), sum(y < 0.5)]);
  opts = '-s 2 -B -1 -q';

  L = ball2loid(X);
  
  optc = 0;
  optacc = -inf;
  optypred = [];

  for ci = 1:length(cvec)
    ypred = zeros(length(y), 1);

    for fo = 1:nfold
      cv = cvpartition(y, 'KFold', nfold);
      Ltrain = L(training(cv, fo),:);
      ytrain = y(training(cv, fo));
      Ltest = L(test(cv, fo),:);

      w = htrain(ytrain, Ltrain, cvec(ci));
      score = minkowski_innerprod(Ltest, w);

      ypred(test(cv, fo)) = score;
    end

    [~,pr] = auc(y, ypred);

    if pr > optacc
      optacc = pr;
      optc = cvec(ci);
      optypred = ypred;
    end
  end

  w = htrain(y, L, optc);
  [a, b] = platt(optypred, 2 * y - 1, sum(y < 0.5), sum(y > 0.5));
  
  function obj = objfn(w, x, y, C)
      margin_term = -minkowski_innerprod(w, w)/2;
      misclass_term = max(asinh(1) - asinh(y .* minkowski_innerprod(x, w)), 0);
      obj = margin_term + C * sum(misclass_term);
  end

  function wgrad = gradfn(w, x, y, C)
      wgrad_margin = [-w(1), w(2:end)];
      z = y .* minkowski_innerprod(x, w);
      missed = (asinh(1) - asinh(z)) > 0;
      wgrad_misclass = bsxfun(@times, missed .* -(1 ./ sqrt(1 + z.^2)) .* y, [x(:,1), -x(:,2:end)]);
      wgrad = wgrad_margin + C * sum(wgrad_misclass, 1);
  end

  function flag = isfeasible(w)
      flag = minkowski_innerprod(w, w) < 0;
  end

  function pt = boundary(pt, alpha, ep)
    pt(2:end) = (1 + alpha) * pt(2:end);
    pt(1) = sqrt(sum(pt(2:end).^2) - ep);
  end

  % Training subroutine
  function w = htrain(y2, L2, c)

    % Initialize with SVM in ambient Euclidean space
    % "train" function is from the LIBLINEAR package
    optstr = sprintf('%s -c %s', opts, num2str(c));
    model = train(y2, sparse(L2), optstr);

    % Flip signs accordingly
    if model.Label(1) == 1; model.w = -model.w; end
    w = model.w;
    w(1) = -w(1);

    % Convert to -1/+1 labels for the main routine
    y2 = 2 * y2 - 1;
  
    % If the initial w always predict the same label,
    % there is not much we can do
    if isfeasible(w) 

      initw = w;
      initobj = objfn(w, L2, y2, c);

      bestw = w;
      bestobj = initobj;

      if debug
        fprintf('%03d: %f\n', 0, initobj);
      end

      step_size = learn_rate;
      retry_left = 10;

      % Gradient descent
      iter = 0;
      while iter < niter
          iter = iter + 1;

          grad = gradfn(w, L2, y2, c);
          w_new = w - step_size * grad;

          if iter == 100 && retry_left > 0
            if obj > initobj % Appears to be diverging
              % Reset to initial w and reduce the learning rate
              w = initw;
              iter = 0;
              step_size = step_size / 10;
              retry_left = retry_left - 1;
              continue
            end
          end

          if debug && rem(iter, 100) == 0 
            fprintf('%03d: %f, best %f, lrate %f\n', iter, obj, bestobj, step_size);
          end
          
          % If we walk out of feasible region,
          % project back
          ep = 1e-4; % want minkowski norm of w < -ep

          if ~isfeasible(w_new)
            w0 = w_new(1)^2;
            w1 = sum(w_new(2:end).^2);
            sgn = sign(w_new(1));

            % (1+alpha) w_new(2:end)
            lo = sqrt(max(0, ep / w1 - 1));
            up = sqrt((ep + w0) / w1 - 1);

            % Euclidean distance to feasible point
            %fn = @(alpha)sum((boundary(w_new,alpha,ep) - w_new).^2);
            fn = @(alpha)((1 + alpha)^2 + alpha^2) * w1 - 2 * w_new(1) * sqrt((1 + alpha)^2 * w1 - ep); % faster

            % Golden section search (find the closest feasible point)
            aopt = goldsearch(fn, lo, up, 10);

            w_new = boundary(w_new, aopt, ep);

            assert(isfeasible(w_new));
          end

          w = w_new;

          obj = objfn(w, L2, y2, c);
          if obj < bestobj
            bestw = w;
            bestobj = obj;
          end
      end

      if debug
        pause
      end

      % Return the best parameters observed 
      w = bestw;
    end
  end
end
