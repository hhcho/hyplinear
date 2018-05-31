% Golden section search
% Minimize a function over [xmin, xmax]
function [xopt, fopt] = goldsearch(fn, xmin, xmax, niter)

  if ~exist('niter', 'var')
    niter = 10;
  end

  assert(xmin < xmax);

  a = xmin;
  fa = fn(a);
  b = xmax;
  fb = fn(b);

  phi = (1 + sqrt(5)) / 2;
  c = b - (b - a) / phi;
  fc = fn(c);
  d = a + (b - a) / phi;
  fd = fn(d);

  for i = 1:niter
    if fc < fd
      b = d;
      fb = fd;
      d = c;
      fd = fc;

      c = b - (b - a) / phi;
      fc = fn(c);
    else
      a = c;
      fa = fc;
      c = d;
      fc = fd;

      d = a + (b - a) / phi;
      fd = fn(d);
    end
  end

  x = [a b c d];
  [fopt, ind] = min([fa fb fc fd]);
  xopt = x(ind);

end
