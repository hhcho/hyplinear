% Convert coordinates from the hyperboloid model
% to the Poincare ball model
% x is a n-by-d matrix where d is the number of dimensions
function b = loid2ball(x)
    b = bsxfun(@rdivide, x(:,2:end), 1 + x(:,1));
end