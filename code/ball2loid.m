% Convert coordinates from the Poincare
% ball model to the hyperboloid model
% b is a n-by-d matrix where d is the number of dimensions
function x = ball2loid(b)
    x0 = 2 ./ (1 - sum(b.^2,2)) - 1;
    x = [x0, bsxfun(@times, b, x0 + 1)];
end