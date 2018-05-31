% x is a n-by-d matrix
% y is a m-by-d matrix
% output D is a n-by-m matrix
function D = minkowski_innerprod(x, y)
    x(:,2:end) = -x(:,2:end);
    D = x * y';
end