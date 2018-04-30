function [x,err] = PoisSolver(f,a,b,n)

h = (b-a)/n;
j = 1:n;
xj = a + (j-0.5).*h;
A = diag([-3,-2*ones(1,n-2),-3]) + diag(ones(1,n-1),1) ...
    + diag(ones(1,n-1),-1);
A
B = h.^2*f(xj);
B = B'
x = (A\B);
x
g = @(xj) sin(4*pi*xj);
err = g(xj)'-x;

end