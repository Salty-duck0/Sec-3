
function dx = lorenz(t, x)

%
%[t, y] = ode45('lorenz', [0 20], [10 10 10]);
% Standard constants for the Lorenz Attractor

sigma = 10;
rho = 28;
beta = 8/3;

% I like to initialize my arrays
dx = [0; 0; 0];

% The lorenz strange attractor
dx(1) = sigma*(x(2)-x(1));
dx(2) = x(1)*(rho-x(3))-x(2);
dx(3) = x(1)*x(2)-beta*x(3);