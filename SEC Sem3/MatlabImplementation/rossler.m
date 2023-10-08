
function dx = rossler(t, x)
%tspan = 0:0.001:10
%options = odeset('RelTol', 1e-6);
%[t, y] = ode45('rossler', tspan, [1.0 0.01 0.254], options);
% Standard constants for the Lorenz Attractor

a = 0.5;
b = 2.0;
c = 4.0;

% I like to initialize my arrays
dx = [0; 0; 0];

% The lorenz strange attractor
dx(1) = -x(2)-x(3);
dx(2) = x(1) + a*x(2);
dx(3) = b + x(3)*(x(1)-c);

