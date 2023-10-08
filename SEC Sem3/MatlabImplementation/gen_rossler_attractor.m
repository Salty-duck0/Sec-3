clc
clear

tspan = 0:0.001:120;
options = odeset('RelTol', 1e-6);



[t_rossler, data_rossler] = ode45('rossler', tspan, [1.0 0.01 0.254], options);

[t_lorenz, data_lorenz] = ode45('lorenz', tspan, [1.0 0.01 0.254], options);


plot3(data_rossler(:,1),data_rossler(:,2),data_rossler(:,3));
plot3(data_lorenz(:,1),data_lorenz(:,2),data_lorenz(:,3));

save('rossler.mat','t_rossler', 'data_rossler');
save('lorenz.mat','t_lorenz', 'data_lorenz');