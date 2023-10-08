%
% sample run of Noah James Shofer
%
%

clc;
clear




sigma = 1;
K = 1;
N = 3;
L = 1;
d = rand;

Win = zeros(N,K);
Wout = zeros(L,N);
Wfb = zeros(N,L);
W = zeros(N,N);
dw = 0.5;
wash_t = 50;
Tt = 3;
Tint = 0.01;
T0 = 0;
sr = 0.90; % Spectral radius
%beta = 0.01;
beta = 0.001;
test_l = 1;
% reservior params
leak_rate = 0.75;
nu = 0.01;
bias = 1;





% Generate Win with signa_in = 1, matrix using erdos renyi method.

for i = 1:N
    for j = 1:K
        
        p = rand;
        if p < d
           Win(i,j) = -sigma + (2*sigma)*rand;
        else 
            Win(i,j) = 0;
        end
    end
end


% Generate Win matrix using erdos renyi method.

for i = 1:N
    for j = 1:L
        p = rand;
        if p < d
           Wfb(i,j) = -sigma + (2*sigma)*rand;
        else 
            Wfb(i,j) = 0;
        end
    end
end


% Gnerate W using density d_w given. 

W_t = sprand(N,N,dw);
W_mask = (W_t~=0); 

for i = 1:N
    for j = 1:N
        if W_t(i,j)~=0
            W(i,j) =  -dw + (2*dw)*rand; 
        end
    end
end
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* ( 1 /(sr*rhoW)); % rescaled to spectal radius 0.8 (1/1.25)



% Generate training input signal function f(t) = 0.5*sin(8*pi*t).

t = T0 : Tint : Tt;
f = 0.5*sin(8*pi*t);
figure(1)
plot(f);


% Generate training output signal function g(t) = sin(8*pi*t).

t = T0 : Tint : Tt;
g = sin(8*pi*t);
figure(2)
plot(g);


% time series for reservior training input.
u_bar = f;

% time series for reservior training output.
y_bar = g;



% run the reservior and collect the intermediate states.
%
% x(t+1) = (1-leak_rate)*x(t) + leak_rate*tanh[U{u(t+1),x(t),y(t)} + bias];
%
%U{u(t+1),x(t),y(t)} = Win*u(t+1) + W*x(t) + Wfb*y(t) + nu; 
%

% TRAINING PART OF THE RESERVOIR
x = zeros(N,1);
u = u_bar;
y = y_bar;

for i = 1:length(t)
 
    x =  (1-leak_rate)*x + leak_rate*tanh( (Win*u(i) + W*x + Wfb*y(i) + nu) + bias);
    
    if i > 50
       X(:,i-50) = x;
    end
end

M = X';
T = y(wash_t+1:end)';

figure(3)
plot(M);

Wout = (T'*M)*(inv(M'*M + beta*eye(N,N))); 


t = Tt : Tint : Tt+1;
f = 0.5*sin(8*pi*t);
g = sin(8*pi*t);

% PREDICTIVE PART OF THE RESERVOIR
yp = g(1);
u = f;
num = 1;
for i = 1:length(t)-1
   x =  (1-leak_rate)*x + leak_rate*tanh( (Win*u(i) + W*x + Wfb*yp(i) + nu) + bias);
   
   %yout(:,num) = Wout*x;%X(:,num);
   yp(i+1) = Wout*x;
 
end

figure(4)
plot(t,yp, t,g);

%figure(5)
%plot(t(1,end-wash_t:end),u_bar(end-wash_t:end),t(1,end-wash_t:end),yout);




















