%
% sample run of Noah James Shofer
%
%

clc;
clear

    
sigma = 1;           % (-sigma, +sigma) used for noise sampling random number
K = 3;               % Number of Inputs.
N = 100;             % Reservoir size or Number of Nurons.
L = 3;               % Number of Outputs.
d = rand;

Win = zeros(N,K);    % input weight matrix
Wout = zeros(L,N);   % output weight matrix
Wfb = zeros(N,L);    % feedback weight matrix;
W = zeros(N,N);      % reservoir weight matrix or reservoir size.
dw = 0.50;           % Reservoir density
wash_t = 100;        % washout time
Tt = 120;            % total time
Tint = 0.001;        % step size of data sequence
T0 = 0;              % initial time
sr = 0.80;            % Spectral radius
beta = 0.01;         % regularization constant 
train_l = 100000;     % training length
test_l = 60000;      % test length
TtH = Tt/2;
% reservior params
leak_rate = 0.75;     % leak rate
nu = 0.01;            % noise term.
bias = 1;            % bias term
d_fb = 0.5;          % Feedback density
sigma_fb = 0.3;      % Feedback radius 


% load time series of your choice.
load('rossler.mat'); % data_rossler (Tt X 3)
data_rossler = d_nom(data_rossler);
%plot3(data_rossler(:,1), data_rossler(:,2), data_rossler(:,3));
data_in = data_rossler(1:train_l+1,:)';
data_out = data_rossler(1:train_l+1,:)';



% index for input and output time series in atrractors
j_in = 1:3;            % input 1 for x; 2 for y; 3 for z time series 
j_out = 1:3;           % output 1 for x; 2 for y; 3 for z time series
input_var = 3;         % number of inputs






% Generate Win with sigma_in = 1, matrix using erdos renyi method.

for i = 1:N
    for j = 1:K
        p = rand;
        if p < d
           Win(i,j) = -sigma + (2*sigma)*rand;    % [-sigma_w to sigma_w]
        else 
            Win(i,j) = 0;
        end
    end
end


% Generate Wfb matrix using erdos renyi method.

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

t = T0 : Tint : TtH;
f = data_in(j_in,1:length(t));
% figure(1)
% plot(f');


% Generate training output signal function g(t) = sin(8*pi*t).

t = T0 : Tint : TtH;
g = data_out(j_out,1:length(t));
% figure(2)
% plot(g');


% time series for reservior training input.
u_bar = f;

% time series for reservior training output.
y_bar = g;


%TRAINING PART OF THE RESERVOIR
X = zeros(length(t)-wash_t,N,input_var);
x = zeros(N,input_var);
u = u_bar;
y = y_bar;

for k = 1:input_var
    
    for i = 1:length(t)
 
        x(:,k) =  (1-leak_rate)*x(:,k) + leak_rate*tanh( (Win(:,k)*u(k,i) + W*x(:,k) + Wfb(:,k)*y(k,i) + nu) + bias);
    
        if i > wash_t
            X(i-wash_t,:,k) = x(:,k);
        end
    end
    
    M(:,:,k) = X(:,:,k);
    
end


T = y(:,wash_t+1:end)';
%T(:,2:3) = 0;




% figure(3)
% plot(M(:,:,1));

for k = 1: input_var
    Wout(k,:) = (T(:,k)'*M(:,:,k))*(inv(M(:,:,k)'*M(:,:,k) + beta*eye(N,N)));
end

% 
% figure()
% plot((Wout(1,:)*X(:,:,1)')'); hold on;
% plot(u(1,:)); hold off;
% 
% figure()
% plot((Wout(2,:)*X(:,:,2)')'); hold on;
% plot(u(2,:)); hold off;
% 
% figure()
% plot((Wout(3,:)*X(:,:,3)')'); hold on;
% plot(u(3,:)); hold off;



%PREDICTIVE PART OF THE RESERVOIR
%%%%% x as ionput %%%%

% clear x;
% clear u;
% clear y;
% u = data_rossler(train_l+1:train_l+test_l,:)';
% x1 = zeros(N,1); x2 = zeros(N,1); x3 = zeros(N,1);
% y(:,1) = u(:,1);
% 
% 
%     for i = 1:test_l-1
%         x1 =  (1-leak_rate)*x1 + leak_rate*tanh( (Win(:,1)*u(1,i) + W*x1 + Wfb(:,1)*y(1,i) + nu) + bias);
%         y(2,i+1) = Wout(2,:)*x1;
%         y(3,i+1) = Wout(3,:)*x1;
%         y(1,i+1) = u(1,i+1);
%     end
% 
%     
%%% Y as input%%%    



%clear x;
clear u;
clear y;
u = data_rossler(train_l+1,:)';

x1=x(:,1);
x2=x(:,2);
x3=x(:,3);

%x1 = zeros(N,1); x2 = zeros(N,1); x3 = zeros(N,1);
y = u;


    for i = 1:test_l-1
        x1 =  (1-leak_rate)*x1 + leak_rate*tanh( (Win(:,1)*u(1,i) + W*x1 + Wfb(:,1)*y(1,i) + nu) + bias);
        x2 =  (1-leak_rate)*x2 + leak_rate*tanh( (Win(:,2)*u(2,i) + W*x2 + Wfb(:,2)*y(2,i) + nu) + bias);
        x3 =  (1-leak_rate)*x3 + leak_rate*tanh( (Win(:,3)*u(3,i) + W*x3 + Wfb(:,3)*y(3,i) + nu) + bias);
        
        y(1,i+1) = Wout(1,:)*x1;
        y(2,i+1) = Wout(2,:)*x2;
        y(3,i+1) = Wout(3,:)*x3;
        
        u=y;
        
    end
    
    
yout = y;    
pp= d_nom(yout')';
yout(1,:)=pp(1,:);
yout(3,:)=pp(3,:);


figure()
plot(t(1,1:test_l),yout(1,:),'-r'); hold on;
plot(t(1,1:test_l),yout(2,:),'-g'); hold on;
plot(t(1,1:test_l),yout(3,:),'-b'); hold off;



figure();
plot(t(1,1:test_l),data_rossler(train_l+1:train_l+test_l,3)'); hold on;
plot(t(1,1:test_l),yout(3,:),'-r'); hold off

figure();
plot(t(1,1:test_l),data_rossler(train_l+1:train_l+test_l,2)'); hold on;
plot(t(1,1:test_l),yout(2,:),'-r'); hold off;

figure();
plot(t(1,1:test_l),data_rossler(train_l+1:train_l+test_l,1)');hold on;
plot(t(1,1:test_l),yout(1,:),'-g'); hold off;



figure();
plot3(y(1,:),y(2,:),y(3,:));











