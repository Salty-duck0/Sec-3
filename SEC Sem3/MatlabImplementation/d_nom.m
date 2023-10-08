
function out = d_nom(in)
%
% data_rossler  (120000X3);
%

x = in(:,1);
y = in(:,2);
z = in(:,3);

x_m = mean(x); y_m = mean(y); z_m = mean(z);

num = (x-x_m);
dnum1 = (x-x_m).^2;
dnum2 = sqrt(mean(dnum1));

out(:,1) = num/dnum2; 



num = (y-y_m);
dnum1 = (y-y_m).^2;
dnum2 = sqrt(mean(dnum1));

out(:,2) = num/dnum2;



num = (z-z_m);
dnum1 = (z-z_m).^2;
dnum2 = sqrt(mean(dnum1));

out(:,3) = num/dnum2;

%plot3(out(:,1), out(:,2), out(:,3));
end