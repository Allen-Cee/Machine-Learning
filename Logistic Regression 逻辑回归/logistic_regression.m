function [opt_theta] = logistic_regression()

load('data.txt');

m = size(data, 1);
n = size(data, 2);

x0 = ones(m, 1);
x1 = data(:, 1);
x2 = data(:, 2);
y = data(:, 3);
X = [x0 x1 x2];
%code above for future plotting
%unfinished

x1_1 = x1;
x1_0 = x1;
x1_1(y == 0, :) = [];
x1_0(y == 1, :) = [];
x2_1 = x2;
x2_0 = x2;
x2_1(y == 0, :) = [];
x2_0(y == 1, :) = [];

options = optimset('GradObj', 'On', 'MaxIter', 500);
theta = zeros(n, 1);
[opt_theta, function_value, exit_flag] = fminunc(@cost_J, theta, options);

hold on
scatter(x1_1, x2_1, 'r')
scatter(x1_0, x2_0, 'b')
plot(x1, (opt_theta(1)-0.5+opt_theta(2)*x1)/(-opt_theta(3)), 'k')
%decision boundary = 0.5

