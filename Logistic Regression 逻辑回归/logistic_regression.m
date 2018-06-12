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

options = optimset('GradObj', 'On', 'MaxIter', 500);
theta = zeros(n, 1);
[opt_theta, function_value, exit_flag] = fminunc(@cost_J, theta, options);

