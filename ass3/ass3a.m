%% line
clear all; close all; clc
line = importdata('data/line.mat');
subplot(1, 3, 2);
gscatter(line.x, line.t);
title('Line');

xrange = linspace(0, 1, 100);
% preprocessing data
x = line.x;
t = line.t - ones(25, 1) * mean(line.t);

number_of_nodes = [2, 3, 4, 5, 7, 10, 16, 25];

figure;
for i = 1:length(number_of_nodes)
    M = [1:5; 6:10; 11:15; 16:20; 21:25];
    subplot(length(number_of_nodes) / 2, 2, i)
    hold on;
    for j = 1:5 % 5 fold verification
        A = linspace(1, 25, 25);
        A(A == M(j, 1)) = [];
        A(A == M(j, 2)) = [];
        A(A == M(j, 3)) = [];
        A(A == M(j, 4)) = [];
        A(A == M(j, 5)) = [];

        x_training = x(A, :);
        t_training = t(A, :);
        x_test = x(M(j, :), :);
        t_test = t(M(j, :), :);

        nhidden = number_of_nodes(i);
        nout = 1;
        cycles = 60;
        alpha = 0.01;

        net = mlp(1, nhidden, nout, 'linear', alpha);
        options = zeros(1, 18);
        options(1) = 1;
        options(14) = cycles;
        options(18) = 0.01;

        [net] = netopt(net, options, x_training, t_training, 'scg');
        yg = mlpfwd(net, xrange');
        d = mlpfwd(net, x_test);

        error(i, j) = mean((d - t_test) .^2);
        plot(xrange, yg)
    end

    scatter(x, t);
    hold off;
    title(sprintf('MLP with %d hidden nodes', number_of_nodes(i)));
    root_mean_squared_error(i) = sqrt(mean(error(i, :)'));
end

polyfit_degrees = [1, 2, 3, 4, 5, 10];
figure;
for k = 1:length(polyfit_degrees)
    subplot(1, length(polyfit_degrees), k);
    hold on;
    scatter(x, t, 'r', 'fill');
    for l = 1:5 % 5 fold verification
        A = linspace(1, 25, 25);
        A(A == M(l, 1)) = [];
        A(A == M(l, 2)) = [];
        A(A == M(l, 3)) = [];
        A(A == M(l, 4)) = [];
        A(A == M(l, 5)) = [];

        rand = randperm(25);
        x_training = x(rand(1:20));
        t_training = t(rand(1:20));
        x_test = x(rand(21:25));
        t_test = t(rand(21:25));

        p = polyfit(x_training, t_training, polyfit_degrees(k));
        plot(xrange, polyval(p, xrange));
        error_polyfit(k, l) = mean((polyval(p, x_test) - x_test) .^2);
    end

    error_polymean = sqrt(mean(error_polyfit(:,:)'));
end

figure;
scatter(number_of_nodes, root_mean_squared_error, 'fill', 'k');
title('Root mean square error of MLP with 5-fold cross validation');
ylabel('RMS error');
xlabel('Number of hidden nodes');
figure;
scatter(polyfit_degrees, error_polymean, 'fill', 'r');
title('Root mean squared error of polyfit error');
xlabel('Polyfit degree');
ylabel('RMS error');

%% sinus
clear all; close all; clc;
sinus = importdata('data/sinus.mat');
subplot(1, 3, 3);
gscatter(sinus.x, sinus.t);
title('Sinus');

xrange = linspace(0, 1, 100);
% preprocessing data
x = sinus.x;
t = 0.5 + sinus.t ./ 4;

number_of_nodes = [2, 3, 4, 5, 7, 10, 15, 25];
rand = randperm(25);

figure;
for i = 1:length(number_of_nodes)
    M = [1:5, 6:10, 11:15, 16:20, 21:25];
    subplot(length(number_of_nodes) / 2, 2, i);
    hold on;
    for j = 1:5 % 5-fold cross verification
        test_values = (((j - 1) * 5 + 1):j * 5);
        train_values = linspace(1, 25, 25);
        train_values(test_values) = [];
        x_training = x(rand(train_values));
        t_training = t(rand(train_values));
        x_test = x(rand(test_values));
        t_test = t(rand(test_values));

        nhidden = number_of_nodes(i);
        nout = 1;
        cycles = 60;

        net = mlp(1, nhidden, nout, 'logistic');
        options = zeros(1, 18);
        options(1) = 1;
        options(14) = cycles;
        options(18) = 0.01;

        [net] = netopt(net, options, x_training, t_training, 'scg');
        yg = mlpfwd(net, xrange');
        d = mlpfwd(net, x_test);
        error(i, j) = mean((d - t_test) .^ 2);

        plot(xrange, yg);
    end

    scatter(x, t);
    hold off;
    title(sprintf('MLP with %d hidden nodes', number_of_nodes(i)));
    root_mean_squared_error(i) = sqrt(mean(error(i, :)'));
end

polyfit_degrees= [1, 2, 3, 4, 5, 10];
figure;
for k = 1:length(polyfit_degrees)
    subplot(1, length(polyfit_degrees), k)
    hold on;
    scatter(x, t, 'r', 'fill');
    for l = 1:5 % 5-fold verification
        test_values = (((l - 1) * 5 + 1):l * 5);
        train_values = linspace(1, 25, 25);
        train_values(test_values) = [];
        x_training = x(rand(train_values));
        t_training = t(rand(train_values));
        x_test = x(rand(test_values));
        t_test = t(rand(test_values));

        p = polyfit(x_training, t_training, polyfit_degrees(k));
        plot(xrange, polyval(p, xrange));
        error_polyfit(k, l) = mean((polyval(p, x_test) - x_test) .^ 2);
    end
    error_polymean = sqrt(mean(error_polyfit(:, :)'));
end

figure;
scatter(number_of_nodes, root_mean_squared_error, 'fill', 'k');
title('Root mean square error of MLP with 5-fold cross validation');
ylabel('RMS error');
xlabel('Number of hidden nodes');
figure;
scatter(polyfit_degrees, error_polymean, 'fill', 'r')
title('Root mean squared error of polyfit error')
xlabel('Polyfit degree')
ylabel('RMS error')

%% irregular
clear all; close all; clc;
irregular = importdata('data/irregular.mat');
subplot(1, 3, 1);
gscatter(irregular.x, irregular.t);
title('Irregular');

xrange = linspace(0, 12, 100);
% preprocessing data
t = irregular.t ./ 200;
x = irregular.x - ones(91, 1) * min(irregular.x);

number_of_nodes = [2, 3, 4, 5, 7, 10, 15, 25];

rand = randperm(91);

figure;
for i = 1:length(number_of_nodes)    
    subplot(length(number_of_nodes) / 2, 2, i);
    hold on;
    for j = 1:5 % 5-fold cross validation
        test_values = (((j - 1) * 18 + 1:(j - 1) * 18 + 18));
        train_values = linspace(1, 91, 91);
        train_values(test_values) = [];
        x_training = x(rand(train_values));
        t_training = t(rand(train_values));
        x_test = x(rand(test_values));
        t_test = t(rand(test_values));

        nhidden = number_of_nodes(i);
        nout = 1;
        c = 60;

        net = mlp(1, nhidden, nout, 'logistic'); 
        options = zeros(1,18);
        options(1) = 1; 
        options(14) = c;
        options(18) = 0.01;
        [net] = netopt(net, options, x_training, t_training, 'scg');
        yg = mlpfwd(net, xrange');
        d = mlpfwd(net, x_test); 
        error(i, j) = mean((d - t_test) .^ 2);
        plot(xrange, yg);
    end
    scatter(x, t);
    hold off;
    title(sprintf('MLP with %d hidden nodes', number_of_nodes(i)));
    root_mean_squared_error(i) = sqrt(mean(error(i, :)'));
end

polyfit_degrees= [1, 2, 3, 4, 5, 10];
figure;
for k = 1:length(polyfit_degrees)
    subplot(1, length(polyfit_degrees), k)
    hold on;
    scatter(x,t,'r','fill');
    for l = 1:5 % 5-fold verification
        test_values = (((l - 1) * 18 + 1:(l - 1) * 18 + 18));
        train_values = linspace(1,91,91);
        train_values(test_values) = [];
        x_training = x(rand(train_values));
        t_training = t(rand(train_values));
        x_test = x(rand(test_values));
        t_test = t(rand(test_values));

        p = polyfit(x_training, t_training, polyfit_degrees(k));
        plot(xrange, polyval(p, xrange));
        error_polyfit(k, l) = mean((polyval(p, x_test) - x_test) .^ 2);
    end
    error_polymean = sqrt(mean(error_polyfit(:, :)'))
end

figure;
scatter(number_of_nodes, root_mean_squared_error, 'fill', 'k');
title('Root mean square error of MLP with 5-fold cross validation');
ylabel('RMS error');
xlabel('Number of hidden nodes');
figure;
scatter(polyfit_degrees, error_polymean, 'fill', 'r')
title('Root mean squared error of polyfit error')
xlabel('Polyfit degree')
ylabel('RMS error')
