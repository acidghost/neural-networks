irregular = importdata('data/irregular.mat');
line = importdata('data/line.mat');
sinus = importdata('data/sinus.mat');

subplot(1, 3, 1)
gscatter(irregular.x, irregular.t)
title('Irregular')

subplot(1, 3, 2)
gscatter(line.x, line.t)
title('Line')

subplot(1, 3, 3)
gscatter(sinus.x, sinus.t)
title('Sinus')

xrange=linspace(0,1,100);
number_of_nodes = [2, 3, 4, 5, 7, 10, 15, 25];

figure;
for i = 1:length(number_of_nodes)
    M = [1:5; 6:10; 11:15; 16:20; 21:25];
    subplot(length(number_of_nodes) / 2, 2, i)
    hold on
    for j = 1:5
        A = linspace(1, 25, 25);
        A(A == M(j, 1)) = [];
        A(A == M(j, 2)) = [];
        A(A == M(j, 3)) = [];
        A(A == M(j, 4)) = [];
        A(A == M(j, 5)) = [];

        x_training = line.x(A, :);
        t_training = line.t(A, :);
        x_test = line.x(M(j, :), :);
        t_test = line.t(M(j, :), :);

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

        error(i, j) = mean((d - t_test) .^2);
        plot(xrange, yg)
    end

    scatter(x, t);
    hold off;
    title(strcat('MLP with hidden layer of ', num2str(number_of_nodes(i)), ' nodes'))
    RMS(i) = sqrt(mean(error(i, :)'));
end


figure;
scatter(number_of_nodes, RMS, 'fill', 'k');
title('Root mean square error of MLP solution (5 fold cross validation)');
ylabel('RMS error');
xlabel('Number of hidden nodes');