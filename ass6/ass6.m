close all;

samples = 1000;
if exist('star', 'var') == 0
    disp('Generating dataset.')
    [star, xv, yv] = generate_star(samples);
end

test_map_sizes = true;
test_learning_rates = true;

nin = 2;
map_sides = [2:10 11:2:21];
map_size = [16, 16];
options(1) = 0;     % Don't show iteration info
options(5) = 1;     % Random sampled
options(14) = 400;  % # of iterations 
options(15) = 1;    % Final neighborhood size
options(16) = 0.05;  % Final learning rate
options(17) = round(sqrt(map_size(1,2)^2 + map_size(1,1)^2)/2); % initial neighborhood size
options(18) = 0.9;  % Initial learning rate

if test_map_sizes
    errors = [map_sides', zeros(length(map_sides), 1)];
    figure
    for i = 1:length(map_sides);
        map_side = map_sides(i);
        map_size = [map_side, map_side];

        disp(['Doing SOM with square map of side ', num2str(map_side)])

        options(17) = round(sqrt(map_size(1,2)^2 + map_size(1,1)^2)/2); % initial neighborhood size

        net = som(nin, map_size);
        net = somtrain(net, options, star);
        c = sompak(net);
        map = net.map;

        subplot(length(map_sides) / 3, 3, i)
        % plot(c(:, 1), c(:, 2), '-o')
        plot_lattice(map, star)
        title(['map\_size = [', num2str(map_size(1,1)), ', ', num2str(map_size(1,2)), ']'])

        for j = 1:size(c, 1);
            if ~ inpolygon(c(j, 1), c(j, 2), xv, yv)
                errors(i, 2) = errors(i, 2) + 1;
            end
        end
        errors(i, 2) = errors(i, 2) / errors(i, 1)^2;
    end

    figure
    plot(errors(:, 1), errors(:, 2))
    xlabel('Square map size')
    ylabel('Neurons inside training star shape')
end

figure
subplot(1, 4, 1)
plot(star(:, 1), star(:, 2), 'o')
title('Generated dataset')
disp('Doing SOM with square map of side 16')
net1 = som(nin, map_size);
map1 = net1.map;
subplot(1, 4, 2)
plot_lattice(map1, star)
title(['SOM before training _{map\_size=[', num2str(map_size(1)), ', ', num2str(map_size(2)), ']}'])
options(14) = 50;  % # of iterations 
options(16) = 0.5;  % Final learning rate
options(18) = 0.9;  % Initial learning rate
net2 = somtrain(net1, options, star);
map2 = net2.map;
subplot(1, 4, 3)
plot_lattice(map2, star)
title(['SOM after ordering _{map\_size=[', num2str(map_size(1)), ', ', num2str(map_size(2)), ']}'])
options(14) = 400;   % # of iterations 
options(16) = 0.05;  % Final learning rate
options(18) = 0.5;  % Initial learning rate
net3 = somtrain(net2, options, star);
map3 = net3.map;
subplot(1, 4, 4)
plot_lattice(map3, star)
title(['SOM after convergence _{map\_size=[', num2str(map_size(1)), ', ', num2str(map_size(2)), ']}'])


if test_learning_rates
    map_size = [16, 16];
    % Testing initial learning rate
    init_learnings = .9:-.1:.1;
    options(16) = .05;  % Final learning rate
    figure
    for i = 1:length(init_learnings);
        init_learning = init_learnings(i);
        disp(['Doing SOM with map of size [', num2str(map_size), '] and initial learning rate ', num2str(init_learning)])

        options(18) = init_learning;  % Initial learning rate
        net = som(nin, map_size);
        net = somtrain(net, options, star);
        map = net.map;
        subplot(ceil(length(init_learnings) / 3), 3, i)
        plot_lattice(map, star)
        title(['map\_size=[', num2str(map_size), '] init. learning rate ', num2str(init_learning)])
    end

    % Testing final learning rate
    final_learnings = .9:-.1:.1;
    figure
    options(18) = 0.9;  % Initial learning rate
    for i = 1:length(final_learnings);
        final_learning = final_learnings(i);
        disp(['Doing SOM with map of size [', num2str(map_size), '] and final learning rate ', num2str(final_learning)])

        options(16) = final_learning;  % Final learning rate
        net = som(nin, map_size);
        net = somtrain(net, options, star);
        map = net.map;
        subplot(ceil(length(init_learnings) / 3), 3, i)
        plot_lattice(map, star)
        title(['map\_size=[', num2str(map_size), '] final learning rate ', num2str(final_learning)])
    end

    % Testing equal learning rate
    learnings = .9:-.1:.1;
    figure
    for i = 1:length(learnings);
        learning = learnings(i);
        disp(['Doing SOM with map of size [', num2str(map_size), '] and learning rate ', num2str(learning)])

        options(18) = learning;  % Initial learning rate
        options(16) = learning;  % Final learning rate
        net = som(nin, map_size);
        net = somtrain(net, options, star);
        map = net.map;
        subplot(ceil(length(init_learnings) / 3), 3, i)
        plot_lattice(map, star)
        title(['map\_size=[', num2str(map_size), '] learning rate ', num2str(learning)])
    end
end


disp('\nPlease wait for the graphs to load...')
