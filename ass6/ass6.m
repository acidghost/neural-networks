close all;

samples = 1000;
if exist('star', 'var') == 0
    disp('Generating dataset.')
    [star, xv, yv] = generate_star(samples);
end
subplot(1, 3, 1)
plot(star(:, 1), star(:, 2), 'o')
title('Generated dataset')

nin = 2;
map_sides = [2:10 11:2:21];
map_size = [32, 32];
options(5) = 1;     % Random sampled
options(14) = 400;  % # of iterations 
options(15) = 1;    % Final neighborhood size
options(16) = 0.05;  % Final learning rate
options(17) = round(sqrt(map_size(1,2)^2 + map_size(1,1)^2)/2); % initial neighborhood size
options(18) = 0.5;  % Initial learning rate

disp('Doing SOM with square map of side 32')
net1 = som(nin, map_size);
subplot(1, 3, 2)
plot(c1(:, 1), c1(:, 2), '-o')
title(['SOM before training _{map\_size=[', num2str(map_size(1)), ', ', num2str(map_size(2)), ']}'])
net2 = somtrain(net1, options, star);
c2 = sompak(net2);
subplot(1, 3, 3)
plot(c2(:, 1), c2(:, 2), '-o')
title(['SOM after training _{map\_size=[', num2str(map_size(1)), ', ', num2str(map_size(2)), ']}'])

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
    
    subplot(length(map_sides) / 3, 3, i)
    plot(c(:, 1), c(:, 2), '-o')
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
