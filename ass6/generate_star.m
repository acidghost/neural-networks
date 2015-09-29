function [star, xv, yv] = generate_star(samples)

theta = 0:4/5*pi:4*pi;
xv = cos(theta);
yv = sin(theta);
min_x = min(xv);
max_x = max(xv);
min_y = min(yv);
max_y = max(yv);

star = zeros(samples, 2);
i = 1;
while i < samples
    x = (max_x - min_x) * rand + min_x;
    y = (max_y - min_y) * rand + min_y;
    if inpolygon(x, y, xv, yv)
        star(i, :) = [x, y];
        i = i+1;
    end
end

% save('star.mat', 'star');

end
