function plot_lattice( map, training )

hold on
plot(training(:, 1), training(:, 2), 'o', 'markersize', 3)

[~, mapw, maph] = size(map);

for i = 1:mapw-1;
    for j = 1:maph-1;
        scatter([map(1, i, j) map(1, i+1, j)], [map(2, i, j) map(2, i+1, j)], 20, 'MarkerFaceColor', 'y')
        scatter([map(1, i, j) map(1, i, j+1)], [map(2, i, j) map(2, i, j+1)], 20, 'MarkerFaceColor', 'y')
        
        
        plot([map(1, i, j) map(1, i+1, j)], [map(2, i, j) map(2, i+1, j)], 'r-')
        plot([map(1, i, j) map(1, i, j+1)], [map(2, i, j) map(2, i, j+1)], 'r-')
    end
end

hold off

end

