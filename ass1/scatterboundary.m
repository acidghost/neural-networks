function scatterboundary( dataset, weights )

    hold on
    gscatter(dataset(:, 1), dataset(:, 2), dataset(:, 3))
    x = linspace(min(dataset(:, 1)), max(dataset(:, 1)));
    y = x * (-weights(2) / weights(3)) - (weights(1) / weights(3));
    plot(x, y)
    hold off

end

