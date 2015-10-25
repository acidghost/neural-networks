function [ angles ] = compute_angles( points, source)

    angles = zeros(size(points, 1), 1);
    
    for i = 1:size(points, 1);
        point = points(i, :);
        hypo = pdist2(point, source);
        opposite = point(1, 2) - source(1, 2);
        angles(i) = asin(opposite / hypo);
    end

end
