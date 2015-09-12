function boundaries_with_errors = calc_boundaries_errors( dataset, costs )

    classes = size(dataset, 1);
    values = size(dataset, 2);
    possible_boundaries = values - 1;
    boundaries_with_errors = zeros(classes, possible_boundaries);
    
    for boundary = 1:possible_boundaries;
        boundaries_with_errors(1, boundary) = costs(1) * sum(dataset(1, boundary:possible_boundaries+1));
        boundaries_with_errors(2, boundary) = costs(2) * sum(dataset(2, 1:boundary-1));
    end

end