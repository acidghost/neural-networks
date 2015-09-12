function boundaries_with_errors = calc_boundaries_errors( posteriors )

    classes = size(posteriors, 1);
    values = size(posteriors, 2);
    possible_boundaries = values - 1;
    boundaries_with_errors = zeros(classes, possible_boundaries);
    
    for boundary = 1:possible_boundaries;
        boundaries_with_errors(1, boundary) = sum(posteriors(1, 1:boundary-1));
        boundaries_with_errors(2, boundary) = sum(posteriors(2, boundary:possible_boundaries));
    end

end

