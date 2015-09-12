function dataset = gen_dataset( samples, values )

    num_values = size(values, 2);
    mean_values = mean(values);
    epsilon = max(values) / 4;
    mu1 = mean_values - epsilon;
    mu2 = mean_values + epsilon;
    sigma = mean_values - 1.4 * epsilon;
    dataset = zeros(2, num_values);
    i = 1;
    while i <= samples;
       if rand() < .5
           val = normrnd(mu1, sigma);
           class = 1;
       else
           val = normrnd(mu2, sigma);
           class = 2;
       end
       if val <= max(values) && val >= min(values)
          val = round(val);
          dataset(class, val) = dataset(class, val) + 1;
          i = i+1;
       end
    end

end

