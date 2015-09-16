function dataset = get_data(n, scale)
    dataset = zeros(n, 3);
    epsilon = scale / 20;

    i = 1;
    while i <= n
       rand_x = rand * scale;
       rand_y = rand * scale;
       if rand_y > (scale / 2) + epsilon && rand_x > (scale / 2) + epsilon 
       %if rand_y > (scale / 2) + epsilon
           dataset(i,:) = [ rand_x, rand_y, 1 ];
       elseif rand_y < (scale / 2) - epsilon  && rand_x < (scale / 2) - epsilon
       %elseif rand_y < (scale / 2) - epsilon
           dataset(i,:) = [ rand_x, rand_y, -1 ];
       else
           i = i - 1;   
       end
       i = i + 1;
    end
end
