function [ correct ] = testweights( dataset, weights )

    correct = 0;
    for i = 1:size(dataset, 1);
      class = dataset(i, 3);
      if class == 0
        class = -1;
      end
      input = [ 1 dataset(i, 1:2) ];
      net_out = sign(input * weights');
      if class == net_out
        correct = correct + 1;
      end  
    end

end

