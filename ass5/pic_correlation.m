function [ r ] = pic_correlation( a, b )

a = a - mat_mean(a);
b = b - mat_mean(b);
r = sum(sum(a.*b)) / sqrt(sum(sum(a.*a)) * sum(sum(b.*b)));

end

function y = mat_mean(x)
  y = sum(x(:)) / numel(x);
end
