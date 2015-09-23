function reshaped = reshape_pics(pictures, M, N)

n_pics = size(pictures, 1);
reshaped = zeros(M, N, n_pics);
for i = 1:n_pics;
    reshaped(:, :, i) = reshape(pictures(i, :), M, N);
end

end
