clear; close all; clc;

load('pics.mat');

[n_pics, nin] = size(pics);

pics2D = reshape_pics(pics, 56, 46);

n_components = 20;
[coeff, pc] = pca(pics(:, :), n_components);

% Encode
pics_enc = zeros(n_pics, n_components);
for i = 1:n_pics;
    pics_enc(i, :) = pics(i, :) * pc;
end

% Decode
pics_dec = zeros(n_pics, nin);
for i = 1:n_pics;
    pics_dec(i, :) = pics_enc(i, :) * pc';
end

pics_dec2D = reshape_pics(pics_dec, 56, 46);

% Show first two and 200th
subplot(2, 2, 1)
imagesc(pics2D(:, :, 1))
subplot(2, 2, 2)
imagesc(pics_dec2D(:, :, 1))
subplot(2, 2, 3)
imagesc(pics2D(:, :, 200))
subplot(2, 2, 4)
imagesc(pics_dec2D(:, :, 200))
