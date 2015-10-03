clear; close all; clc;

load('pics.mat');

[n_pics, nin] = size(pics);

pics2D = reshape_pics(pics, 56, 46);

% Test different number of principal components
components_to_try = [1:60 61:5:199 200:100:2000 nin];
mean_correlations = zeros(length(components_to_try), 1);
for n = 1:length(components_to_try);
    n_components = components_to_try(n);
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
    
    correlations = zeros(n_pics, 1);
    for i = 1:n_pics;
        % function y = mean2(x)
        %   y = sum(x(:),'double') / numel(x);
        % end
        % function r = corr2(a, b)
        %   a = a - mean2(a);
        %   b = b - mean2(b);
        %   r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));
        % end
        correlations(i) = corr2(pics2D(:, :, 1), pics_dec2D(:, :, 1));
    end
    mean_correlations(n) = mean(correlations);
    disp(['Mean correlation with ', num2str(n_components), ' components: ', num2str(mean_correlations(n))])
end

subplot(1, 2, 1)
plot(components_to_try, mean_correlations, 'o-')
xlabel('Number of principal components')
ylabel('Mean correlation of images')

subplot(1, 2, 2)
semilogx(components_to_try, mean_correlations, 'o-')
xlabel('Number of principal components')
ylabel('Mean correlation of images')

% Show first two and 200th
% figure
% subplot(2, 2, 1)
% imagesc(pics2D(:, :, 1))
% subplot(2, 2, 2)
% imagesc(pics_dec2D(:, :, 1))
% subplot(2, 2, 3)
% imagesc(pics2D(:, :, 200))
% subplot(2, 2, 4)
% imagesc(pics_dec2D(:, :, 200))
