clear; close all; clc;

load('pics.mat');

[n_pics, nin] = size(pics);

pics2D = reshape_pics(pics, 56, 46);

% Test different number of principal components
components_to_try = [1:60 61:5:199 200:100:2000 nin];   % # of PC to test
pic_to_show = 1;    % Show first picture
pic_to_show2 = 200; % Show 200th picture
pics_to_take = [2, 12, 60, 96, 500, nin];   % which PCA to show
decoded_pics = zeros(56, 46, length(pics_to_take));
decoded_pics2 = zeros(56, 46, length(pics_to_take));
pics_to_take_i = 1;
mean_correlations = zeros(length(components_to_try), 1);
correlations_to_show = zeros(length(pics_to_take), 1);
correlations_to_show2 = zeros(length(pics_to_take), 1);
for n = 1:length(components_to_try);
    n_components = components_to_try(n);
    [~, pc] = pca(pics(:, :), n_components);
    
    % Encode
    pics_enc = pics * pc;

    % Decode
    pics_dec = pics_enc * pc';

    pics_dec2D = reshape_pics(pics_dec, 56, 46);
    
    correlations = zeros(n_pics, 1);
    for i = 1:n_pics;
        correlations(i) = pic_correlation(pics2D(:, :, i), pics_dec2D(:, :, i));
    end
    mean_correlations(n) = mean(correlations);
    disp(['Mean correlation with ', num2str(n_components), ' components: ', num2str(mean_correlations(n))])
    
    if n_components == pics_to_take(pics_to_take_i)
        decoded_pics(:, :, pics_to_take_i) = pics_dec2D(:, :, pic_to_show);
        decoded_pics2(:, :, pics_to_take_i) = pics_dec2D(:, :, pic_to_show2);
        correlations_to_show(pics_to_take_i) = correlations(pic_to_show);
        correlations_to_show2(pics_to_take_i) = correlations(pic_to_show2);
        pics_to_take_i = pics_to_take_i + 1;
    end
end

subplot(1, 2, 1)
plot(components_to_try, mean_correlations, 'o-')
xlabel('Number of principal components')
ylabel('Mean correlation of images')

subplot(1, 2, 2)
semilogx(components_to_try, mean_correlations, 'o-')
xlabel('Number of principal components')
ylabel('Mean correlation of images')

figure
for i = 1:length(pics_to_take);
    n_components = pics_to_take(i);
    subplot(length(pics_to_take) / 3, 3, i)
    imagesc(decoded_pics(:,:,i))
    title([num2str(n_components), ' PCs _{correlation = ', num2str(correlations_to_show(i)), '}'])
end
figure; imagesc(pics2D(:, :, pic_to_show))
title('Original image')

figure
for i = 1:length(pics_to_take);
    n_components = pics_to_take(i);
    subplot(length(pics_to_take) / 3, 3, i)
    imagesc(decoded_pics2(:,:,i))
    title([num2str(n_components), ' PCs _{correlation = ', num2str(correlations_to_show2(i)), '}'])
end
figure; imagesc(pics2D(:, :, pic_to_show2))
title('Original image')

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
