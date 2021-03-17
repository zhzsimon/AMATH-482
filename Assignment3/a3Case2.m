close all; clear all; clc;
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')

% implay(vidFrames1_2)
% implay(vidFrames2_2)
% implay(vidFrames3_2)

numFrames1_2 = size(vidFrames1_2,4);
numFrames2_2 = size(vidFrames2_2,4);
numFrames3_2 = size(vidFrames3_2,4);

width = 50;
filter = zeros(480,640);
filter(300 - 3 * width:1:300 + 3 * width, 350 - width:1:350 + width) = 1;

cam_1 = zeros(numFrames1_2, 2);
for j = 1:numFrames1_2
    X = vidFrames1_2(:,:,:,j);
    X_gray = rgb2gray(X);
    X_gray = im2double(X_gray);
    X_gray = X_gray .* filter;
    
    X_gray = im2uint8(X_gray);
    index = find(X_gray > 250);
    [row, col] = ind2sub(size(X_gray), index);
    cam_1(j,:) = [mean(col), mean(row)];
    % X_gray(X_gray < 250) = 0;
    % imshow(X_gray); drawnow
end

width = 50;
filter2 = zeros(480,640);
filter2(240 - 4 * width:1:240+4 * width, 320 - 3 * width:1:320 + 3 * width) = 1;

cam_2 = zeros(numFrames2_2, 2);
for j = 1:numFrames2_2
    X = vidFrames2_2(:,:,:,j);
    X_gray = rgb2gray(X);
    X_gray = im2double(X_gray);
    X_gray = X_gray .* filter2;
    
    X_gray = im2uint8(X_gray);
    index = find(X_gray > 249);
    [row, col] = ind2sub(size(X_gray), index);
    cam_2(j,:) = [mean(col), mean(row)];
    % X_gray(X_gray < 249) = 0;
    % imshow(X_gray); drawnow
end

width = 50;
filter3 = zeros(480,640);
filter3(240 - 2 * width:1:240 + 2 * width, 320 - 2 * width:1:320 + 2 * width) = 1;

cam_3 = zeros(numFrames3_2, 2);
for j = 1:numFrames3_2
    X = vidFrames3_2(:,:,:,j);
    X_gray = rgb2gray(X);
    X_gray = im2double(X_gray);
    X_gray = X_gray .* filter3;
    X_gray = im2uint8(X_gray);
    
    index = find(X_gray > 245);
    [row, col] = ind2sub(size(X_gray), index);
    cam_3(j,:) = [mean(col), mean(row)];
    % X_gray(X_gray < 245) = 0;
    % imshow(X_gray); drawnow
end

[minimum, ind] = min(cam_1(1:20, 2));
cam_1 = cam_1(ind:end,:);

[minimum,ind] = min(cam_2(1:20,2));
cam_2  = cam_2(ind:end,:);

[minimum,ind] = min(cam_3(1:20,2));
cam_3  = cam_3(ind:end,:);

cam_2 = cam_2(1:length(cam_1), :);
cam_3 = cam_3(1:length(cam_1), :);

cam_all = [cam_1'; cam_2'; cam_3']; % X
mean_all = mean(cam_all, 2);
cam_all_centered = cam_all - mean_all; % center

[u,s,v] = svd(cam_all_centered / sqrt(length(cam_1) - 1), 'econ'); % SVD
lambda = diag(s).^2;
Y = u' * cam_all_centered; % projection

figure(1)
plot(1:6, lambda/sum(lambda), 'mo', 'Linewidth', 2);
title("Case 2: Energy of each Diagonal Variance");
xlabel("Diagonal Variances"); ylabel("Energy Captured");

figure(2)
subplot(2,1,1)
plot(1:295, cam_all_centered(2,:),1:295, cam_all_centered(1,:))
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Case 2: Displacement Across z-axis and xy-plane");
legend("Z", "XY")
subplot(2,1,2)
plot(1:295, Y(1,:), 1:295, Y(2,:))
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Case 2: Displacement across principal component directions");
legend("PC1", "PC2")