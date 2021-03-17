close all; clear all; clc;
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')

implay(vidFrames1_3)
% implay(vidFrames2_3)
% implay(vidFrames3_3)

numFrames1_3 = size(vidFrames1_3,4);
numFrames2_3 = size(vidFrames2_3,4);
numFrames3_3 = size(vidFrames3_3,4);

width = 50;
filter = zeros(480,640);
filter(300 - 3 * width:1:300 + 3 * width, 350 - width:1:350 + width) = 1;

cam_1 = zeros(numFrames1_3, 2);
for j = 1:numFrames1_3
    X = vidFrames1_3(:,:,:,j);
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

cam_2 = zeros(numFrames2_3, 2);
for j = 1:numFrames2_3
    X = vidFrames2_3(:,:,:,j);
    X_gray = rgb2gray(X);
    % X_gray = im2double(X_gray);
    % X_gray = X_gray .* filter;
    
    % X_gray = im2uint8(X_gray);
    index = find(X_gray > 249);
    [row, col] = ind2sub(size(X_gray), index);
    cam_2(j,:) = [mean(col), mean(row)];
    % X_gray(X_gray < 249) = 0;
    % imshow(X_gray); drawnow
end

width = 50;
filter3 = zeros(480,640);
filter3(240 - 2 * width:1:240 + 2 * width, 320 - 3 * width:1:320 + 3 * width) = 1;

cam_3 = zeros(numFrames3_3, 2);
for j = 1:numFrames3_3
    X = vidFrames3_3(:,:,:,j);
    X_gray = rgb2gray(X);
    X_gray = im2double(X_gray);
    X_gray = X_gray .* filter3;
    X_gray = im2uint8(X_gray);
    
    index = find(X_gray > 246);
    [row, col] = ind2sub(size(X_gray), index);
    cam_3(j,:) = [mean(col), mean(row)];
    % X_gray(X_gray < 246) = 0;
    % imshow(X_gray); drawnow
end

[minimum, ind] = min(cam_1(1:20, 2));
cam_1 = cam_1(ind:end,:);

[minimum,ind] = min(cam_2(1:20,2));
cam_2  = cam_2(ind:end,:);

[minimum,ind] = min(cam_3(1:20,2));
cam_3  = cam_3(ind:end,:);

cam_1 = cam_1(1:length(cam_3), :);
cam_2 = cam_2(1:length(cam_3), :);

cam_all = [cam_1'; cam_2'; cam_3']; % X
mean_all = mean(cam_all, 2);
cam_all_centered = cam_all - mean_all; % center

[u,s,v] = svd(cam_all_centered / sqrt(length(cam_1) - 1), 'econ'); % SVD
lambda = diag(s).^2;
Y = u' * cam_all_centered; % projection

figure(1)
plot(1:6, lambda/sum(lambda), 'mo', 'Linewidth', 2);
title("Case 3: Energy of each Diagonal Variance");
xlabel("Diagonal Variances"); ylabel("Energy Captured");

figure(2)
subplot(2,1,1)
plot(1:237, cam_all_centered(2,:),1:237, cam_all_centered(1,:))
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Case 3: Displacement Across z-axis and xy-plane");
legend("Z", "XY")
subplot(2,1,2)
plot(1:237, Y(1,:), 1:237, Y(2,:), 1:237, Y(3,:), 1:237, Y(4,:))
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Case 3: Displacement across principal component directions");
legend("PC1", "PC2", "PC3", "PC4")