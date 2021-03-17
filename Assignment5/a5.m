% Clear workspace
clear all; close all; clc;
    
%% import videos
vid_ski_drop = VideoReader('ski_drop_low.mp4');
% vid_ski_drop = VideoReader('monte_carlo_low.mp4');

dt = 1/vid_ski_drop.Framerate;
t = 0:dt:vid_ski_drop.Duration;
vidFrames_ski = read(vid_ski_drop);
numFrames_ski = get(vid_ski_drop,'numberOfFrames');

frame = im2double(vidFrames_ski(:,:,:,1));
X = zeros(size(frame,1) * size(frame,2), numFrames_ski);

for j = 1:numFrames_ski
    frame = vidFrames_ski(:,:,:,j);
    frame = im2double(rgb2gray(frame));
    
    X(:,j) = reshape(frame, 540 * 960, []);
    % imshow(frame); drawnow
end

%% DMD
X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U, Sigma, V] = svd(X1,'econ');
lambda = diag(Sigma).^2;

% Plot SVD Results
% Singular values1
figure(1)
plot(diag(Sigma),'ko','Linewidth',2)
ylabel('\sigmaj')

figure(2)
plot(1:453, lambda/sum(lambda), 'mo', 'Linewidth', 2);
title("Energy of each Diagonal Variance (Ski-Drop)");
xlabel("Diagonal Variances"); ylabel("Energy Captured");

%%
r = 1;
U = U(:, 1:r);
Sigma = Sigma(1:r, 1:r);
V = V(:, 1:r);
 
%%
S = U' * X2 * V * diag(1 ./ diag(Sigma));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu) / dt;
Phi = U * eV;

%% DMD solution
y0 = Phi \ X1(:,1); % pseudoinverse to get initial conditions
umodes = zeros(length(y0), length(t) - 1);
for iter = 1:length(t) - 1
    umodes(:,iter) = y0 .* exp(omega * t(iter));
end
udmd = Phi * umodes;

%% Sparse
X_sparse = X1 - abs(udmd(:,size(udmd, 2)));
neg_vals = X_sparse < 0;
R = X_sparse .* neg_vals;

udmd_new = R + abs(udmd(:,size(udmd, 2)));
X_sparse_new = X_sparse - R;

X_reconstructed = X_sparse_new + udmd_new;

%% Show
temp =  reshape(X_reconstructed, [size(frame, 1), size(frame, 2), length(t) - 1]);
imshow(im2uint8(temp(:,:,150)))
for i = 1:453
    imshow(im2uint8(temp(:,:,i)))
end
title("Total Reconstruction (Sparse + Low Rank)");