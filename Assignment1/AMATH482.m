% Clean workspace2
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1);
x = x2(1:n);
y = x;
z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; % frequency component
ks = fftshift(k);

[X,Y,Z]=meshgrid(x, y, z);
[Kx,Ky,Kz]=meshgrid(ks, ks, ks);

ave = zeros(n,n,n); % summation for average
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j), n, n, n);
    Utn = fftn(Un);
    ave = ave + Utn;
    M = max(abs(Utn), [], 'all');
    % close all, isosurface(X, Y, Z, abs(Un) / M, 0.4)
    % axis([-20 20 -20 20 -20 20])
    % grid on
    % drawnow
    % pause(1)
end

figure(1)
isosurface(X, Y, Z, fftshift(abs(Utn)) / M, 0.5)
axis([-20 20 -20 20 -20 20])
grid on
drawnow

ave = abs(fftshift(ave)) / 49; % average to recover the noisy signal
[maximum, index] = max(ave(:));
[r,c,p] = ind2sub(size(ave),index); % locate the signal

figure(2)
isosurface(Kx, Ky, Kz, ave / max(ave(:)), 0.4)
axis([-20 20 -20 20 -20 20]), grid on, drawnow

kx0 = Kx(r, c, p);
ky0 = Ky(r, c, p);
kz0 = Kz(r, c, p);

tau = 0.2;
filter = exp(-tau * ((Kx - kx0).^2 + (Ky - ky0).^2 + (Kz - kz0).^2));

x_pos = zeros(49,1);
y_pos = zeros(49,1);
z_pos = zeros(49,1);

for j=1:49
    unft(:,:,:)=reshape(subdata(:,j), n, n, n);
    unft = fftn(unft) .* fftshift(filter); %applying filter
    unft = fftshift(unft);
    unf = ifftn(unft); % transform back to time domain
    [maximum,index] = max(abs(unf(:)));
    [x,y,z] = ind2sub(size(unf),index);
    x_pos(j) = X(x, y, z);
    y_pos(j) = Y(x, y, z);
    z_pos(j) = Z(x, y, z);
end

figure(3)
plot3(x_pos, y_pos, z_pos, '-o', 'MarkerIndices',1)

tracker_pos = [x_pos, y_pos];
