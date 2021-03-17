%% 
clear all; close all; clc

% Reproduce music score for GNR
% figure(1)
[y, Fs] = audioread('GNR.m4a');
n = length(y); % Fourier modes
trgnr = n / Fs; % record time in seconds
t = (1:n) / Fs;
% plot(t, y);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Sweet Child O'' Mine');
% p8 = audioplayer(y,Fs); playblocking(p8);

k = (1 / trgnr) * [0:(n/2 - 1) (-n/2):-1]; % frequency component
ks = fftshift(k);

tau = 0:0.1:trgnr;
a = 1000;
S = y';
Sgt_spec = zeros(n, length(tau));

for j = 1:length(tau)
    g = exp(-a * (t - tau(j)).^2); % Window function
    Sg = g .* S;
    Sgt = fft(Sg);
    [maximum, index] = max(abs(Sgt));
    Sgtf = Sgt .* exp(-0.01 * (k - k(index)).^2); % filter overtone
    Sgt_spec(:, j) = fftshift(abs(Sgtf)); % We don't want to scale it
end

figure(2)
yyaxis left
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0 1000],'Fontsize',10)
yticks([277.18, 369.99, 415.30, 554.37, 698.46, 739.99])
yticklabels({'#C', '#F', '#G', '#C', 'F', '#F'})
colormap(hot)
xlabel('time (t)'), ylabel('Notes')

yyaxis right
set(gca, 'ylim', [0, 1000], 'Fontsize', 10);
ylabel('Frequency (k) in Hz')


%% 

% Reproduce music score for Floyd
clear all; close all; clc
% figure(1)
[y, Fs] = audioread('Floyd.m4a');
n = length(y); % Fourier modes
trgnr = n / Fs; % record time in seconds
t = (1:n) / Fs;
% plot(t, y);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Comfortably Numb');
% p8 = audioplayer(y,Fs); playblocking(p8);

k = (1 / trgnr) * [0:(n/2 - 1) (-n/2):-1]; % frequency component
ks = fftshift(k);

tau = 0:1:trgnr;
a = 6000;
S = y';
Sgt_spec = zeros(n - 1, length(tau));

for j = 1:length(tau)
    g = exp(-a * (t - tau(j)).^2); % Window function
    Sg = g .* S;
    Sgt = fft(Sg);
    
    Sgt = Sgt(1:n-1);
    [maximum, index] = max(abs(Sgt));
    filter = exp(-0.01 * (k - k(index)).^2);
    Sgtf = Sgt .* filter; % filter overtone with a Gaussian filter
    Sgtf(k > 250) = 0; % filter out any frequency higher than 250
    Sgtf(k < 60) = 0; % filter out any frequency lower than 60
    Sgt_spec(:, j) = fftshift(abs(Sgtf)); % We don't want to scale it
end

figure(3)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0 500],'Fontsize',10)
yticks([87.307, 110.00, 123.47, 185.00, 246.94])
yticklabels({'F','A', 'B', '#F', 'B'})
colormap(hot)
xlabel('time (t)'), ylabel('Notes')

yyaxis right
set(gca, 'ylim', [0 500], 'Fontsize', 10);
ylabel('Frequency (k) in Hz')



%%

clear all; close all; clc
% figure(1)
[y, Fs] = audioread('Floyd.m4a');
n = length(y); % Fourier modes
samples=[1,n - (50 * Fs)];
[y,Fs] = audioread('Floyd.m4a',samples);
n = length(y); % Fourier modes
trgnr = n / Fs; % record time in seconds
t = (1:n) / Fs;
% plot(t, y);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Comfortably Numb');
% p8 = audioplayer(y,Fs); playblocking(p8);

k = (1 / trgnr) * [0:(n/2 - 1) -n/2:-1]; % frequency component
ks = fftshift(k);

tau = 0:0.1:trgnr;
a = 6000;
S = y';
Sgt_spec = zeros(n-1, length(tau));

for j = 1:length(tau)
    g = exp(-a * (t - tau(j)).^2); % Window function
    Sg = g .* S;
    % Sg_lowpass = lowpass(Sg, 250, Fs);
    Sgt = fft(Sg);
    
    Sgt = Sgt(1:n-1);
    [maximum, index] = max(abs(Sgt));
    filter = exp(-0.01 * (k - k(index)).^2);
    Sgtf = Sgt .* filter; % filter overtone with a Gaussian filter
    bass_notes = Sgtf;
    bass_notes(k > 250) = 0; % filter out any frequency higher than 250
    bass_notes(k < 60) = 0; % filter out any frequency lower than 60
    % subtract bass from the music to get the guitar
    guitar_notes = Sgtf - bass_notes;
    Sgt_spec(:, j) = fftshift(abs(guitar_notes)); % We don't want to scale it
end

figure(4)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0 1200],'Fontsize',10)
yticks([369.99, 493.88, 622.25, 880.00, 987.77])
yticklabels({'#F','B', '#D', 'A', 'B'})
colormap(hot)
xlabel('time (t)'), ylabel('Notes')

yyaxis right
set(gca, 'ylim', [0 1200], 'Fontsize', 10);
ylabel('Frequency (k) in Hz')