close all
clear variables

% Choose what data set to use
% 0: Dataset we collected
% 1: Dataset from IEEE
dataset_select = 0;
n_sec = 15; % Duration of STFT in seconds
Fs = 256; % Sample Frequency in Hz
f0 = 5e9; % baseband frequency
if(dataset_select)
    load('../data/Databreath.mat');
    data = [lown;
        Normaln;
        highn];
    data = data - mean(data(:));
    data = data / std(data(:));
    
    t = repmat(linspace(1, 60, (Fs*60)), 225, 1);  
    t = t + 2 * 1e-2 * data/3e8; % Add time delay due to displacement in cm
    y  = exp(1j*2*pi*f0*t);
    figure
    pspectrum(y(200, 1:Fs*n_sec), 256, 'FrequencyResolution', 1, 'spectrogram', 'FrequencyLimits', [56,64])
    title("High Breathing Rate")
else
    load('../data/micro-Doppler_collection.mat');
    data = [slow;
        normal;
        high;];
    data = data - mean(data(:));
    data = data / std(data(:));
    figure
    pspectrum(data(1, 1:Fs*n_sec)', 256, 'FrequencyResolution', 1, 'spectrogram', 'FrequencyLimits', [60,68])
    title("Low Breathing Rate")
end
