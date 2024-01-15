%-----------------------

%Toobox list
%Signal processing toolbox
%Control system toolbox

%Extra file list
%HigFracDim.m (Provided by Guideline)
%framinmg.m (for auto segmentation)

% NOTE: THIS LIST MAY BE INCOMPLETE, LET ME gKNOW IN WHATSAPP IF YOU'RE MISSING SOME FILES!

%============================ USER TUTORIAL ===============================
%   This program is made for processing
%       EMGforce1.txt.txt
%       EMGforce2.txt.txt
% 
% 
% 
%   This program CANNOT process the following
%       EM_EMG_SQUEEZE1.txt
%       EM_EMG_SQUEEZE2.txt
%
%
%
%   Reason: Projauto.m cannot identify segments from these 2 data (for some reason)
%
%   Step 1: Put HigFracDim.m and framinmg.m in the same directory as this code file so that matlab can read it
%   Step 2: Choose the data you want to process
%   Step 3: enter the file name in target= load(  ); (in line 56)
%                                               ^here
%
%   Step 4:Adjust the low pass filter based on your need for the report (in line 109 and 110)
%
%   !Remember to record the low pass filter adjustments (so you can revert it)!
%
%                               Outputs  List
%
%      Graphs(Original+polyfit)              Corresponding correlation coeficient 
%          RMS value                        RRRMS
%          Mean Freq.                       RRMeanFrequency
%        Median Freq.                       RRMedainFrequency
%       Fractal dimension                   RRFD
%============================END OF TUTORIAL===============================

%=======================BEGINNING OF CODE==================================
%-----------------------
clear, clc, close all  %!Clears all data each time you run the code!
%Task 1
%Import data
importdata EMGforce1.txt;
importdata EMGforce2.txt;
importdata EM_EMG_SQUEEZE1.txt;
importdata EM_EMG_SQUEEZE2.txt;

target= load("EMGforce1.txt"); %This time we will analyze this .txt file
fs = 2000; % Sample rate 2000 Hz
% NOTE: The sampling rate is 2000 Hz per channel and the EMG sample values are in mV.

time= target(:,1); % The first column, TIME (sample frequency 2000 Hz)
force= target(:,2); % The second column, FORCE (in arbitrary units)
emgmv= target(:,3); % The third column, EMG SIGNAL (mV)

%-----------------------
%Task 2
%Force signal normalization

% Calculate the minimum and maximum values
min_force = min(force);
max_force = max(force);

% Normalize the data
normalized_force = 100*((force - min_force) / (max_force - min_force));

%plot graph
% Force vs Time
figure(1)
subplot(3,1,1)
plot(time,normalized_force)
xlabel('Time (s)')
ylabel('normalized force (in %MVC)')
title('Input force')
% EMG vs Time
figure(1)
subplot(3,1,2)
plot(time,emgmv)
xlabel('Time (s)')
ylabel('EMG signal (in mV)')
title('Input EMG')

%-----------------------
%Task 3
%Low pass filter (From my assignment code for low pass filter)

% frequency spectrum
emg_fft = fft(emgmv); % FT of emg. ecg_fft is a complex vector.
N = length(emgmv);    % number of samples, emg_fft is also this long.
f = (0:N-1)/fs; % frequency in Hz (corrsponding to real time frequency), complete your code here

figure(2)
subplot(3,1,1)
semilogx(f(1:N/2),20*log10(abs(emg_fft(1:N/2))))
xlabel('Frequency (Hz)')
title('Frequency Spectrum of EMG')


%Low pass filter option

fc = 0.03                                        %Change the cutoff frequency
[b,a] = butter(1,2*pi*fc,'s') ;                 %Change the order of the filter, e.g, 1, 2, or 5 ! Don't change 2*pi*fc and 's'!

lowpass = tf(b,a);  % TF of the filter in cont' time
[mag_lp, phase_lp] = bode(lowpass, 2*pi*f); % mag is in absolute, phase is in deg.
% the size of mag_lp is 1x1xm; need to convert this to 1xm
mag_lp = squeeze(mag_lp);
phase_lp = squeeze(phase_lp);
figure(2)   % frequency response of the filter.
subplot(3,1,2)
semilogx(f(1:N/2),20*log10(mag_lp(1:N/2)))
title('Frequency Response of the Filter')
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')



%Expected spectrum NOTE:(Not needed here, but kept it here just in case we do)
% figure(2)
% subplot(3,1,3)
% magnitude_db = 20 * log10(mag_lp);
% plot(magnitude_db)
% title('Expected spectrum')
% xlabel('Frequency (Hz)')
% ylabel('Magnitude (dB)')

lowpass_ct = lowpass;
H_lp_dt = c2d(lowpass_ct,1/fs);

%Inverse Fourier Transform
emg_fft_filtered = emg_fft.*squeeze(freqresp(H_lp_dt,2*pi*f));
emg_filtered = ifft(emg_fft_filtered);

figure(1)
subplot(3,1,3)
plot(time,emg_filtered)
xlabel('Time (s)')
ylabel('EMG Signal (MV)')
title('Filtered EMG')

%Task 4 --------------------------
%TEST FROM SOURCE CODE FROM GITHUB

% ---------------------------------------------------------- Part 1


%Auto segmenatation

%-----------Identify Segments--------------------------------------------
x = force;                                    % get the first channel
%x = x/max(abs(x));                              % normalize the signal
N = length(x);                                  % signal length
fss=2000;
t = (0:N-1)/fss;                                 % time vector
%% signal framing
frlen = round(fss);                        % frame length
hop = round(frlen/10);                           % hop size
[FRM, tfrm] = framing(emgmv, frlen, hop, fss);       % signal framing (Here emgmv is used because the code can identify segments without filtering)

%% determine the Short-time energy
STE = sum(FRM.^2);

%% determine the Short-time zero-crossing rate
STZCR = sum(abs(diff(FRM > 0)))/size(FRM, 1);

%% plot the results
% plot the signal waveform
figure(5)
subplot(3, 1, 1)
plot(t, x)
grid on
xlim([0 max(time)])
ylim([-1.1*max(abs(x)) 1.1*max(abs(x))])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Amplitude')
title('The signal in the time domain')

% % plot the STE
% subplot(3, 1, 2)
% plot(tfrm, STE)
% grid on
% xlim([0 max(tfrm)])
% ylim([0 1.1*max(STE)])
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% xlabel('Time, s')
% ylabel('Energy')
% title('Short-time Energy')
% 
% % plot the STZCR
% subplot(3, 1, 3)
% plot(tfrm, STZCR)
% grid on
% xlim([0 max(tfrm)])
% ylim([0 1.1*max(STZCR)])
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% xlabel('Time, s')
% ylabel('ZCR')
% title('Short-time ZCR')

%% mark the signal
% Note: (i) the activity flag is rised when there is high energy frame
% in the signal stream; (ii) the consonant flag is rised when there are
% high ZCR activity and high energy in the signal stream.
AF = STE > 0.5*median(STE);
CF = AF & STZCR>2*median(STZCR);

subplot(3, 1, 1)
hold on
plot(tfrm, AF, 'r', 'LineWidth', 1.5)
plot(tfrm, CF, 'g', 'LineWidth', 1.5)
legend('signal', 'activity flag', 'consonant flag', ...
       'Location', 'southeast')
%--------------Segment location computation--------------------------------
s = AF;
% signal = s(:,2);
tt =length(s)
% timesdifference = 1000;
% slopetime = 3000;
% thershold = (0.4);
% figure(1)
segmentpoint=[];

loop=length(tt);
for i = (1:1:tt-1)
    if abs(s(i+1)-s(i)) == 1
        segmentpoint=[segmentpoint i];
    end
end

IdentSegTime=segmentpoint*(length(emg_fft)/length(AF))/2000;
%               Output of IdentSegTime
% Start time and end time of each segments, which is used to extract the
% code for further calculation 

% END of AUTO CODE----------------------------------


% Plot of the EMG (mV) and normalized force (in percent % MVC) vs  Time
% (Sec) with segments marked
EMG_VoltageForceVsTimeSeg=figure('Name','EMG signal and Normalized force vs. Time'); % Create a new figure
subplot(2,1,1); plot(time, normalized_force)
hold on;
plot(IdentSegTime,0,'r*');  % Highlight 
hold off;
ylabel('Normalized Force (%MVC)');
xlabel('Time (Sec)');
subplot(2,1,2); plot(time, emgmv)
hold on;
plot(IdentSegTime,0,'r*');  % Highlight 
hold off;
ylabel('EMG Signal (mV)');
xlabel('Time (Sec)');
title('EMG signal and Normalized force vs. Time');
axis auto; 

% ---------------------------------------------------------- Part 3
% For each segment of the EMG identified compute the DR, MS , RMS and ZCR
% parameters.

SampleInterval = 0.0005; % The EMG signals are sampled with a period of 0.0005s (Maybe we can change the value later?)
CounterEnd= length(IdentSegTime)/2

for Counter=1:1:CounterEnd % no. of segemnts in this signal, may change for other data
    Segment = Counter * 2 - 1;
    disp(['Segment = ', num2str(Segment)]); % Display in console
    disp(['Start = ', num2str(IdentSegTime(Segment))]); % Display in console
    disp(['Stop = ', num2str(IdentSegTime(Segment + 1))]); % Display in console

    % Identify the segments of interest from the EMG signals
    s1 = IdentSegTime(Segment) / SampleInterval; % First point index
    s2 = IdentSegTime(Segment + 1) / SampleInterval; % Second point index
    EMG_ForceSeg = normalized_force(s1:s2); % Segment of interest of force signal
    EMG_VoltageSeg = emgmv(s1:s2); % Segment of interest of voltage signal  

    % Calculate Mean Squared Value
    NumSamplesInSeg = s2 - s1 + 1; % The number of samples in the segment
    MeanSquared(Counter) = sum(EMG_VoltageSeg.^2) / NumSamplesInSeg;

    %-----------------------Parameters we need------------------------
    % Calculate RMS
    RMS(Counter) = sqrt(MeanSquared(Counter));

    % Calculate mean frequency
    meanfrequency(Counter)= meanfreq(EMG_VoltageSeg,fs);

    % Calculate median frequency
    medianfrequency(Counter)= medfreq(EMG_VoltageSeg,fs);

    % Calculate fractal dimension
    FractDim(Counter)= HigFracDim(EMG_VoltageSeg, 8);

    %-----------------------End of Parameters we need------------------------

    % Calculate mean voltage value for each segment
    ForceMean(Counter) = mean(EMG_ForceSeg)

    %subplot(14,1,Segment); plot(EMG_ForceSeg); %uncomment plots force segments        
end

% ---------------------------------------------------------- Part 4
% Plot the DR, MS, RMS, and ZCR values versus force in %MVC.  Lable the
% axis with appropriate units.

DRMSRMSZCRvsMVC=figure('Name','EMG parameters vs. force'); % Create a new figure

%-----------------------Plots we need------------------------
subplot(4,1,1); plot(ForceMean,meanfrequency)
title('Mean frequency vs. Normalized Force');
ylabel('Mean frequency');
xlabel('Normalized Force (%MVC)');
subplot(4,1,2); plot(ForceMean,medianfrequency)
title('Median frequency vs. Normalized Force');
ylabel('Median frequency');
xlabel('Normalized Force (%MVC)');
subplot(4,1,3); plot(ForceMean, RMS)
title('Root Mean Squared (average magnitude) vs. Normalized Force');
ylabel('RMS (mV)');
xlabel('Normalized Force (%MVC)');
subplot(4,1,4); plot(ForceMean,FractDim)
title('Fractal dimension vs. Normalized Force');
ylabel('Dimensionality');
xlabel('Normalized Force (%MVC)');
%-----------------------End of Plots we need------------------------



% ---------------------------------------------------------- Part 5
% Using the polyfit function obtain a straight line fit to represent teh
% variation of each EMG parametere vs. Force.  Use polyval to evaluate the
% values of the dependant variable giving by the models and plot them vs.
% the orignal values

Counter = (1:1:CounterEnd); % segment index variable for linear parameters
% Remember that ForceMean is a size (number of segments) vector of mean force values per seg


%-----------------------Linear plots we need------------------------
% Calculate a linear model for Mean frequency and add it to the plot
PMF= polyfit(ForceMean,meanfrequency,1);
subplot(4,1,1); hold on; plot(ForceMean, polyval(PMF,ForceMean),'r--'); hold off;
title('Mean Frequency (blue) & Linear Model (red) vs. Normalized Force');

% Calculate a linear model for Mean frequency and add it to the plot
PMEF= polyfit(ForceMean,medianfrequency,1);
subplot(4,1,2); hold on; plot(ForceMean, polyval(PMEF,ForceMean),'r--'); hold off;
title('Median Frequency (blue) & Linear Model (red) vs. Normalized Force');

% Calculate a linear model for Root mean squared and add it to the plot
PRMS = polyfit(ForceMean, RMS, 1);
subplot(4,1,3); hold on; plot(ForceMean, polyval(PRMS,ForceMean),'r--'); hold off;
title('Root Mean Squared (average magnitude) (blue) & Linear Model (red) vs. Normalized Force');

% Calcualte a linear model for Fractal dimension
PFD=polyfit(ForceMean,FractDim,1);
subplot(4,1,4); hold on; plot(ForceMean, polyval(PFD,ForceMean),'r--'); hold off;
title('Fractal dimension (blue) & Linear Model (red) vs. Normalized Force');
%-----------------------End of Linear plots we need------------------------



% ---------------------------------------------------------- Part 6
% Computer the correlation coefficient r^2 and analyze the goodness of fit
% for each parameter

% Calculate the correlation coefficient for Mean Frequency
x = ForceMean; % calculated parameter
%y = POLYVAL(PDynamicRange, ForceMean) % Linear Model
y = meanfrequency;
N = CounterEnd; % Index size

%Verify formula to calculate the top works

Top = ( sum(x .* y) - N * mean(x) * mean(y) ).^2;

%Verify formula to calculate the bottomL works

BottomL = sum(x.^2) - N * ((mean(x)).^2);
BottomR = sum(y.^2) - N * ((mean(y)).^2);
RRMeanFrequency = Top / (BottomL * BottomR)

% Calculate the correlation coefficient for MS
x = ForceMean; % calculated parameter
%y = POLYVAL(PMeanSquared, ForceMean); % Linear Model
y = medianfrequency;
N = CounterEnd; % Index size
Top = ( sum(x .* y) - N * mean(x) * mean(y) ).^2;
BottomL = sum(x.^2) - N * (mean(x).^2);
BottomR = sum(y.^2) - N * (mean(y).^2);
RRMedianFrequency = Top / (BottomL * BottomR)


% Calculate the correlation coefficient for RMS
x = ForceMean; % calculated parameter
%y = POLYVAL(PRMS, ForceMean) % Linear Model
y = RMS;
N = CounterEnd; % Index size
Top = ( sum(x .* y) - N * mean(x) * mean(y) ).^2;
BottomL = sum(x.^2) - N * (mean(x).^2);
BottomR = sum(y.^2) - N * (mean(y).^2);
RRRMS = Top / (BottomL * BottomR)

% Calculate the correlation coefficient for Fractal Dimension
x = ForceMean; % calculated parameter
%y = POLYVAL(PRMS, ForceMean) % Linear Model
y = FractDim;
N = CounterEnd; % Index size
Top = ( sum(x .* y) - N * mean(x) * mean(y) ).^2;
BottomL = sum(x.^2) - N * (mean(x).^2);
BottomR = sum(y.^2) - N * (mean(y).^2);
RRFD = Top / (BottomL * BottomR)


%End of test-----------------------------


