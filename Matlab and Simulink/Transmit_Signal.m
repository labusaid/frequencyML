% Script to transmit a signal using Simulink modulation models.

clc
clearvars

modulation_model = 'BPSK_Modulation';    % Name of the modulation model
f_carrier = 800e6;      % The carrier frequency (Hz). Accepted values: 7e+07 - 6e+09.
f_sample = 1e5;         % The input sampling frequency
simulation_time = 1e-4; % The duration of the simulation
seed = 0;           % The seed used for the random number generator input
gain_dB = -20;      % The transmitter gain
number_transmit = 10;   % The number of times the signal should be transmitted

% Signal is transmitted number_transmit times by running the simulation
% multiple times
for i = 1:number_transmit
    sim(modulation_model, simulation_time);
    
    % Wait for current simulation to stop until next one is started
    while(get_param(modulation_model, 'SimulationStatus') ~= 'stopped')
    end
end