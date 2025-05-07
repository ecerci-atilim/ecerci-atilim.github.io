%% Simplified Adaptive Tau FTN Simulation & M-BCJR Receiver (Detector Debug)
%
% Simulates transmission with adaptive tau and attempts detection using
% M-BCJR with an approximate timing grid strategy at the receiver.
% Corrected tau calculation, z_time definition, and detector logic.

clear; clc; close all;

%% Simulation Parameters (as variables)
% -------------------------------------------------------------------------
N = 3;           % Message length (bits)
L = 2;           % Memory / Preamble length / Tail length (bits)
p1_start = 0;    % Peak location (sample index) of the *first* symbol (b_1) - Used by Tx ONLY
SNR_dB = 10;     % Signal-to-Noise Ratio in dB

fs = 10;         % Sampling frequency (samples per Tsym)
fd = 1;          % Symbol rate (symbols per second)
Tsym = 1/fd;     % Symbol duration

tau_opts = [0.6, 0.9]; % [tau_same_sign, tau_diff_sign]

% RRC Filter Parameters
alpha = 0.3;     % Rolloff factor
gdelay = 4;      % Group delay in symbol durations
filterType = 'sqrt';

% Simulation Oversampling (for accurate Tx signal generation)
sim_os_factor = 16;
fs_sim = fs * sim_os_factor;

%% Calculated Parameters
% -------------------------------------------------------------------------
K_total = L + N + L; % Total symbols
num_states = 2^L;    % Number of trellis states
modOrder = 2;        % BPSK

fprintf('--- Adaptive Tau FTN Simulation & M-BCJR Receiver ---\n');
fprintf('N=%d, L=%d, K_total=%d symbols\n', N, L, K_total);
fprintf('fs=%d Hz, fd=%d Hz, Tsym=%.2f s\n', fs, fd, Tsym);
fprintf('SNR = %.1f dB\n', SNR_dB);

%% 1. Transmitter Side (Ground Truth Generation)
% =========================================================================
fprintf('\n--- Transmitter Simulation ---\n');

% 1a. Bit Generation & Modulation
tx_bits = [repmat(0, 1, L), randi([0 1], 1, N), repmat(0, 1, L)]; % b_1 ... b_K
tx_symbols = 1 - 2 * tx_bits; % BPSK: 0->+1, 1->-1
fprintf('Generated %d bits.\n', K_total);
disp('True Tx Bits (b_1..K):'); disp(tx_bits);

% 1b. Calculate *True* Adaptive Tau Sequence and Peak Times
true_taus_intervals = calculate_interval_taus(tx_symbols, tau_opts); % tau_2 ... tau_K (Length K-1)
true_peak_indices = zeros(1, K_total); % p_1 ... p_K
true_peak_indices(1) = p1_start;
for k = 2:K_total
    tau_k = true_taus_intervals(k-1);
    true_peak_indices(k) = true_peak_indices(k-1) + tau_k * fs * Tsym;
end
fprintf('Calculated %d relevant tau values and %d peak locations.\n', length(true_taus_intervals), length(true_peak_indices));
disp('True Taus (tau_2..K):'); disp(true_taus_intervals);
disp('True Peak Indices (p_1..K):'); disp(true_peak_indices);

% 1c. Generate Filters
h = rcosdesign(alpha, 2 * gdelay, fs / fd, filterType);
h = h / sqrt(sum(h.^2)); % **** Normalize h to unit energy ****
h_len = length(h);
h_delay_samples = (h_len - 1) / 2;
g = conv(h, h); % g will NOT have peak 1 now, but has correct shape/energy
g_len = length(g);
g_center_idx = floor(g_len / 2) + 1;
g_time_axis = (-floor(g_len/2):floor(g_len/2));
fprintf('Generated filters h (len %d, unit energy) and g (len %d).\n', h_len, g_len);

% 1d. Generate Transmit Signal (High Oversampling)
fprintf('Generating Tx signal (oversampled)...\n');
last_peak_sim = true_peak_indices(end) * sim_os_factor;
sim_signal_len = ceil(last_peak_sim) + sim_os_factor * fs * gdelay * 2;
impulse_train_sim = zeros(1, sim_signal_len);
for k = 1:K_total
    pk_sim_idx = round(true_peak_indices(k) * sim_os_factor) + 1;
    if pk_sim_idx >= 1 && pk_sim_idx <= sim_signal_len
        impulse_train_sim(pk_sim_idx) = tx_symbols(k);
    else
        warning('Tx Peak index %d (k=%d) out of bounds %d.', pk_sim_idx, k, sim_signal_len);
    end
end
h_sim = interpft(h, sim_os_factor * h_len);
h_sim = h_sim * sqrt(sim_os_factor); % Energy adjustment for interpft
% h_sim = h_sim / sqrt(sum(h_sim.^2)); % Re-normalize upsampled (optional)
tx_signal_sim = filter(h_sim, 1, impulse_train_sim);
tx_signal = tx_signal_sim(1:sim_os_factor:end);
fprintf('Generated Tx signal (length %d at fs=%d Hz).\n', length(tx_signal), fs);

% 1e. Add Noise
signal_power = mean(abs(tx_signal).^2);
noise_power_db = 10*log10(signal_power) - SNR_dB;
noise_variance = 10^(noise_power_db / 10);
noise = sqrt(noise_variance) * randn(size(tx_signal));
noisy_tx_signal = tx_signal + noise;
fprintf('Added AWGN (Noise Variance = %.4e).\n', noise_variance);

% 1f. Matched Filter
h_rx = h(end:-1:1);
z = conv(noisy_tx_signal, h_rx);
mf_delay_samples = 2 * h_delay_samples;
z_time = (0:length(z)-1) / fs;
fprintf('Performed matched filtering (output length %d).\n', length(z));

%% 2. Receiver Side (M-BCJR Detection)
% =========================================================================
fprintf('\n--- Receiver Simulation (M-BCJR) ---\n');

% 2a. Calculate Branch Metrics (Approximate Timing)
fprintf('Calculating M-BCJR Branch Metrics (log gamma)...\n');
noise_variance_est = noise_variance; % Assume perfect estimate
avg_tau = mean(tau_opts);
nominal_samples_per_sym = avg_tau * fs * Tsym;

log_gamma = -inf(K_total, num_states, modOrder);
win_radius = floor(g_len / 2);
win_len = 2 * win_radius + 1;

for k = 1:K_total % Symbol index b_k (transition k-1 -> k)
    nominal_center_idx_z = round(mf_delay_samples + 1 + (k-1) * nominal_samples_per_sym);
    z_start_idx = max(1, nominal_center_idx_z - win_radius);
    z_end_idx = min(length(z), nominal_center_idx_z + win_radius);
    z_indices = z_start_idx:z_end_idx;
    if isempty(z_indices) || length(z_indices) < 5, continue; end
    z_samples = z(z_indices);

    if k <= L || k > L + N
        log_prior_0 = 0; log_prior_1 = -69; % log(1), log(1e-30)
    else
        log_prior_0 = -0.693; log_prior_1 = -0.693; % log(0.5)
    end
    current_log_priors = [log_prior_0, log_prior_1];

    for s_prime_idx = 1:num_states
        s_prime = s_prime_idx - 1;
        for bit_idx = 1:modOrder
            input_bit = bit_idx - 1;
            input_symbol = 1 - 2 * input_bit;

            % --- Construct Expected Signal Shape (Simplified ISI Timing) ---
            expected_shape = zeros(1, win_len);
            shape_center_idx = win_radius + 1;

            % 1. Main pulse b_k * g(t - nominal_t_k)
            g_indices_main = 1:g_len;
            shape_indices_main = g_indices_main - g_center_idx + shape_center_idx;
            valid_shape_main = (shape_indices_main >= 1) & (shape_indices_main <= win_len);
            expected_shape(shape_indices_main(valid_shape_main)) = input_symbol * g(g_indices_main(valid_shape_main));

            % 2. ISI from previous L symbols based on s'
            temp_state = s_prime;
            current_delay_samples = 0;
            % Calculate local taus based ONLY on hypothesis (s', input_bit)
            taus_local = zeros(1, L);
            b_k_hyp = input_bit;
            b_km1_hyp = bitand(bitshift(s_prime, -(L-1)), 1);
            if b_k_hyp == b_km1_hyp, taus_local(1) = tau_opts(1); else, taus_local(1) = tau_opts(2); end
            temp_s_prime_for_tau = s_prime;
            for i_tau = 1:L-1
                b1 = bitand(bitshift(temp_s_prime_for_tau, -(L-1)), 1);
                b2 = bitand(bitshift(temp_s_prime_for_tau, -(L-2)), 1);
                if b1 == b2, taus_local(i_tau+1) = tau_opts(1); else, taus_local(i_tau+1) = tau_opts(2); end
                temp_s_prime_for_tau = bitshift(temp_s_prime_for_tau, 1);
            end

            for i = 1:L % ISI symbol b_{k-i}
                prev_bit = bitand(temp_state, 1);
                prev_symbol_isi = 1 - 2 * prev_bit;
                temp_state = bitshift(temp_state, -1);

                % Sum relevant LOCAL taus: tau_k + ... + tau_{k-i+1}
                current_delay_samples = sum(taus_local(1:i)) * fs * Tsym;
                delay_offset = round(current_delay_samples);

                g_indices_isi = 1:g_len;
                shape_indices_isi = g_indices_isi - g_center_idx + shape_center_idx - delay_offset;
                valid_shape_isi = (shape_indices_isi >= 1) & (shape_indices_isi <= win_len);
                valid_g_isi = valid_shape_isi; % Indices in g match shape indices here

                indices_to_update = shape_indices_isi(valid_shape_isi);
                g_indices_to_use = g_indices_isi(valid_g_isi);

                if ~isempty(indices_to_update)
                    expected_shape(indices_to_update) = expected_shape(indices_to_update) ...
                                                        + prev_symbol_isi * g(g_indices_to_use);
                end
            end
            % --- End Construct Expected Signal ---

            shape_start_idx = max(1, shape_center_idx - (nominal_center_idx_z - z_start_idx));
            shape_end_idx = min(win_len, shape_center_idx + (z_end_idx - nominal_center_idx_z));
            shape_indices_extract = shape_start_idx:shape_end_idx;

            if length(shape_indices_extract) ~= length(z_samples)
                 % Pad shorter array with zeros to match length
                 len_diff = length(z_samples) - length(shape_indices_extract);
                 if len_diff > 0 % z_samples is longer
                     expected_shape_samples = [expected_shape(shape_indices_extract), zeros(1, len_diff)];
                 elseif len_diff < 0 % expected_shape is longer
                     expected_shape_samples = expected_shape(shape_indices_extract(1:length(z_samples)));
                 else
                      expected_shape_samples = expected_shape(shape_indices_extract);
                 end
                 warning('Metric Calc: Adjusted length mismatch k=%d, s''=%d, b=%d. z=%d, shape=%d', k, s_prime, input_bit, length(z_samples), length(shape_indices_extract));
            else
                expected_shape_samples = expected_shape(shape_indices_extract);
            end

            % Calculate metric only if lengths match after potential adjustment
            if length(expected_shape_samples) == length(z_samples)
                distance_sq = sum(abs(z_samples - expected_shape_samples).^2);
                % Check for NaN/Inf distance (can happen with empty samples)
                if isnan(distance_sq) || isinf(distance_sq)
                    log_likelihood = -inf;
                else
                    log_likelihood = -distance_sq / (2 * noise_variance_est);
                end
                metric = log_likelihood + current_log_priors(bit_idx);
            else
                 metric = -inf; % Should not happen with padding, but safety check
            end

            % Check for NaN/Inf metric before assignment
            if isnan(metric) || isinf(metric)
                 log_gamma(k, s_prime_idx, bit_idx) = -1e30; % Assign large negative instead of Inf
                 if isnan(metric)
                      warning('NaN metric detected k=%d, s''=%d, b=%d', k, s_prime, input_bit);
                 end
            else
                 log_gamma(k, s_prime_idx, bit_idx) = metric;
            end

        end % End loop input_bit
    end % End loop s_prime
end % End loop k
fprintf('Branch metric calculation complete.\n');

% 2b. Run M-BCJR Algorithm
fprintf('Running M-BCJR Algorithm...\n');
log_alpha = -inf(K_total + 1, num_states);
log_beta = -inf(K_total + 1, num_states);
log_alpha(1, 1) = 0;
log_beta(K_total + 1, 1) = 0;

% Forward Recursion (alpha)
for k = 1:K_total
    for s_idx = 1:num_states
        s = s_idx - 1;
        max_val = -inf;
        for s_prime_idx = 1:num_states
            s_prime = s_prime_idx - 1;
            for bit_idx = 1:modOrder
                input_bit = bit_idx - 1;
                expected_next_s = bitshift(s_prime, -1) + bitshift(input_bit, L - 1);
                if expected_next_s == s
                    gamma_val = log_gamma(k, s_prime_idx, bit_idx);
                    alpha_prev = log_alpha(k, s_prime_idx);
                    % Check for -inf + -inf case
                    if ~isinf(alpha_prev) && ~isinf(gamma_val)
                        current_val = alpha_prev + gamma_val;
                        if current_val > max_val
                            max_val = current_val;
                        end
                    end
                end
            end
        end
        log_alpha(k + 1, s_idx) = max_val;
    end
    % Optional normalization: Can help if values drift too low
    % max_alpha_k = max(log_alpha(k+1, :));
    % if ~isinf(max_alpha_k)
    %     log_alpha(k+1, :) = log_alpha(k+1, :) - max_alpha_k;
    % end
end

% Backward Recursion (beta)
 for k = K_total:-1:1
    for s_prime_idx = 1:num_states % State s' at time k-1 (index)
        s_prime = s_prime_idx - 1;
        max_val = -inf;
         for bit_idx = 1:modOrder % Input bit b_k
             input_bit = bit_idx - 1;
             s = bitshift(s_prime, -1) + bitshift(input_bit, L - 1); % Next state s at k
             s_idx = s + 1;
             if s_idx >= 1 && s_idx <= num_states
                 gamma_val = log_gamma(k, s_prime_idx, bit_idx);
                 beta_next = log_beta(k + 1, s_idx);
                 % Check for -inf + -inf case
                 if ~isinf(beta_next) && ~isinf(gamma_val)
                     current_val = beta_next + gamma_val;
                     if current_val > max_val
                         max_val = current_val;
                     end
                 end
             end
         end
         log_beta(k, s_prime_idx) = max_val; % Store beta_{k-1} at index k
    end
     % Optional normalization:
     % max_beta_k = max(log_beta(k, :));
     % if ~isinf(max_beta_k)
     %     log_beta(k, :) = log_beta(k, :) - max_beta_k;
     % end
 end

% LLR Calculation
llr = zeros(1, K_total);
for k = 1:K_total
    lambda_1 = -inf;
    lambda_0 = -inf;
    for s_prime_idx = 1:num_states
        s_prime = s_prime_idx - 1;
        for bit_idx = 1:modOrder
            input_bit = bit_idx - 1;
            s = bitshift(s_prime, -1) + bitshift(input_bit, L - 1);
            s_idx = s + 1;
            if s_idx >= 1 && s_idx <= num_states
                alpha_val = log_alpha(k, s_prime_idx);
                gamma_val = log_gamma(k, s_prime_idx, bit_idx);
                beta_val = log_beta(k + 1, s_idx);
                % Check for -inf contributions
                if ~isinf(alpha_val) && ~isinf(gamma_val) && ~isinf(beta_val)
                    metric = alpha_val + gamma_val + beta_val;
                    if input_bit == 1
                        if metric > lambda_1, lambda_1 = metric; end
                    else
                        if metric > lambda_0, lambda_0 = metric; end
                    end
                end
            end
        end
    end
    % Handle cases where one lambda might be -inf
    if isinf(lambda_1) && isinf(lambda_0)
        llr(k) = 0; % Undetermined, assign 0 LLR
    elseif isinf(lambda_1)
        llr(k) = -100; % Strongly favor 0 if no path for 1 found
    elseif isinf(lambda_0)
        llr(k) = 100;  % Strongly favor 1 if no path for 0 found
    else
        llr(k) = lambda_1 - lambda_0;
    end
end
fprintf('M-BCJR calculation complete.\n');

% 2c. Sequence Estimation
detected_bits = double(llr > 0);

%% 3. Results
% =========================================================================
fprintf('\n--- Results ---\n');
fprintf('True Tx Bits:      '); fprintf('%d ', tx_bits); fprintf('\n');
fprintf('Detected Bits:     '); fprintf('%d ', detected_bits); fprintf('\n');
fprintf('LLR Values:        '); fprintf('%.2f ', llr); fprintf('\n');

message_indices = (L+1):(L+N);
bit_errors = sum(tx_bits(message_indices) ~= detected_bits(message_indices));
if N > 0
    ber = bit_errors / N;
    fprintf('\nBit Errors (Message only): %d\n', bit_errors);
    fprintf('Message BER: %.4f\n', ber);
else
    fprintf('\nNo message bits to calculate BER.\n');
end

% Plot received signal with NOMINAL peak locations AND corresponding z values
figure;
plot(z_time, z, 'DisplayName', 'Received Signal z(t)');
hold on;
nominal_peak_indices_rx = round(mf_delay_samples + 1 + (0:K_total-1) * nominal_samples_per_sym);
valid_indices = nominal_peak_indices_rx >= 1 & nominal_peak_indices_rx <= length(z);
nominal_peak_indices_rx_valid = nominal_peak_indices_rx(valid_indices);
time_at_nominal_peaks = (nominal_peak_indices_rx_valid - 1) / fs; % Correct time conversion
z_values_at_nominal_peaks = z(nominal_peak_indices_rx_valid);
plot(time_at_nominal_peaks, z_values_at_nominal_peaks, ...
     'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 7, ...
     'DisplayName', 'z(t) at Nominal Peaks (Rx)');
plot(true_peak_indices / fs, zeros(size(true_peak_indices)), ...
     'rv', 'MarkerFaceColor', 'r', 'MarkerSize', 5, ...
     'DisplayName', 'True Peaks (Tx)');
hold off;
xlim_max_z = min(max(true_peak_indices/fs)+gdelay*4*Tsym, z_time(end));
xlim([0 xlim_max_z]);
title('Received Signal z(t) with Actual Values at Nominal Peak Times');
xlabel('Time (s)'); ylabel('Amplitude'); grid on; legend('show');

% Plot pulse shape g
figure;
plot(g_time_axis, g);
title('Overall Pulse Shape g = conv(h,h)');
xlabel('Sample Index (Relative to Peak)');
ylabel('Amplitude (Unnormalized)'); % Changed label as g peak is not 1 now
grid on;

fprintf('\n--- Simplified Simulation Finished ---\n');


%% Helper Function: Calculate Interval Taus (tau_2 ... tau_K)
% =========================================================================
function taus_intervals = calculate_interval_taus(symbols, tau_opts)
    K = length(symbols);
    if K < 2, taus_intervals = []; return; end
    taus_intervals = zeros(1, K-1);
    for k = 2:K
        current_symbol = symbols(k);
        prev_symbol = symbols(k-1);
        if sign(current_symbol) == sign(prev_symbol)
            taus_intervals(k-1) = tau_opts(1);
        else
            taus_intervals(k-1) = tau_opts(2);
        end
    end
end % End calculate_interval_taus