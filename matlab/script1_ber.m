%% Monte Carlo Simulation of Adaptive Tau FTN System & M-BCJR Receiver
%
% Simulates multiple frames to estimate BER using M-BCJR with an
% approximate timing grid strategy at the receiver.

clear; clc; close all;

%% Simulation Parameters
% -------------------------------------------------------------------------
N = 100;         % Message length (bits) - Larger N for meaningful BER
L = 2;           % Memory / Preamble length / Tail length (bits)
SNR_dB = 8;      % Signal-to-Noise Ratio in dB to test

% Monte Carlo Parameters
max_loops = 1000;   % Maximum number of frames (loops) to simulate
max_frame_errors = 100; % Stop simulation after this many frames have errors
min_bit_errors = 100;   % Minimum number of bit errors to collect for reliable BER

% System Parameters (as variables)
p1_start = 0;    % Peak location (sample index) of the *first* symbol (b_1) - Used by Tx ONLY
fs = 10;         % Sampling frequency (samples per Tsym)
fd = 1;          % Symbol rate (symbols per second)
Tsym = 1/fd;     % Symbol duration
tau_opts = [0.6, 0.9]; % [tau_same_sign, tau_diff_sign]
alpha = 0.3;     % Rolloff factor
gdelay = 4;      % Group delay in symbol durations
filterType = 'sqrt';
sim_os_factor = 16; % Tx oversampling
fs_sim = fs * sim_os_factor;

%% Calculated Parameters & Initialization
% -------------------------------------------------------------------------
K_total = L + N + L; % Total symbols per frame
num_states = 2^L;
modOrder = 2;

% Initialize counters
total_bits_simulated = 0;
total_bit_errors = 0;
total_frame_errors = 0;
loop_count = 0;

% Generate Filters (once)
h = rcosdesign(alpha, 2 * gdelay, fs / fd, filterType);
h = h / sqrt(sum(h.^2)); % Normalize h to unit energy
h_len = length(h);
h_delay_samples = (h_len - 1) / 2;
g = conv(h, h); % Overall pulse shape (for receiver)
g_len = length(g);
g_center_idx = floor(g_len / 2) + 1;
mf_delay_samples = 2 * h_delay_samples;

% Calculate noise variance (once)
% Need an estimate of signal power. We can estimate based on pulse shape
% energy or run one Tx cycle first. Let's estimate based on h energy (should be 1).
% Power after MF for unit energy pulse is approx energy of g.
% A rough estimate: Assume average symbol power is 1 after MF.
% This is an approximation, true power depends on ISI.
estimated_signal_power = 1; % Approximation for noise calculation
noise_power_db = 10*log10(estimated_signal_power) - SNR_dB;
noise_variance = 10^(noise_power_db / 10);
noise_variance_est = noise_variance; % Receiver's estimate

fprintf('--- Monte Carlo Simulation: Adaptive Tau FTN & M-BCJR ---\n');
fprintf('N=%d, L=%d, K_total=%d symbols/frame\n', N, L, K_total);
fprintf('Target SNR = %.1f dB (Noise Variance ~ %.4e)\n', SNR_dB, noise_variance);
fprintf('Max Loops: %d, Max Frame Errors: %d, Min Bit Errors: %d\n', max_loops, max_frame_errors, min_bit_errors);
fprintf('---------------------------------------------------------\n');

%% Monte Carlo Loop
% -------------------------------------------------------------------------
tic; % Start timer

while (loop_count < max_loops && total_frame_errors < max_frame_errors)

    loop_count = loop_count + 1;

    % --- Transmitter Side ---
    % Generate bits for this frame
    tx_bits = [zeros(1, L), randi([0 1], 1, N), zeros(1, L)];
    tx_symbols = 1 - 2 * tx_bits;

    % Calculate true taus and peaks for this frame
    true_taus_intervals = calculate_interval_taus(tx_symbols, tau_opts);
    true_peak_indices = zeros(1, K_total);
    true_peak_indices(1) = p1_start;
    for k = 2:K_total
        tau_k = true_taus_intervals(k-1);
        true_peak_indices(k) = true_peak_indices(k-1) + tau_k * fs * Tsym;
    end

    % Generate Tx signal (oversampled)
    last_peak_sim = true_peak_indices(end) * sim_os_factor;
    sim_signal_len = ceil(last_peak_sim) + sim_os_factor * fs * gdelay * 2; % Dynamic length
    impulse_train_sim = zeros(1, sim_signal_len);
    for k = 1:K_total
        pk_sim_idx = round(true_peak_indices(k) * sim_os_factor) + 1;
        if pk_sim_idx >= 1 && pk_sim_idx <= sim_signal_len
            impulse_train_sim(pk_sim_idx) = tx_symbols(k);
        end % No warning in loop for speed
    end
    h_sim = interpft(h, sim_os_factor * h_len);
    h_sim = h_sim * sqrt(sim_os_factor);
    tx_signal_sim = filter(h_sim, 1, impulse_train_sim);
    tx_signal = tx_signal_sim(1:sim_os_factor:end);

    % Add Noise
    noise = sqrt(noise_variance) * randn(size(tx_signal));
    noisy_tx_signal = tx_signal + noise;

    % Matched Filter
    h_rx = h(end:-1:1);
    z = conv(noisy_tx_signal, h_rx);

    % --- Receiver Side ---
    % Calculate Branch Metrics
    avg_tau = mean(tau_opts);
    nominal_samples_per_sym = avg_tau * fs * Tsym;
    log_gamma = -inf(K_total, num_states, modOrder);
    win_radius = floor(g_len / 2);
    win_len = 2 * win_radius + 1;

    for k = 1:K_total
        nominal_center_idx_z = round(mf_delay_samples + 1 + (k-1) * nominal_samples_per_sym);
        z_start_idx = max(1, nominal_center_idx_z - win_radius);
        z_end_idx = min(length(z), nominal_center_idx_z + win_radius);
        z_indices = z_start_idx:z_end_idx;
        if isempty(z_indices) || length(z_indices) < 5, continue; end
        z_samples = z(z_indices);

        if k <= L || k > L + N, current_log_priors = [0, -69]; else, current_log_priors = [-0.693, -0.693]; end

        for s_prime_idx = 1:num_states
            s_prime = s_prime_idx - 1;
            for bit_idx = 1:modOrder
                input_bit = bit_idx - 1;
                input_symbol = 1 - 2 * input_bit;
                expected_shape = zeros(1, win_len);
                shape_center_idx = win_radius + 1;
                % Main pulse
                g_indices_main = 1:g_len; shape_indices_main = g_indices_main - g_center_idx + shape_center_idx; valid_shape_main = (shape_indices_main >= 1) & (shape_indices_main <= win_len); valid_g_idx = valid_shape_main; expected_shape(shape_indices_main(valid_shape_main)) = input_symbol * g(g_indices_main(valid_g_idx));
                % ISI
                temp_state = s_prime; current_delay_samples = 0; taus_local = zeros(1, L); b_k_hyp = input_bit; b_km1_hyp = bitand(bitshift(s_prime, -(L-1)), 1); if b_k_hyp == b_km1_hyp, taus_local(1) = tau_opts(1); else, taus_local(1) = tau_opts(2); end; temp_s_prime_for_tau = s_prime; for i_tau = 1:L-1, b1 = bitand(bitshift(temp_s_prime_for_tau, -(L-1)), 1); b2 = bitand(bitshift(temp_s_prime_for_tau, -(L-2)), 1); if b1 == b2, taus_local(i_tau+1) = tau_opts(1); else, taus_local(i_tau+1) = tau_opts(2); end; temp_s_prime_for_tau = bitshift(temp_s_prime_for_tau, 1); end
                for i = 1:L, prev_bit = bitand(temp_state, 1); prev_symbol_isi = 1 - 2 * prev_bit; temp_state = bitshift(temp_state, -1); current_delay_samples = sum(taus_local(1:i)) * fs * Tsym; delay_offset = round(current_delay_samples); g_indices_isi = 1:g_len; shape_indices_isi = g_indices_isi - g_center_idx + shape_center_idx - delay_offset; valid_shape_isi = (shape_indices_isi >= 1) & (shape_indices_isi <= win_len); valid_g_isi = valid_shape_isi; indices_to_update = shape_indices_isi(valid_shape_isi); g_indices_to_use = g_indices_isi(valid_g_isi); if ~isempty(indices_to_update), expected_shape(indices_to_update) = expected_shape(indices_to_update) + prev_symbol_isi * g(g_indices_to_use); end; end

                shape_start_idx = max(1, shape_center_idx - (nominal_center_idx_z - z_start_idx)); shape_end_idx = min(win_len, shape_center_idx + (z_end_idx - nominal_center_idx_z)); shape_indices_extract = shape_start_idx:shape_end_idx;
                if length(shape_indices_extract) ~= length(z_samples), len_diff = length(z_samples) - length(shape_indices_extract); if len_diff > 0, expected_shape_samples = [expected_shape(shape_indices_extract), zeros(1, len_diff)]; elseif len_diff < 0, expected_shape_samples = expected_shape(shape_indices_extract(1:length(z_samples))); else, expected_shape_samples = expected_shape(shape_indices_extract); end; else, expected_shape_samples = expected_shape(shape_indices_extract); end
                if length(expected_shape_samples) == length(z_samples), distance_sq = sum(abs(z_samples - expected_shape_samples).^2); if isnan(distance_sq) || isinf(distance_sq), log_likelihood = -inf; else, log_likelihood = -distance_sq / (2 * noise_variance_est); end; metric = log_likelihood + current_log_priors(bit_idx); else, metric = -inf; end
                if isnan(metric) || isinf(metric), log_gamma(k, s_prime_idx, bit_idx) = -1e30; else, log_gamma(k, s_prime_idx, bit_idx) = metric; end
            end % End bit_idx
        end % End s_prime_idx
    end % End k (gamma calc)

    % Run M-BCJR Algorithm
    log_alpha = -inf(K_total + 1, num_states); log_beta = -inf(K_total + 1, num_states); log_alpha(1, 1) = 0; log_beta(K_total + 1, 1) = 0;
    % Forward
    for k = 1:K_total, for s_idx = 1:num_states, s = s_idx - 1; max_val = -inf; for s_prime_idx = 1:num_states, s_prime = s_prime_idx - 1; for bit_idx = 1:modOrder, input_bit = bit_idx - 1; expected_next_s = bitshift(s_prime, -1) + bitshift(input_bit, L - 1); if expected_next_s == s, gamma_val = log_gamma(k, s_prime_idx, bit_idx); alpha_prev = log_alpha(k, s_prime_idx); if ~isinf(alpha_prev) && ~isinf(gamma_val), current_val = alpha_prev + gamma_val; if current_val > max_val, max_val = current_val; end; end; end; end; end; log_alpha(k + 1, s_idx) = max_val; end; end
    % Backward
    for k = K_total:-1:1, for s_prime_idx = 1:num_states, s_prime = s_prime_idx - 1; max_val = -inf; for bit_idx = 1:modOrder, input_bit = bit_idx - 1; s = bitshift(s_prime, -1) + bitshift(input_bit, L - 1); s_idx = s + 1; if s_idx >= 1 && s_idx <= num_states, gamma_val = log_gamma(k, s_prime_idx, bit_idx); beta_next = log_beta(k + 1, s_idx); if ~isinf(beta_next) && ~isinf(gamma_val), current_val = beta_next + gamma_val; if current_val > max_val, max_val = current_val; end; end; end; end; log_beta(k, s_prime_idx) = max_val; end; end
    % LLR
    llr = zeros(1, K_total);
    for k = 1:K_total, lambda_1 = -inf; lambda_0 = -inf; for s_prime_idx = 1:num_states, s_prime = s_prime_idx - 1; for bit_idx = 1:modOrder, input_bit = bit_idx - 1; s = bitshift(s_prime, -1) + bitshift(input_bit, L - 1); s_idx = s + 1; if s_idx >= 1 && s_idx <= num_states, alpha_val = log_alpha(k, s_prime_idx); gamma_val = log_gamma(k, s_prime_idx, bit_idx); beta_val = log_beta(k + 1, s_idx); if ~isinf(alpha_val) && ~isinf(gamma_val) && ~isinf(beta_val), metric = alpha_val + gamma_val + beta_val; if input_bit == 1, if metric > lambda_1, lambda_1 = metric; end; else, if metric > lambda_0, lambda_0 = metric; end; end; end; end; end; end; if isinf(lambda_1) && isinf(lambda_0), llr(k) = 0; elseif isinf(lambda_1), llr(k) = -100; elseif isinf(lambda_0), llr(k) = 100; else, llr(k) = lambda_1 - lambda_0; end; end

    % Sequence Estimation
    detected_bits = double(llr > 0);

    % --- Update Counters ---
    message_indices = (L+1):(L+N);
    current_bit_errors = sum(tx_bits(message_indices) ~= detected_bits(message_indices));
    current_frame_error = double(current_bit_errors > 0);

    total_bit_errors = total_bit_errors + current_bit_errors;
    total_frame_errors = total_frame_errors + current_frame_error;
    total_bits_simulated = total_bits_simulated + N; % Count only message bits

    % --- Display Progress ---
    if mod(loop_count, 50) == 0 || loop_count == 1
        ber_current = total_bit_errors / total_bits_simulated;
        fer_current = total_frame_errors / loop_count;
        fprintf('Loop: %d/%d | Bit Errors: %d (%d total) | Frame Errors: %d (%d total) | BER: %.2e | FER: %.3f\n', ...
                loop_count, max_loops, current_bit_errors, total_bit_errors, current_frame_error, total_frame_errors, ber_current, fer_current);
    end

    % --- Check Stop Condition ---
    if total_bit_errors >= min_bit_errors && total_frame_errors >= max_frame_errors % Check combined condition more robustly
         fprintf('\nStopping early: Reached minimum bit errors AND maximum frame errors.\n');
         break;
    end

end % End Monte Carlo Loop

toc; % Stop timer

%% Final Results
% =========================================================================
fprintf('\n--- Final Simulation Results ---\n');
if total_bits_simulated > 0
    final_ber = total_bit_errors / total_bits_simulated;
    fprintf('Total Loops: %d\n', loop_count);
    fprintf('Total Message Bits Simulated: %d\n', total_bits_simulated);
    fprintf('Total Bit Errors: %d\n', total_bit_errors);
    fprintf('Final BER: %.4e\n', final_ber);
    fprintf('Total Frame Errors: %d\n', total_frame_errors);
    if loop_count > 0
        final_fer = total_frame_errors / loop_count;
        fprintf('Final FER: %.4f\n', final_fer);
    end
else
    fprintf('No bits simulated.\n');
end
fprintf('----------------------------------\n');


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