% Improved Adaptive FTN with Max-Log-BCJR Receiver
% Incorporating logic from the provided implementation
clear all;
close all;

% --- Parameters ---
% Message Params
N = 100;        % Number of information bits
L = 3;          % Memory length (determines states = 2^L)

% FTN Params
tau_set = [0.6, 0.9]; % Available time acceleration factors
tau_constructive = tau_set(1);
tau_destructive = tau_set(2);

% Pulse Shaping Params
alpha = 0.3;    % Roll-off factor
gdelay = 4;     % Group delay in symbol periods
fs = 10;        % Samples per symbol period T (fs/fd)
fd = 1;         % Symbol rate (Nyquist)
T = 1/fd;       % Nyquist symbol period

% Simulation Params
SNR_dB_list = 0:10; % SNR values in dB (Eb/N0)
num_trials = 100;     % Number of Monte Carlo trials per SNR

% Derived Params
N_total = N + 2*L;    % Total bits including preamble/tail
num_states = 2^L;
samples_per_symbol_nyquist = fs / fd;

% Generate RRC pulse
h = rcosdesign(alpha, 2 * gdelay, samples_per_symbol_nyquist, 'sqrt'); % Tx/Rx Filter
g = conv(h, h); % Overall pulse response p(t) = h(t) * h(-t)
g_peak_val = max(g);
g_peak_idx = find(g == g_peak_val, 1); % Peak location offset due to pulse
g_len = length(g);
g = g / sqrt(sum(g.^2)); % Normalize overall pulse energy (for SNR calc)

% BPSK Mapping: 0 -> +1, 1 -> -1
map_bpsk = @(bits) (1 - 2 * bits);
unmap_bpsk = @(symbols) (1 - symbols) / 2; % Maps +1 -> 0, -1 -> 1

% State Definition Helper Functions
get_state_idx = @(bits) bi2de(bits, 'left-msb') + 1;
get_state_bits = @(idx) de2bi(idx-1, L, 'left-msb');

% Results Storage
ber_results = zeros(size(SNR_dB_list));

% --- Simulation Loop ---
fprintf('Starting Simulation (N=%d, L=%d, Trials=%d)...\n', N, L, num_trials);
sim_start_time = tic;

for snr_idx = 1:length(SNR_dB_list)
    SNR_dB = SNR_dB_list(snr_idx);
    fprintf('Running SNR = %.1f dB\n', SNR_dB);

    % Calculate noise variance for likelihood calculation
    SNR_lin = 10^(SNR_dB/10); % Linear Eb/N0
    Eb = 1; % Energy per bit (normalized BPSK)
    Es = Eb; % Energy per symbol for BPSK
    N0 = Es / SNR_lin;
    sigma2 = N0 / 2; % Noise variance
    sigma = sqrt(sigma2); % Noise standard deviation

    num_errors = 0;
    total_bits = 0;

    trial_start_time = tic;
    for trial = 1:num_trials
        if mod(trial, max(1, floor(num_trials/10))) == 0
            elapsed_time = toc(trial_start_time);
            est_rem_time = (elapsed_time / trial) * (num_trials - trial);
            fprintf('  Trial %d / %d (Est. Rem. Time: %.1f s)\n', trial, num_trials, est_rem_time);
        end

        % --- Transmitter ---
        % 1. Generate Random Info Bits
        info_bits = randi([0 1], 1, N);

        % 2. Add Preamble/Tail (zeros)
        tx_bits = [zeros(1, L), info_bits, zeros(1, L)];

        % 3. BPSK Modulation
        tx_symbols = map_bpsk(tx_bits); % 0->+1, 1->-1

        % 4. Adaptive Tau Sequence & Timing
        taus = zeros(1, N_total - 1);
        delta_samples = zeros(1, N_total - 1);
        symbol_peak_indices = zeros(1, N_total);
        current_idx = g_peak_idx; % First symbol after filter delay
        symbol_peak_indices(1) = current_idx;

        for k = 1:(N_total - 1)
            if sign(tx_symbols(k)) == sign(tx_symbols(k+1))
                current_tau = tau_constructive; % Same sign -> constructive ISI
            else
                current_tau = tau_destructive; % Opposite sign -> destructive ISI
            end
            taus(k) = current_tau;
            delta_k = round(current_tau * samples_per_symbol_nyquist);
            delta_samples(k) = delta_k;
            current_idx = current_idx + delta_k;
            symbol_peak_indices(k+1) = current_idx;
        end

        % 5. Non-uniform Upsampling & Pulse Shaping
        max_req_idx = symbol_peak_indices(end) + (g_len - g_peak_idx);
        signal_len = max_req_idx;
        
        tx_signal_noiseless = zeros(1, signal_len);

        for k = 1:N_total
            peak_idx = round(symbol_peak_indices(k));
            start_idx = peak_idx - (g_peak_idx - 1);
            end_idx = peak_idx + (g_len - g_peak_idx);

            % Indices within the pulse 'g'
            pulse_start = 1;
            pulse_end = g_len;

            % Indices within the signal 'tx_signal_noiseless'
            sig_start = start_idx;
            sig_end = end_idx;

            % Adjust indices if pulse goes out of bounds
            if sig_start < 1
                pulse_start = pulse_start + (1 - sig_start);
                sig_start = 1;
            end
            if sig_end > signal_len
                pulse_end = pulse_end - (sig_end - signal_len);
                sig_end = signal_len;
            end

            % Check for valid range and consistent length before adding
            if sig_start <= sig_end && pulse_start <= pulse_end && (sig_end - sig_start == pulse_end - pulse_start)
                tx_signal_noiseless(sig_start:sig_end) = tx_signal_noiseless(sig_start:sig_end) + tx_symbols(k) * g(pulse_start:pulse_end);
            end
        end

        % --- Channel ---
        % Add AWGN
        noise = sigma * randn(size(tx_signal_noiseless));
        rx_signal = tx_signal_noiseless + noise;

        % --- Receiver (Max-Log-BCJR) ---
        [log_alpha, log_beta, log_gamma_storage, timing_p, pred_state] = max_log_bcjr(rx_signal, N_total, L, num_states, tau_set, samples_per_symbol_nyquist, g, g_peak_idx, sigma2, map_bpsk, get_state_bits, get_state_idx);

        % --- Calculate LLRs ---
        LLR = zeros(1, N_total);
        for k = 1:N_total
            max_term_bit0 = -inf; % Max metric for transitions where bit = 0
            max_term_bit1 = -inf; % Max metric for transitions where bit = 1

            for prev_state_idx = 1:num_states
                if isinf(log_alpha(k, prev_state_idx)) continue; end

                for current_bit_val = [0, 1]
                    % Determine next state based on prev_state and current_bit
                    bits_prev = get_state_bits(prev_state_idx);
                    if L == 0
                        next_state_idx = 1;
                    else
                        bits_next = [current_bit_val, bits_prev(1:L-1)];
                        next_state_idx = get_state_idx(bits_next);
                    end

                    % Retrieve log_gamma for this transition
                    lg = log_gamma_storage(k, prev_state_idx, current_bit_val+1);
                    if isinf(lg) continue; end

                    % Combine alpha, gamma, beta for this path
                    term = log_alpha(k, prev_state_idx) + lg + log_beta(k+1, next_state_idx);

                    % Update max term based on current bit value
                    if current_bit_val == 0
                        max_term_bit0 = max(max_term_bit0, term);
                    else
                        max_term_bit1 = max(max_term_bit1, term);
                    end
                end
            end

            % Calculate LLR = log(P(bit=0)/P(bit=1))
            LLR(k) = max_term_bit0 - max_term_bit1;

            % Handle special cases
            if isinf(max_term_bit0) && ~isinf(max_term_bit1)
                LLR(k) = -inf; % Definitely bit 1
            elseif ~isinf(max_term_bit0) && isinf(max_term_bit1)
                LLR(k) = inf;  % Definitely bit 0
            elseif isinf(max_term_bit0) && isinf(max_term_bit1)
                LLR(k) = 0;    % Undecidable (assign zero LLR)
            end
        end

        % --- Decision ---
        % LLR > 0 means bit 0 is more likely
        estimated_bits = (LLR < 0);

        % --- BER Calculation ---
        % Compare info bits only (exclude preamble/tail)
        info_indices = (L+1):(N+L);
        errors_this_trial = sum(estimated_bits(info_indices) ~= info_bits);
        num_errors = num_errors + errors_this_trial;
        total_bits = total_bits + N;
    end

    % Calculate BER for this SNR
    ber = num_errors / total_bits;
    if ber == 0
        ber = 1 / (total_bits * 10); % Small non-zero value for log plot
    end
    ber_results(snr_idx) = ber;
    fprintf('  SNR: %.1f dB, BER: %.6f (%d errors / %d bits)\n', SNR_dB, ber, num_errors, total_bits);
end

total_time = toc(sim_start_time);
fprintf('Simulation finished in %.2f seconds (%.2f min).\n', total_time, total_time/60);

% --- Plot BER Curve ---
figure;
semilogy(SNR_dB_list, ber_results, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('Bit Error Rate (BER)');
title(sprintf('Adaptive FTN (\\tau=%.1f/%.1f) with Max-Log-BCJR (L=%d, N=%d)', tau_constructive, tau_destructive, L, N));
min_ber_plot = max(1e-7, min(ber_results(ber_results>0))/5);
ylim([min_ber_plot 1]);
legend('Simulation');

% --- Max-Log-BCJR Function ---
function [log_alpha, log_beta, log_gamma_storage, timing_p, pred_state] = max_log_bcjr(rx_signal, N_total, L, num_states, tau_set, samples_per_symbol, g, g_peak_idx, sigma2, map_bpsk, get_state_bits, get_state_idx)
    % Initialization
    log_alpha = -inf(N_total + 1, num_states);
    log_beta = -inf(N_total + 1, num_states);
    timing_p = nan(N_total + 1, num_states);
    pred_state = zeros(N_total + 1, num_states, 'uint32');
    log_gamma_storage = -inf(N_total, num_states, 2);

    % Initial state (all zeros)
    initial_state_bits = zeros(1, L);
    initial_state_idx = get_state_idx(initial_state_bits);
    log_alpha(1, initial_state_idx) = 0;

    % --- Forward Recursion (alpha) ---
    for k = 1:N_total
        for prev_state_idx = 1:num_states
            if isinf(log_alpha(k, prev_state_idx)) continue; end

            for current_bit_val = [0, 1]
                % Determine next state
                bits_prev = get_state_bits(prev_state_idx);
                if L == 0
                    next_state_idx = 1; bits_next = [];
                else
                    bits_next = [current_bit_val, bits_prev(1:L-1)];
                    next_state_idx = get_state_idx(bits_next);
                end

                % Calculate log_gamma for this transition
                [lg, pk_current] = calculate_log_gamma(rx_signal, k, prev_state_idx, current_bit_val, timing_p, pred_state, L, tau_set, samples_per_symbol, g, g_peak_idx, sigma2, map_bpsk, get_state_bits, get_state_idx);
                log_gamma_storage(k, prev_state_idx, current_bit_val+1) = lg;

                if isinf(lg) continue; end

                % Update log_alpha using max (max-log approximation)
                new_log_alpha = log_alpha(k, prev_state_idx) + lg;

                if new_log_alpha > log_alpha(k+1, next_state_idx)
                    log_alpha(k+1, next_state_idx) = new_log_alpha;
                    timing_p(k+1, next_state_idx) = pk_current;
                    pred_state(k+1, next_state_idx) = prev_state_idx;
                end
            end
        end

        % Normalization
        max_log_alpha = max(log_alpha(k+1, :));
        if ~isinf(max_log_alpha) && max_log_alpha ~= 0
            log_alpha(k+1, :) = log_alpha(k+1, :) - max_log_alpha;
        end
    end

    % --- Backward Recursion (beta) ---
    % Initialize beta at final state (all zeros)
    final_state_bits = zeros(1, L);
    final_state_idx = get_state_idx(final_state_bits);
    log_beta(N_total + 1, :) = -inf;
    log_beta(N_total + 1, final_state_idx) = 0;

    for k = N_total:-1:1
        temp_log_beta_k = -inf(1, num_states);

        for prev_state_idx = 1:num_states
            current_max_log_beta = -inf;

            for current_bit_val = [0, 1]
                % Determine next state
                bits_prev = get_state_bits(prev_state_idx);
                if L == 0
                    next_state_idx = 1;
                else
                    bits_next = [current_bit_val, bits_prev(1:L-1)];
                    next_state_idx = get_state_idx(bits_next);
                end

                % Retrieve stored log_gamma
                lg = log_gamma_storage(k, prev_state_idx, current_bit_val+1);
                if isinf(lg) continue; end

                % Update log_beta using max (max-log approximation)
                term = lg + log_beta(k+1, next_state_idx);
                current_max_log_beta = max(current_max_log_beta, term);
            end
            temp_log_beta_k(prev_state_idx) = current_max_log_beta;
        end

        log_beta(k, :) = temp_log_beta_k;

        % Normalization
        max_log_beta = max(log_beta(k, :));
        if ~isinf(max_log_beta) && max_log_beta ~= 0
            log_beta(k, :) = log_beta(k, :) - max_log_beta;
        end
    end
end

% --- Calculate Log Gamma Function ---
function [log_gamma, pk_current] = calculate_log_gamma(rx_signal, k, prev_state_idx, current_bit_val, timing_p, pred_state, L, tau_set, samples_per_symbol, g, g_peak_idx, sigma2, map_bpsk, get_state_bits, get_state_idx)
    log_gamma = -inf;
    pk_current = NaN;

    % 1. Determine Symbols and Tau
    current_symbol = map_bpsk(current_bit_val);
    bits_prev_state = get_state_bits(prev_state_idx);

    if k == 1
        % First bit b_1, previous state is all zeros
        prev_symbol = map_bpsk(0);
    else
        % Get previous symbol from state
        prev_symbol = map_bpsk(bits_prev_state(1));
    end

    % Determine tau based on symbol signs
    if sign(prev_symbol) == sign(current_symbol)
        current_tau = tau_set(1); % Constructive ISI -> tau = 0.6
    else
        current_tau = tau_set(2); % Destructive ISI -> tau = 0.9
    end
    delta_pk = round(current_tau * samples_per_symbol);

    % 2. Determine Peak Timing
    if k == 1
        % Peak of first symbol
        pk_current = g_peak_idx;
    else
        % Find timing from previous state
        pk_prev = timing_p(k, prev_state_idx);
        if isnan(pk_prev) || isinf(pk_prev) return; end
        pk_current = pk_prev + delta_pk;
    end

    % 3. Reconstruct Signal Segment
    symbols = zeros(1, L + 1);
    timings = nan(1, L + 1);

    symbols(1) = current_symbol;
    timings(1) = pk_current;

    % Trace back path to reconstruct signal history
    current_trace_state_idx = prev_state_idx;
    current_trace_time_idx = k;

    for j = 1:L
        if current_trace_time_idx <= 1 || current_trace_state_idx == 0
            break;
        end

        state_idx_kmj = current_trace_state_idx;

        % Get symbol from state
        bits_state_kmj = get_state_bits(state_idx_kmj);
        symbols(j+1) = map_bpsk(bits_state_kmj(1));

        % Get timing from stored values
        timings(j+1) = timing_p(current_trace_time_idx, state_idx_kmj);

        % Move to previous state in traceback
        state_idx_kmj_minus_1 = pred_state(current_trace_time_idx, current_trace_state_idx);
        if state_idx_kmj_minus_1 == 0 break; end

        current_trace_state_idx = state_idx_kmj_minus_1;
        current_trace_time_idx = current_trace_time_idx - 1;
    end

    % Calculate signal window
    g_len = length(g);
    valid_timings = timings(~isnan(timings));
    if isempty(valid_timings) log_gamma = -inf; return; end

    min_pk = min(valid_timings);
    max_pk = max(valid_timings);

    % Define window based on pulse extent
    win_start = floor(min_pk - g_len/2);
    win_end = ceil(max_pk + g_len/2);
    rx_start_idx = max(1, win_start);
    rx_end_idx = min(length(rx_signal), win_end);
    rx_indices = rx_start_idx:rx_end_idx;

    if isempty(rx_indices) log_gamma = -inf; return; end

    % Generate hypothesized signal
    s_hyp = zeros(size(rx_indices));
    rx_segment = rx_signal(rx_indices);

    for j = 1:(L+1)
        sym = symbols(j);
        pk = timings(j);
        if isnan(pk) || sym == 0 continue; end

        pk_int = round(pk);
        pulse_indices_in_g = (rx_indices - pk_int) + g_peak_idx;

        valid_g_mask = (pulse_indices_in_g >= 1) & (pulse_indices_in_g <= g_len);
        valid_pulse_indices = pulse_indices_in_g(valid_g_mask);
        valid_s_hyp_indices = find(valid_g_mask);

        if ~isempty(valid_s_hyp_indices)
            s_hyp(valid_s_hyp_indices) = s_hyp(valid_s_hyp_indices) + sym * g(valid_pulse_indices);
        end
    end

    % 4. Calculate Log Likelihood (Squared Error)
    if length(rx_segment) ~= length(s_hyp)
        log_gamma = -inf; return;
    end

    squared_error = sum((rx_segment - s_hyp).^2);
    log_gamma = -squared_error / (2 * sigma2 + eps);

    if isnan(log_gamma) || isinf(log_gamma)
        log_gamma = -inf;
    end
end