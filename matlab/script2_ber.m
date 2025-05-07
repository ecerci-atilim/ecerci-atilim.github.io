% --- Main Simulation Script ---
clear; clc; close all;

% --- Parameters ---
% Message Params
N = 5;         % Number of information bits (Increase for smoother BER curves)
L = 2;           % Memory length (determines states = 2^L)

% FTN Params
tau_set = [0.6, 0.9]; % Available time acceleration factors
tau_constructive = 0.6;
tau_destructive = 0.9;

% Pulse Shaping Params
alpha = 0.3;     % Roll-off factor
gdelay = 4;      % Group delay in symbol periods (determines filter length)
fs = 10;         % Samples per symbol period T (fs/fd)
fd = 1;          % Symbol rate (Nyquist)
T = 1/fd;        % Nyquist symbol period

% Simulation Params
SNR_dB_list = 0:10; % SNR values in dB (Eb/N0)
num_trials = 5000;   % Number of Monte Carlo trials per SNR (Increase for accuracy)
plot_signals = false; % Flag to plot example signals (for debugging first run)

% Derived Params
N_total = N + 2*L; % Total bits including preamble/tail
num_states = 2^L;
samples_per_symbol_nyquist = fs / fd;
h = rcosdesign(alpha, 2 * gdelay, samples_per_symbol_nyquist, 'sqrt'); % Tx/Rx Filter
g = conv(h, h); % Overall pulse response p(t) = h(t) * h(-t)
g_peak_val = max(g);
g_peak_idx = find(g == g_peak_val, 1); % Peak location offset due to pulse
g_len = length(g);
g = g / sqrt(sum(g.^2)); % Normalize overall pulse energy (important for SNR calc)

% --- BPSK Mapping ---
% Using standard mapping: 0 -> +1, 1 -> -1
map_bpsk = @(bits) (1 - 2 * bits);
unmap_bpsk = @(symbols) (1 - symbols) / 2; % Maps +1 -> 0, -1 -> 1

% --- State Definition Helper ---
% State S_k = (b_k, b_{k-1}, ..., b_{k-L+1})
% Index s_k = bi2de([b_k, ..., b_{k-L+1}], 'left-msb') + 1
get_state_idx = @(bits) bi2de(bits, 'left-msb') + 1;
get_state_bits = @(idx) de2bi(idx-1, L, 'left-msb');

% --- Results Storage ---
ber_results = zeros(size(SNR_dB_list));

% --- Simulation Loop ---
fprintf('Starting Simulation (N=%d, L=%d, Trials=%d)...\n', N, N_total, num_trials);
sim_start_time = tic;

for snr_idx = 1:length(SNR_dB_list)
    SNR_dB = SNR_dB_list(snr_idx);
    fprintf('Running SNR = %.1f dB\n', SNR_dB);

    % Calculate noise variance per sample for likelihood calc (N0/2)
    SNR_lin = 10^(SNR_dB/10); % Linear Eb/N0
    Eb = sum(abs(map_bpsk([0 1])).^2)/2; % Energy per bit (should be 1 for +/-1 symbols)
    Es = Eb; % Energy per symbol for BPSK
    N0 = Es / SNR_lin;
    sigma2 = N0 / 2; % Noise variance (double-sided PSD N0/2) for AWGN model used in likelihood
    sigma = sqrt(sigma2); % Noise standard deviation

    num_errors = 0;
    total_bits_processed = 0;

    trial_loop_start_time = tic;
    for trial = 1:num_trials
        if mod(trial, max(1, floor(num_trials/10))) == 0 && trial > 1
            elapsed_trial_time = toc(trial_loop_start_time);
            est_rem_time = (elapsed_trial_time / trial) * (num_trials - trial);
            fprintf('  Trial %d / %d (Est. Rem. Time: %.1f s)\n', trial, num_trials, est_rem_time);
        end

        % --- Transmitter ---
        % 1. Generate Info Bits
        info_bits = randi([0 1], 1, N);

        % 2. Add Preamble/Tail (L zeros at start, L zeros at end)
        tx_bits = [zeros(1, L), info_bits, zeros(1, L)];

        % 3. BPSK Modulation
        tx_symbols = map_bpsk(tx_bits); % 0->+1, 1->-1

        % 4. Adaptive Tau Sequence & Timing (Calculate peak indices first)
        taus = zeros(1, N_total - 1);
        delta_samples = zeros(1, N_total - 1);
        symbol_peak_indices = zeros(1, N_total);
        current_idx = g_peak_idx; % First symbol peaks after filter delay
        symbol_peak_indices(1) = current_idx;

        for k = 1:(N_total - 1)
            if sign(tx_symbols(k)) == sign(tx_symbols(k+1))
                current_tau = tau_constructive;
            else
                current_tau = tau_destructive;
            end
            taus(k) = current_tau;
            delta_k = round(current_tau * samples_per_symbol_nyquist);
            delta_samples(k) = delta_k;
            current_idx = current_idx + delta_k;
            symbol_peak_indices(k+1) = current_idx;
        end

        % 5. Non-uniform Upsampling & Pulse Shaping (Revised Allocation)
        % Calculate exact required length based on last peak and pulse extent
        max_req_idx = symbol_peak_indices(end) + (g_len - g_peak_idx);
        signal_len = max_req_idx; % Allocate exactly what's needed

        % Check for potential non-positive length (e.g., if N=0, L=0)
        if signal_len <= 0
            error('Calculated signal length is non-positive. Check N, L, gdelay.');
        end

        tx_signal_noiseless = zeros(1, signal_len);

        for k = 1:N_total
            peak_idx = round(symbol_peak_indices(k)); % Ensure integer index
            start_idx = peak_idx - (g_peak_idx - 1);
            end_idx = peak_idx + (g_len - g_peak_idx); % Correct end index calculation relative to peak

            % Indices within the pulse 'g'
            pulse_start = 1;
            pulse_end = g_len;

            % Indices within the signal 'tx_signal_noiseless'
            sig_start = start_idx;
            sig_end = end_idx;

            % Adjust indices if pulse goes out of bounds (start < 1 or end > signal_len)
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
            else
                 % This might indicate an issue if it happens often, e.g., rounding errors
                 % fprintf('Warning: Index mismatch or invalid range during pulse shaping for symbol %d (sig %d:%d, pulse %d:%d)\n', k, sig_start, sig_end, pulse_start, pulse_end);
            end
        end
        % --- No trimming line needed here anymore ---

        % --- Channel ---
        % Add AWGN
        noise = sigma * randn(size(tx_signal_noiseless));
        rx_signal = tx_signal_noiseless + noise;

        % --- Receiver (Max-Log-BCJR) ---
        [log_alpha, log_beta, log_gamma_storage, timing_p, pred_state] = max_log_bcjr(rx_signal, N_total, L, num_states, tau_set, samples_per_symbol_nyquist, g, g_peak_idx, sigma2, map_bpsk, get_state_bits, get_state_idx);

        % --- Calculate LLRs (Revised) ---
        LLR = zeros(1, N_total);
        for k = 1:N_total % Calculate LLR for bit b_k
            max_term_bit0 = -inf; % Max combined metric for transitions where b_k = 0
            max_term_bit1 = -inf; % Max combined metric for transitions where b_k = 1

            for prev_state_idx = 1:num_states
                if isinf(log_alpha(k, prev_state_idx)) continue; end % Skip unreachable previous states

                for current_bit_val = [0, 1] % Hypothesized b_k
                    % Determine next_state based on prev_state and current_bit
                    bits_prev = get_state_bits(prev_state_idx); % S_{k-1} = (b_{k-1}, ..., b_{k-L})
                    if L == 0
                        next_state_idx = 1; % Only one state
                    else
                        % S_k = (b_k, b_{k-1}, ..., b_{k-L+1})
                        bits_next = [current_bit_val, bits_prev(1:L-1)];
                        next_state_idx = get_state_idx(bits_next);
                    end

                    % Retrieve stored log_gamma for this transition (k, prev_state, current_bit)
                    lg = log_gamma_storage(k, prev_state_idx, current_bit_val+1);
                    if isinf(lg) continue; end % Skip impossible transitions

                    % Combine alpha, gamma, beta (Max-Log version)
                    % Term = alpha_{k-1}(prev) * gamma_k(prev,next) * beta_k(next)
                    % Indices: log_alpha(k, prev), lg, log_beta(k+1, next)
                    term = log_alpha(k, prev_state_idx) + lg + log_beta(k+1, next_state_idx);

                    % Update max term based on the current bit value
                    if current_bit_val == 0 % Bit 0 (+1 symbol)
                        max_term_bit0 = max(max_term_bit0, term);
                    else % Bit 1 (-1 symbol)
                        max_term_bit1 = max(max_term_bit1, term);
                    end
                end % loop current_bit_val
            end % loop prev_state_idx

            % Calculate LLR = log(P(bk=0)/P(bk=1)) approx max_term_bit0 - max_term_bit1
            LLR(k) = max_term_bit0 - max_term_bit1;

            % Handle cases where one hypothesis might be impossible (-inf)
            if isinf(max_term_bit0) && ~isinf(max_term_bit1)
                LLR(k) = -inf; % Definitely bit 1
            elseif ~isinf(max_term_bit0) && isinf(max_term_bit1)
                LLR(k) = inf;  % Definitely bit 0
            elseif isinf(max_term_bit0) && isinf(max_term_bit1)
                LLR(k) = 0;    % Undecidable / Error state (assign arbitrary or zero LLR)
            end
        end % loop k for LLR

        % --- Decision ---
        % LLR > 0 means bit 0 (+1 symbol) is more likely
        estimated_bits = (LLR < 0); % LLR<0 -> bit 1 (-1 symbol) more likely

        % --- BER Calculation ---
        % Compare info bits only (exclude preamble/tail)
        info_indices = (L+1):(N+L);
        errors_this_trial = sum(estimated_bits(info_indices) ~= info_bits);
        num_errors = num_errors + errors_this_trial;
        total_bits_processed = total_bits_processed + N;

        % --- Plotting (Optional) ---
        if plot_signals && trial == 1 && snr_idx == 1
            figure;
            subplot(3,1,1); plot(tx_signal_noiseless); title('Noiseless Tx Signal'); xlim([0 length(tx_signal_noiseless)]);
            hold on; stem(symbol_peak_indices, tx_symbols, 'r', 'Marker', 'none'); hold off;
            ylabel('Amplitude');
            subplot(3,1,2); plot(rx_signal); title(['Received Signal (SNR = ' num2str(SNR_dB) ' dB)']); xlim([0 length(rx_signal)]);
             ylabel('Amplitude');
            subplot(3,1,3); stem(1:N_total, LLR); title('LLRs'); xlim([0 N_total+1]);
            xlabel('Bit Index'); ylabel('LLR');
            sgtitle('Example Signals (Trial 1, SNR 0 dB)');
            drawnow;
            plot_signals = false; % Plot only once
        end

    end % trial loop

    ber = num_errors / total_bits_processed;
    % Handle case where num_errors is 0 to avoid log(0) in plots
    if ber == 0
        ber = 1 / (total_bits_processed * 10); % Assign a small non-zero value below typical plot limits
    end
    ber_results(snr_idx) = ber;
    fprintf('  SNR: %.1f dB, BER: %.6f (%d errors / %d bits)\n', SNR_dB, ber, num_errors, total_bits_processed);

end % snr loop

total_time = toc(sim_start_time);
fprintf('Simulation finished in %.2f seconds (%.2f min).\n', total_time, total_time/60);

% --- Plot BER Curve ---
figure;
semilogy(SNR_dB_list, ber_results, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
grid on;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title(sprintf('Adaptive FTN (\\tau=%.1f/%.1f) with Max-Log-BCJR (L=%d, N=%d)', tau_constructive, tau_destructive, L, N));
min_ber_plot = max(1e-7, min(ber_results(ber_results>0))/5); % Avoid log(0) or negative limits
ylim([min_ber_plot 1]); % Adjust y-axis limits dynamically
legend('Simulation');

% --- Max-Log-BCJR Function ---
function [log_alpha, log_beta, log_gamma_storage, timing_p, pred_state] = max_log_bcjr(rx_signal, N_total, L, num_states, tau_set, samples_per_symbol, g, g_peak_idx, sigma2, map_bpsk, get_state_bits, get_state_idx)

    % Initialization
    log_alpha = -inf(N_total + 1, num_states); % log_alpha(k, state) for state at time k-1
    log_beta = -inf(N_total + 1, num_states);  % log_beta(k, state) for state at time k-1
    timing_p = nan(N_total + 1, num_states); % Store peak timing p_{k-1} for path ending in state at time k
    pred_state = zeros(N_total + 1, num_states, 'uint32'); % Store predecessor state index for path traceback
    log_gamma_storage = -inf(N_total, num_states, 2); % Store gamma(k, prev_state, current_bit)

    % Initial state (k=1, corresponds to state before first bit b_1)
    initial_state_bits = zeros(1, L); % Represents (b_0, b_{-1}, ..., b_{-L+1})
    initial_state_idx = get_state_idx(initial_state_bits);
    log_alpha(1, initial_state_idx) = 0;
    % timing_p(1, initial_state_idx) = 0; % Reference time p_0 (handled in gamma k=1)

    % --- Forward Recursion (alpha) ---
    for k = 1:N_total % Process bit k (transition from time k-1 to k)
        for prev_state_idx = 1:num_states % State S_{k-1} = (b_{k-1}, ..., b_{k-L})
            if isinf(log_alpha(k, prev_state_idx)) continue; end % Skip unreachable states

            for current_bit_val = [0, 1] % Hypothesized bit b_k
                % Determine next state S_k = (b_k, b_{k-1}, ..., b_{k-L+1})
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

                if isinf(lg) continue; end % Skip if gamma is impossible

                % Update log_alpha(k+1, next_state_idx) using max*
                % Note: alpha index k+1 corresponds to state S_k at time k
                new_log_alpha = log_alpha(k, prev_state_idx) + lg; % A priori prob assumed uniform

                if new_log_alpha > log_alpha(k+1, next_state_idx)
                    log_alpha(k+1, next_state_idx) = new_log_alpha;
                    % Store timing p_k and predecessor state S_{k-1} index
                    timing_p(k+1, next_state_idx) = pk_current;
                    pred_state(k+1, next_state_idx) = prev_state_idx;
                end
            end % current_bit_val loop
        end % prev_state_idx loop

        % Normalization (optional but recommended for long sequences)
        max_log_alpha = max(log_alpha(k+1, :));
        if ~isinf(max_log_alpha) && max_log_alpha ~= 0 % Avoid subtracting from all -inf
             log_alpha(k+1, :) = log_alpha(k+1, :) - max_log_alpha;
        end

    end % k loop (forward)

    % --- Backward Recursion (beta) ---
    % Initialize beta at time N_total (state S_{N_total})
    final_state_bits = zeros(1, L); % S_{N_total} = (b_{N_total}, ..., b_{N_total-L+1})
    final_state_idx = get_state_idx(final_state_bits);
    log_beta(N_total + 1, :) = -inf; % Initialize all to -inf first
    log_beta(N_total + 1, final_state_idx) = 0; % Beta value corresponding to state S_{N_total}

    for k = N_total:-1:1 % Process bit k (transition from k-1 to k)
        % Calculate log_beta(k, prev_state_idx) based on log_beta(k+1, next_state_idx)
        temp_log_beta_k = -inf(1, num_states); % Store new values temporarily

        for prev_state_idx = 1:num_states % State S_{k-1}
            current_max_log_beta = -inf;

            for current_bit_val = [0, 1] % Hypothesized bit b_k
                % Determine next state S_k based on prev_state S_{k-1} and b_k
                 bits_prev = get_state_bits(prev_state_idx);
                 if L == 0
                    next_state_idx = 1;
                 else
                    bits_next = [current_bit_val, bits_prev(1:L-1)];
                    next_state_idx = get_state_idx(bits_next);
                 end

                % Retrieve stored log_gamma(k, prev_state, current_bit)
                lg = log_gamma_storage(k, prev_state_idx, current_bit_val+1);
                if isinf(lg) continue; end

                % Update log_beta(k, prev_state_idx) using max*
                % term = log_gamma_k(prev, next) + log_beta_{k}(next)
                term = lg + log_beta(k+1, next_state_idx);
                current_max_log_beta = max(current_max_log_beta, term);

            end % current_bit_val loop
            temp_log_beta_k(prev_state_idx) = current_max_log_beta;
        end % prev_state_idx loop

        % Assign calculated values for time k
        log_beta(k, :) = temp_log_beta_k;

         % Normalization (match alpha normalization)
        max_log_beta = max(log_beta(k, :));
        if ~isinf(max_log_beta) && max_log_beta ~= 0
             log_beta(k, :) = log_beta(k, :) - max_log_beta;
        end

    end % k loop (backward)

end % max_log_bcjr function


% --- Calculate Log Gamma Function (Revised Traceback) ---
function [log_gamma, pk_current] = calculate_log_gamma(rx_signal, k, prev_state_idx, current_bit_val, timing_p, pred_state, L, tau_set, samples_per_symbol, g, g_peak_idx, sigma2, map_bpsk, get_state_bits, get_state_idx)

    log_gamma = -inf; % Default for invalid transitions
    pk_current = NaN;

    % --- 1. Determine Symbols and Tau ---
    current_symbol = map_bpsk(current_bit_val); % s_k
    bits_prev_state = get_state_bits(prev_state_idx); % S_{k-1} = (b_{k-1}, ..., b_{k-L})

    if k == 1
        % First bit b_1. Previous state represents (b_0, ..., b_{-L+1}) = (0,...,0)
        prev_symbol = map_bpsk(0); % Symbol s_0 (from preamble assumption)
    else
        % Get symbol s_{k-1} from the previous state S_{k-1}
        prev_symbol = map_bpsk(bits_prev_state(1)); % b_{k-1}
    end

    % Determine tau_k
    if sign(prev_symbol) == sign(current_symbol)
        current_tau = tau_set(1); % Constructive
    else
        current_tau = tau_set(2); % Destructive
    end
    delta_pk = round(current_tau * samples_per_symbol);

    % --- 2. Determine Hypothesized Peak Timing pk_current ---
    if k == 1
        % Define the peak of the first symbol b_1
        pk_current = g_peak_idx; % Assume first symbol peak is at filter delay
    else
        % Find timing pk-1 associated with the path ending at prev_state at time k
        pk_prev = timing_p(k, prev_state_idx); % p_{k-1} stored from forward pass
        if isnan(pk_prev) || isinf(pk_prev) return; end % Previous state unreachable or timing invalid
        pk_current = pk_prev + delta_pk; % Hypothesized peak location for symbol k
    end

    % --- 3. Reconstruct Hypothesized Signal Segment ---
    symbols = zeros(1, L + 1);
    timings = nan(1, L + 1); % Use NaN for invalid/unset timings

    symbols(1) = current_symbol; % s_k
    timings(1) = pk_current;     % p_k

    % Trace back path using stored timing_p and pred_state (Revised Logic)
    current_trace_state_idx = prev_state_idx; % State S_{k-1} index
    current_trace_time_idx = k; % Time index in alpha/timing/pred arrays

    for j = 1:L % Get s_{k-j} and p_{k-j}
        if current_trace_time_idx <= 1 || current_trace_state_idx == 0
            break; % Reached beginning or invalid state
        end

        % State S_{k-j} index is current_trace_state_idx
        state_idx_kmj = current_trace_state_idx;

        % Symbol s_{k-j} is the first bit in state S_{k-j} = (b_{k-j}, ..., b_{k-j-L+1})
        bits_state_kmj = get_state_bits(state_idx_kmj);
        symbols(j+1) = map_bpsk(bits_state_kmj(1)); % Symbol s_{k-j}

        % Timing p_{k-j} is stored in timing_p(k-j+1, state_idx_kmj)
        timings(j+1) = timing_p(current_trace_time_idx, state_idx_kmj); % p_{k-j}

        % Move to previous step in traceback: Find state S_{k-j-1}
        state_idx_kmj_minus_1 = pred_state(current_trace_time_idx, current_trace_state_idx);
        if state_idx_kmj_minus_1 == 0 break; end % Invalid predecessor

        current_trace_state_idx = state_idx_kmj_minus_1; % Move to state S_{k-j-1}
        current_trace_time_idx = current_trace_time_idx - 1; % Move to time k-j
    end % Traceback loop j

    % --- Construct hypothesized signal s_hyp ---
    g_len = length(g);
    valid_timings = timings(~isnan(timings));
    if isempty(valid_timings) log_gamma = -inf; return; end % No valid timings found

    min_pk = min(valid_timings);
    max_pk = max(valid_timings);

    % Define window based on pulse extent around earliest/latest peaks
    win_start = floor(min_pk - g_len/2);
    win_end = ceil(max_pk + g_len/2);
    rx_start_idx = max(1, win_start);
    rx_end_idx = min(length(rx_signal), win_end);
    rx_indices = rx_start_idx:rx_end_idx;

    if isempty(rx_indices) log_gamma = -inf; return; end % Window invalid

    s_hyp = zeros(size(rx_indices));
    rx_segment = rx_signal(rx_indices);

    for j = 1:(L+1) % Sum contributions from s_k down to s_{k-L}
        sym = symbols(j);
        pk = timings(j);
        if isnan(pk) || sym == 0 continue; end % Skip if timing invalid or symbol is zero

        pk_int = round(pk); % Use integer index for pulse placement
        pulse_indices_in_g = (rx_indices - pk_int) + g_peak_idx;

        valid_g_mask = (pulse_indices_in_g >= 1) & (pulse_indices_in_g <= g_len);
        valid_pulse_indices = pulse_indices_in_g(valid_g_mask);
        valid_s_hyp_indices = find(valid_g_mask);

        if ~isempty(valid_s_hyp_indices)
            s_hyp(valid_s_hyp_indices) = s_hyp(valid_s_hyp_indices) + sym * g(valid_pulse_indices);
        end
    end

    % --- 4. Calculate Log Likelihood (Squared Error) ---
    if length(rx_segment) ~= length(s_hyp)
         % fprintf('Warning k=%d: rx_segment (%d) and s_hyp (%d) length mismatch.\n', k, length(rx_segment), length(s_hyp));
         log_gamma = -inf; return; % Length mismatch error
    end

    squared_error = sum((rx_segment - s_hyp).^2);
    % Add a small epsilon to sigma2 to prevent division by zero if sigma2 is ever zero
    log_gamma = -squared_error / (2 * sigma2 + eps);

    if isnan(log_gamma) || isinf(log_gamma)
        log_gamma = -inf; % Treat NaN/Inf as impossible transition
    end
end