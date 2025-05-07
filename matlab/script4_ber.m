% Improved Adaptive Tau FTN with M-BCJR Receiver
% Properly tracks path-specific tau sequences and handles ISI calculation
clear all;
close all;

% System parameters
L = 3;              % Memory length (state = 2^L)
N = 10;            % Message length
alpha = 0.3;        % Roll-off factor for rRC
gdelay = 4;         % Group delay
fs = 10;            % Sampling frequency
fd = 1;             % Symbol rate

% Adaptive tau parameters
tau_set = [0.6, 0.9]; % [tau_constructive, tau_destructive]
tau_constructive = tau_set(1);  % When symbols have same sign
tau_destructive = tau_set(2);   % When symbols have opposite sign

% M-BCJR parameter
M = 8;              % Number of survivor paths in M-algorithm

% SNR range for simulation
SNR_dB = 0:2:10;
num_snr = length(SNR_dB);

% Initialize BER storage
ber = zeros(1, num_snr);

% Root-raised cosine pulse generation
h = rcosdesign(alpha, 2 * gdelay, fs / fd, 'sqrt');
h_len = length(h);
g = conv(h, h);  % Overall pulse response (TX+RX filters)
g_len = length(g);
g_peak_idx = find(g == max(g), 1);  % Peak location in the overall pulse

% Normalize pulse energies
h = h / sqrt(sum(h.^2));
g = g / max(g);  % Normalize to peak of 1 for easier ISI tracking

% Number of trials per SNR point
num_trials = 200;

% Main simulation loop
fprintf('Starting Simulation (L=%d, M=%d, tau=[%.1f,%.1f])...\n', L, M, tau_set(1), tau_set(2));
sim_start_time = tic;

for snr_idx = 1:length(SNR_dB)
    SNR_dB_val = SNR_dB(snr_idx);
    fprintf('Running SNR = %.1f dB\n', SNR_dB_val);
    
    % Calculate noise variance
    SNR_lin = 10^(SNR_dB_val/10);
    Eb = 1;  % Energy per bit (normalized BPSK)
    sigma2 = Eb/(2*SNR_lin);  % Noise variance
    sigma = sqrt(sigma2);
    
    num_errors = 0;
    total_bits = 0;
    
    for trial = 1:num_trials
        % Generate random binary message
        msg_bits = randi([0, 1], 1, N);
        
        % Add preamble and tail bits for proper initialization
        tx_bits = [zeros(1, L), msg_bits, zeros(1, L)];
        full_len = length(tx_bits);
        
        % BPSK modulation (0->+1, 1->-1)
        tx_symbols = 1 - 2 * tx_bits;
        
        % Generate adaptive tau sequence based on symbol signs
        tau_sequence = zeros(1, full_len-1);
        for i = 1:full_len-1
            if tx_symbols(i) * tx_symbols(i+1) > 0  % Same sign (constructive ISI)
                tau_sequence(i) = tau_constructive;
            else  % Opposite sign (destructive ISI)
                tau_sequence(i) = tau_destructive;
            end
        end
        
        % Calculate cumulative time positions for each symbol
        time_positions = zeros(1, full_len);
        time_positions(1) = 0;  % First symbol at t=0
        for i = 2:full_len
            time_positions(i) = time_positions(i-1) + tau_sequence(i-1);
        end
        
        % Generate FTN signal with variable symbol spacing
        % Use an upsampled symbol sequence with proper timing
        max_time = time_positions(end) + 1;  % Add margin
        signal_len = ceil(max_time * fs) + 2*h_len;
        
        % Initialize discrete-time signal (delta functions for symbols)
        tx_signal_upsampled = zeros(1, signal_len);
        
        % Place symbols at their respective positions
        for i = 1:full_len
            pos = round(time_positions(i) * fs) + 1;
            if pos <= signal_len
                tx_signal_upsampled(pos) = tx_symbols(i);
            end
        end
        
        % Apply transmit pulse shaping
        tx_signal = conv(tx_signal_upsampled, h);
        
        % Add AWGN
        noise = 0;%sigma * randn(size(tx_signal));
        rx_signal = tx_signal + noise;
        
        % Apply matched filter
        rx_matched = conv(rx_signal, h);
        
        % Run M-BCJR detection with proper path-specific ISI handling
        [detected_bits, llrs] = adaptive_tau_mbcjr(rx_matched, tx_symbols, tau_set, ...
                                                  g, g_peak_idx, fs, L, M, N, sigma2);
        
        % Calculate bit errors (excluding preamble and tail)
        if length(detected_bits) >= N
            errors = sum(detected_bits(1:N) ~= msg_bits);
        else
            fprintf('Warning: Detected bits too short (%d < %d)\n', length(detected_bits), N);
            errors = N;  % Count all as errors if detection failed
        end
        
        num_errors = num_errors + errors;
        total_bits = total_bits + N;
        
        if mod(trial, 5) == 0
            fprintf('  Trial %d/%d - Errors: %d/%d\n', trial, num_trials, errors, N);
        end
    end
    
    % Calculate BER
    ber(snr_idx) = num_errors / total_bits;
    fprintf('SNR = %.1f dB, BER = %.6f\n', SNR_dB_val, ber(snr_idx));
end

elapsed_time = toc(sim_start_time);
fprintf('Simulation completed in %.2f seconds\n', elapsed_time);

% Plot BER results
figure;
semilogy(SNR_dB, ber, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
title(sprintf('BER Performance of Adaptive Tau FTN with M-BCJR (M=%d, L=%d)', M, L));
legend('Adaptive Tau M-BCJR');

function [detected_bits, llrs] = adaptive_tau_mbcjr(rx_signal, known_symbols, tau_set, ...
                                                  g, g_peak_idx, fs, L, M, N, sigma2)
    % Implements M-BCJR algorithm with proper path-specific tracking and ISI handling
    % rx_signal: Received signal after matched filtering
    % known_symbols: Known transmitted symbols (for debugging only)
    % tau_set: [tau_constructive, tau_destructive]
    % g: Overall pulse response
    % g_peak_idx: Peak location in pulse response
    % fs: Sampling frequency
    % L: Memory length
    % M: Number of survivor paths
    % N: Message length
    % sigma2: Noise variance
    
    % Total length including preamble and tail
    N_total = N + 2*L;
    
    % Number of states in trellis
    num_states = 2^L;
    
    % Define state structure - we'll store path information with each state
    % This is critical for proper ISI handling
    alpha_paths = cell(N_total+1, num_states);
    
    % Initialize path structure for each state
    for t = 1:N_total+1
        for s = 1:num_states
            alpha_paths{t,s} = struct('metric', -inf, ...
                                      'symbols', [], ...
                                      'tau_seq', [], ...
                                      'positions', []);
        end
    end
    
    % Initialize the first state (all zeros) with probability 1
    alpha_paths{1,1}.metric = 0;  % log domain, so 0 = probability 1
    alpha_paths{1,1}.symbols = zeros(1, L);  % Initial state (all zeros)
    alpha_paths{1,1}.tau_seq = [];  % No tau values yet
    alpha_paths{1,1}.positions = [0];  % Initial position at time 0
    
    % Beta metrics (backward recursion)
    beta = -inf(N_total+1, num_states);
    beta(N_total+1, 1) = 0;  % End in all-zeros state
    
    % =====================
    % Forward Recursion (α)
    % =====================
    for t = 1:N_total
        % Get active states (those with valid paths)
        active_states = [];
        for s = 1:num_states
            if alpha_paths{t,s}.metric > -inf
                active_states = [active_states, s];
            end
        end
        
        % Create metrics array for all potential new paths
        new_metrics = [];
        new_sources = [];
        new_inputs = [];
        new_targets = [];
        
        % For each active state, explore transitions
        for idx = 1:length(active_states)
            s = active_states(idx);
            
            % Get current path information
            curr_path = alpha_paths{t,s};
            
            % For each possible input bit (0/1)
            for input_bit = [0, 1]
                input_symbol = 1 - 2*input_bit;  % Convert to BPSK
                
                % Calculate next state
                if L == 0
                    next_state = 1;
                else
                    state_bits = de2bi(s-1, L, 'left-msb');
                    next_state_bits = [input_bit, state_bits(1:end-1)];
                    next_state = bi2de(next_state_bits, 'left-msb') + 1;
                end
                
                % Determine tau for this transition based on symbol signs
                if isempty(curr_path.symbols)
                    prev_symbol = 1;  % Default for initial state
                else
                    prev_symbol = 1 - 2*curr_path.symbols(1);  % First symbol in state
                end
                
                if prev_symbol * input_symbol > 0  % Same sign
                    curr_tau = tau_set(1);  % Constructive ISI
                else  % Different sign
                    curr_tau = tau_set(2);  % Destructive ISI
                end
                
                % Calculate new position
                if isempty(curr_path.positions)
                    new_pos = 0;  % Shouldn't happen with proper initialization
                else
                    new_pos = curr_path.positions(end) + curr_tau;
                end
                
                % Calculate expected received signal and ISI cancellation
                % ===== THIS IS THE KEY PART FOR ISI HANDLING =====
                
                % Get the symbol history including new input symbol
                symbol_history = [input_symbol, curr_path.symbols];
                
                % Get the tau sequence including new tau
                tau_history = [curr_path.tau_seq, curr_tau];
                
                % Calculate position in samples
                sample_pos = round(new_pos * fs) + g_peak_idx;
                
                % Skip if sample position is out of range
                if sample_pos > length(rx_signal)
                    continue;
                end
                
                % Calculate expected signal at this position
                % This properly reconstructs the ISI
                expected_signal = calculate_expected_signal(symbol_history, tau_history, g, fs);
                
                % Sample received signal at the specified position
                actual_sample = rx_signal(sample_pos);
                
                % Calculate branch metric (log-likelihood)
                branch_metric = -1/(2*sigma2) * (actual_sample - expected_signal)^2;
                
                % Calculate new path metric
                new_metric = curr_path.metric + branch_metric;
                
                % Store potential new path information
                new_metrics = [new_metrics, new_metric];
                new_sources = [new_sources, s];
                new_inputs = [new_inputs, input_bit];
                new_targets = [new_targets, next_state];
            end
        end
        
        % Apply M-algorithm: select M best paths
        % Create a temporary structure for all possible new paths
        all_new_paths = struct('metric', num2cell(new_metrics), ...
                              'source', num2cell(new_sources), ...
                              'input', num2cell(new_inputs), ...
                              'target', num2cell(new_targets));
        
        % If we have new paths to process
        if ~isempty(all_new_paths)
            % Convert to array for easier sorting
            metrics_array = [all_new_paths.metric];
            
            % Sort by metric (descending)
            [~, sort_idx] = sort(metrics_array, 'descend');
            
            % Keep only the best M paths
            keep_count = min(M, length(sort_idx));
            
            % Initialize temporary storage for next alpha paths
            next_alpha_paths = cell(1, num_states);
            for s = 1:num_states
                next_alpha_paths{s} = struct('metric', -inf, ...
                                           'symbols', [], ...
                                           'tau_seq', [], ...
                                           'positions', []);
            end
            
            % Process kept paths
            for i = 1:keep_count
                idx = sort_idx(i);
                
                source_state = all_new_paths(idx).source;
                input_bit = all_new_paths(idx).input;
                target_state = all_new_paths(idx).target;
                new_metric = all_new_paths(idx).metric;
                
                % Get source path information
                source_path = alpha_paths{t, source_state};
                
                % Determine tau for this transition
                input_symbol = 1 - 2*input_bit;
                
                if isempty(source_path.symbols)
                    prev_symbol = 1;
                else
                    prev_symbol = 1 - 2*source_path.symbols(1);
                end
                
                if prev_symbol * input_symbol > 0  % Same sign
                    curr_tau = tau_set(1);
                else  % Different sign
                    curr_tau = tau_set(2);
                end
                
                % Calculate new position
                if isempty(source_path.positions)
                    new_pos = 0;
                else
                    new_pos = source_path.positions(end) + curr_tau;
                end
                
                % Only update if this path is better than existing one
                if new_metric > next_alpha_paths{target_state}.metric
                    % Update symbols (new input symbol becomes first in state)
                    if L == 0
                        new_symbols = [];
                    else
                        new_symbols = [input_bit, source_path.symbols(1:L-1)];
                    end
                    
                    % Update target path
                    next_alpha_paths{target_state}.metric = new_metric;
                    next_alpha_paths{target_state}.symbols = new_symbols;
                    next_alpha_paths{target_state}.tau_seq = [source_path.tau_seq, curr_tau];
                    next_alpha_paths{target_state}.positions = [source_path.positions, new_pos];
                end
            end
            
            % Update alpha paths
            for s = 1:num_states
                alpha_paths{t+1,s} = next_alpha_paths{s};
            end
        end
        
        % Normalize to prevent numerical issues
        max_metric = -inf;
        for s = 1:num_states
            max_metric = max(max_metric, alpha_paths{t+1,s}.metric);
        end
        
        if max_metric > -inf
            for s = 1:num_states
                if alpha_paths{t+1,s}.metric > -inf
                    alpha_paths{t+1,s}.metric = alpha_paths{t+1,s}.metric - max_metric;
                end
            end
        end
    end
    
    % =====================
    % Backward Recursion (β)
    % =====================
    
    % Initialize beta values for the final state
    beta(N_total+1, :) = -inf;
    beta(N_total+1, 1) = 0;  % End in all-zeros state
    
    % Process backward recursion
    for t = N_total:-1:1
        % For each next state
        for next_s = 1:num_states
            if beta(t+1, next_s) == -inf
                continue;  % Skip unreachable states
            end
            
            % Get next state bits
            next_bits = de2bi(next_s-1, L, 'left-msb');
            
            % For each possible previous state
            for prev_s = 1:num_states
                % Get previous state bits
                prev_bits = de2bi(prev_s-1, L, 'left-msb');
                
                % For each input bit that could transition from prev_s to next_s
                for input_bit = [0, 1]
                    % Calculate state transition
                    if L == 0
                        is_valid = true;
                    else
                        % Check if this input bit would create the right transition
                        test_bits = [input_bit, prev_bits(1:end-1)];
                        is_valid = all(test_bits == next_bits);
                    end
                    
                    % Only process valid transitions
                    if is_valid
                        % Skip if alpha path isn't valid
                        if alpha_paths{t,prev_s}.metric == -inf
                            continue;
                        end
                        
                        % Get source path information
                        source_path = alpha_paths{t, prev_s};
                        
                        % Determine tau for this transition
                        input_symbol = 1 - 2*input_bit;
                        
                        if isempty(source_path.symbols)
                            prev_symbol = 1;
                        else
                            prev_symbol = 1 - 2*source_path.symbols(1);
                        end
                        
                        if prev_symbol * input_symbol > 0  % Same sign
                            curr_tau = tau_set(1);
                        else  % Different sign
                            curr_tau = tau_set(2);
                        end
                        
                        % Calculate position for this transition
                        if isempty(source_path.positions)
                            new_pos = 0;
                        else
                            new_pos = source_path.positions(end) + curr_tau;
                        end
                        
                        % Calculate sample position
                        sample_pos = round(new_pos * fs) + g_peak_idx;
                        
                        % Skip if sample position is out of range
                        if sample_pos > length(rx_signal)
                            continue;
                        end
                        
                        % Calculate expected signal at this position
                        symbol_history = [input_symbol, source_path.symbols];
                        tau_history = [source_path.tau_seq, curr_tau];
                        expected_signal = calculate_expected_signal(symbol_history, tau_history, g, fs);
                        
                        % Sample received signal
                        actual_sample = rx_signal(sample_pos);
                        
                        % Calculate branch metric
                        branch_metric = -1/(2*sigma2) * (actual_sample - expected_signal)^2;
                        
                        % Update beta value
                        beta(t, prev_s) = max(beta(t, prev_s), branch_metric + beta(t+1, next_s));
                    end
                end
            end
        end
        
        % Normalize to prevent numerical issues
        max_beta = max(beta(t, :));
        if max_beta > -inf
            beta(t, :) = beta(t, :) - max_beta;
        end
    end
    
    % =====================
    % LLR Calculation
    % =====================
    llrs = zeros(1, N_total);
    
    for t = 1:N_total
        % LLR calculation for symbol at time t+1 (after state t)
        llr_0 = -inf;  % For bit 0
        llr_1 = -inf;  % For bit 1
        
        % For each current state
        for prev_s = 1:num_states
            % Skip if alpha path isn't valid
            if alpha_paths{t,prev_s}.metric == -inf
                continue;
            end
            
            % For each possible input bit
            for input_bit = [0, 1]
                % Calculate next state
                if L == 0
                    next_s = 1;
                else
                    prev_bits = de2bi(prev_s-1, L, 'left-msb');
                    next_bits = [input_bit, prev_bits(1:end-1)];
                    next_s = bi2de(next_bits, 'left-msb') + 1;
                end
                
                % Skip if beta value isn't valid
                if beta(t+1, next_s) == -inf
                    continue;
                end
                
                % Get source path information
                source_path = alpha_paths{t, prev_s};
                
                % Determine tau for this transition
                input_symbol = 1 - 2*input_bit;
                
                if isempty(source_path.symbols)
                    prev_symbol = 1;
                else
                    prev_symbol = 1 - 2*source_path.symbols(1);
                end
                
                if prev_symbol * input_symbol > 0  % Same sign
                    curr_tau = tau_set(1);
                else  % Different sign
                    curr_tau = tau_set(2);
                end
                
                % Calculate position for this transition
                if isempty(source_path.positions)
                    new_pos = 0;
                else
                    new_pos = source_path.positions(end) + curr_tau;
                end
                
                % Calculate sample position
                sample_pos = round(new_pos * fs) + g_peak_idx;
                
                % Skip if sample position is out of range
                if sample_pos > length(rx_signal)
                    continue;
                end
                
                % Calculate expected signal at this position
                symbol_history = [input_symbol, source_path.symbols];
                tau_history = [source_path.tau_seq, curr_tau];
                expected_signal = calculate_expected_signal(symbol_history, tau_history, g, fs);
                
                % Sample received signal
                actual_sample = rx_signal(sample_pos);
                
                % Calculate branch metric
                branch_metric = -1/(2*sigma2) * (actual_sample - expected_signal)^2;
                
                % Combine alpha, branch metric, and beta for this path
                path_metric = alpha_paths{t,prev_s}.metric + branch_metric + beta(t+1, next_s);
                
                % Update LLR for this bit value
                if input_bit == 0
                    llr_0 = max(llr_0, path_metric);
                else
                    llr_1 = max(llr_1, path_metric);
                end
            end
        end
        
        % Calculate final LLR
        if isinf(llr_0) && isinf(llr_1)
            llrs(t) = 0;  % Undecidable
        elseif isinf(llr_0)
            llrs(t) = -100;  % Strong belief in bit 1
        elseif isinf(llr_1)
            llrs(t) = 100;   % Strong belief in bit 0
        else
            llrs(t) = llr_0 - llr_1;
        end
    end
    
    % Hard decisions based on LLRs
    detected_bits = (llrs < 0);
    
    % Remove preamble and tail bits
    if length(detected_bits) > N
        detected_bits = detected_bits(L+1:L+N);
        llrs = llrs(L+1:L+N);
    end
end

function expected = calculate_expected_signal(symbols, tau_sequence, g, fs)
    % Calculates the expected received signal at the current sample point
    % by properly modeling the ISI from all symbols in the history
    
    % Here we're explicitly calculating the ISI at the sampling point
    % This is where ISI calculation and cancellation truly happens
    
    % Get the number of symbols in history
    num_symbols = length(symbols);
    
    % Initialize expected signal value
    expected = 0;
    
    % Contribution from the current symbol (the first in the list)
    expected = expected + symbols(1) * g(ceil(length(g)/2));
    
    % Add ISI contributions from previous symbols
    % This is the key part where we account for all ISI
    if num_symbols > 1
        % Calculate the positions of previous symbols relative to current
        rel_positions = zeros(1, num_symbols-1);
        cumulative_tau = 0;
        
        for i = 1:length(tau_sequence)
            cumulative_tau = cumulative_tau + tau_sequence(i);
            rel_positions(i) = cumulative_tau * fs;
        end
        
        % Add ISI contribution from each previous symbol
        for i = 2:num_symbols
            % Skip if we don't have tau information (shouldn't happen)
            if i-1 > length(rel_positions)
                continue;
            end
            
            % Calculate pulse index for this symbol's contribution
            pulse_idx = ceil(length(g)/2) - round(rel_positions(i-1));
            
            % Add contribution if within pulse range
            if pulse_idx >= 1 && pulse_idx <= length(g)
                expected = expected + symbols(i) * g(pulse_idx);
            end
        end
    end
end