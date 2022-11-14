function mean_errors = compute_averages(A, sampling_probabilities, num_avg, ranks)
    errors = zeros(num_avg, ranks);
    % perform multiple runs
    for i=1:num_avg
        % ranks from 1, ..., ranks
        for j=1:ranks
            errors(i, j) = RCCS(A, sampling_probabilities, j);
        end
    end
    mean_errors = mean(errors);
end