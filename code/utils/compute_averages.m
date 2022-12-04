function mean_projection_errors = compute_averages(A, sampling_probabilities, num_avg, ranks)
    orthogonal_errors = zeros(num_avg, ranks);
    % perform multiple runs
    for i=1:num_avg
        % ranks from 1, ..., ranks
        for j=1:ranks
            [~, projection_error] = RCCS(A, sampling_probabilities, j);
            orthogonal_errors(i, j) = projection_error;
        end
    end
    mean_projection_errors = mean(orthogonal_errors);
end