function [mean_orthogonal_errors, mean_oblique_errors] = compute_averages(A, sampling_probabilities, num_avg, ranks)
    orthogonal_errors = zeros(num_avg, ranks);
    oblique_errors = zeros(num_avg, ranks);
    % perform multiple runs
    for i=1:num_avg
        % ranks from 1, ..., ranks
        for j=1:ranks
            [orthogonal_error, oblique_error] = RCCS(A, sampling_probabilities, j);
            orthogonal_errors(i, j) = orthogonal_error;
            oblique_errors(i, j) = oblique_error;
        end
    end
    mean_orthogonal_errors = mean(orthogonal_errors);
    mean_oblique_errors = mean(oblique_errors);
end