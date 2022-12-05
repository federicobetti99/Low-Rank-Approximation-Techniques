function mean_projection_errors = compute_averages(A, sampling_probabilities, num_avg, ranks)
% COMPUTE_AVERAGES computes average projection errors
%     A: original matrix
%     sampling_probabilities: sampling probability used for sampling
%     num_avg: number of averages to be performed
%     ranks: sequence of sizes for the random column subset

    orthogonal_errors = zeros(num_avg, ranks);
    % perform multiple runs
    for i=1:num_avg
        % ranks from 1, ..., ranks
        for j=1:ranks
            [~, projection_error] = RCCS(A, sampling_probabilities, j); % compute a size j random column subset
            orthogonal_errors(i, j) = projection_error;  % save projection error
        end
    end
    mean_projection_errors = mean(orthogonal_errors);  % compute mean error for each rank between 1 and ranks

end