function Tau = greedy_rank_approximation(A, r)
    [U, ~, V] = svd(A);
    I = greedy_row_selection(U(:, 1:r), r);
    J = greedy_row_selection(V(:, 1:r), r);
    Tau = A(:, J) / A(I, J) * A(I, :);
end

