function I = greedy_row_selection(U, r)
    I = zeros(r, 1);
    for k=1:r
        [~, max_ind] = max(abs(U(:, k)));
        max_ind = max_ind(1);
        I(k) = max_ind;
        U = U - 1/(U(max_ind, k)) * U(:, k) * U(max_ind, :);
    end
end