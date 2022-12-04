function Cu = halving(A, k)
    col = size(A, 2);
    Cu = A;
    Cu_tilda = Cu;
    while col > k*log(k) && col > 1
        col2 = fix(col/2);
        Cu_tilda = Cu(:, randsample(col, col2));
        Cu = Cu_tilda;
        col = col2;
    end
    [~, Sigma, ~] = svd(Cu_tilda, "econ");
    gold_standard = diag(Sigma);
    lambda = 1/sqrt(k) * sqrt(sum(gold_standard(k+1:end).^2));
    ridge_scores = diag(Cu'*(Cu_tilda * Cu_tilda' + lambda^2 * eye(size(Cu_tilda, 1)))*Cu);
    [Cu, ~] = RCCS(Cu, ridge_scores / sum(ridge_scores), fix(k*log(k)));
end
