function error = RCCS(A, scores, j)
   n = size(A, 1);
   C = zeros(n, j);
   for l=1:j
       p = sampling(scores);  % sample from distribution
       C(:, l) = A(:, p) / sqrt(j * scores(p));  % take column and rescale to have unbiased estimate
   end
   error = norm(A-C*pinv(C)*A, 2);
end

