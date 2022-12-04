function [C, projection_error] = RCCS(A, scores, j)
   n = size(A, 1);
   C = zeros(n, j);
   for l=1:j
       p = sampling(scores);  % sample from distribution
       C(:, l) = A(:, p) / sqrt(j * scores(p));  % take column and rescale to have unbiased estimate
   end
   [Q, ~] = qr(C, 0);  % compute reduced QR of C
   projection_error = norm(A-Q*Q'*A, 2);  % projection error
end

