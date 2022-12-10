function [C, projection_error] = RCCS(A, scores, j)
% RCCS  performs a random column subset selection of prescribed size
%       note that sampling is done independently and with replacement
%    A: original matrix
%    scores: probability distribution used for sampling
%    j: size of random column subset

   n = size(A, 1);
   C = zeros(n, j);
   for l=1:j
       p = sampling(scores);  % sample from distribution
       C(:, l) = A(:, p) / sqrt(j * scores(p));  % take column and rescale to have unbiased estimate
   end
   [Q, ~] = qr(C, 0);  % compute reduced QR of C
   projection_error = norm(A-Q*Q'*A, 2);  % projection error onto column space of C

end

