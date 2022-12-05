function k = sampling(p)
% SAMPLING returns 1 <= k <= length(p) with probability p(k)
%     p: probability distribution used for sampling

    P = cumsum(p);
    k = nnz(rand>P)+1;

end
