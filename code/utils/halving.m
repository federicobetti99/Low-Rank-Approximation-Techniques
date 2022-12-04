function M = halving(A, k)
    n = size(A, 2);
    M = zeros(n, n);
    Ah = A;
    col = n;
    while col > k*log(k) && col > 1
       col2 = fix(col/2);
       M = Ah(:, randsample(col, col2)); 
       Ah = M;
       col = col2;
    end
end

