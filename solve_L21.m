function [W] = solve_L21(X, X_s, lambda)
% Solve L2,1-norm for hyperspectral data.
% Objective function:
%    || X - X_s * W ||_{2, 1} + lambda * || W ||_{2, 1}
% Usage
%   [W] = solve_L21(X, X_s, lambda)

MAX_ITER = 30;
CONV_THRES = 1e-4;

[p, n] = size(X);
[~, s] = size(X_s);
m = p + s;
V = [x_s lambda*ones(p)];
D = ones(m);
U = zeros(m, n);
for t = 1:MAX_ITER
    D_inv = pinv(D);
    Ut = D_inv*V'*pinv(V*D_inv*V')*y;
    conv = norm(Ut-U, 2);
    if conv < CONV_THRES 
        break
    end
    for i = 1:m
        D(i, i) = 1/(2*norm(Ut(i, :), 2));
    end
    U = Ut;
end
W = U(1:s, :);
end
