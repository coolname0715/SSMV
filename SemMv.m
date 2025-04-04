function [ v, W, YY ] = SemMv( X, Ymark, opt )
flagL = 'g';
[m,n] = size(Ymark);
switch flagL
    case 'g'
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        S = constructW_xf(Ymark,options);
        S = max(S,S');
        L = diag(sum(S,2)) - S;
    case 'h' 
        Weight = ones(n,1);
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.WeightMode = 'HeatKernel';
        options.t = 1;
        options.bSelfConnected = 1;
        W = constructW_xf(X,options);
        
        Dv = diag(sum(W)'); 
        De = diag(sum(W,2)); 
        invDe = inv(De);
        DV2 = full(Dv)^(-0.5);
        L = eye(n) - DV2 * W * diag(Weight) * invDe * W' *DV2; 
    
        
end





zero_counts = sum(Ymark == 0, 1);
num_cols_to_zero = round(0.2 * n);
[~, sorted_indices] = sort(zero_counts, 'descend');
cols_to_zero = sorted_indices(1:num_cols_to_zero);
Ymark(:, cols_to_zero) = 0;



idx1 = find(sum(Ymark,1)==0);
idx2 = find(sum(Ymark,1)~=0);

lambda1 = opt.lambda1;
lambda2 = opt.lambda2;
lambda3 = opt.lambda3;
num_class = opt.num_class;

flagL = 'g';
Weight = ones(n,1);
islocal = 0;
flagP = 1;   






[n,d] =  size(X);



v0    = zeros(m,1);
idx   = randperm(m);
v0(idx(1:ceil(m/2)))=1;
Q = v0.*Ymark;
W = rand(m,d);
nt = 10;
G = v0.*W;
    for i =1:m
        dn(i) = sqrt(sum((sum(W.*W,2)+eps)))./sum(W(i,:));
    end
    F = diag(dn);
res = abs(sum(Ymark - W*X',2).^2);


L_med    = median(res);

param.type = 'linear';
type=param.type;
switch param.type
    case 'hard'
        K = L_med;
    case 'linear'
        K = L_med;
    case 'log'
        K = L_med;
    case 'mix'
        param.gamma = 2*L_med;
        K           = 1/param.gamma;
    case 'mix_var'
        param.gamma = 2*sqrt(L_med);
        K           = 1/param.gamma;
end

t    = 1;
obji = 1;
while 1
    v = eval_spreg(res, K, param);

Q = v0.*Ymark;
G = v0.*W;
%% 求解W
detfW = -2*Q*X + 2*G*X'*X;
U = W -1/nt.*detfW;
for i2 = 1:m
    if norm(U(i2,:), 'fro') > lambda1/nt
    W(i2,:) = (1-lambda1/(nt*norm(U(i2,:), 'fro'))).*U(i2,:);
    
    elseif norm(U(i2,:), 'fro') <= lambda1/nt
       W(i2,:) = zeros(1,d); 
    end
end
%% 


res = abs(sum(Ymark - W*X',2).^2);
    if strcmp(type,'hard')||strcmp(type,'linear')||strcmp(type,'log')
        K       =  K/0.65;
    else
        K       =  K/1.1;
    end
    
 %% 求解Y
 A = diag(sqrt(v));
  Yu = pinv(A'*A +lambda2* L)*(A'*A*W*X(idx1,:)');
    Yu_flat = Yu(:);
    [sorted_values, sorted_indices] = sort(Yu_flat, 'descend');
    num_elements = numel(Yu_flat);
    class_size = ceil(num_elements / num_class);
    class_labels = zeros(size(Yu_flat));

    for ii = 1:num_elements
        class_labels(sorted_indices(ii)) = ceil(ii / class_size);
    end

    Yu = reshape(class_labels, size(Yu));

Ymark(:, idx1) = Yu;
YY=Ymark;
 
 
    

    t = t+1;
     if  t == 3,    break,     end
end





end

