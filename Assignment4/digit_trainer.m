function [v_digit1, v_digit2, threshold, u, s, v, w, sort_digit1, sort_digit2] = digit_trainer(digit1_train, digit2_train)
    feature = 50;
    data = [digit1_train digit2_train];
    [u,s,v] = svd(data, 'econ');
    u = u(:,1:feature);
    n1 = size(digit1_train, 2);
    n2 = size(digit2_train, 2);
    digits = s * v';

    digit1 = digits(1:feature,1:n1);
    digit2 = digits(1:feature,n1+1:n1+n2);
    
    m1 = mean(digit1,2);
    m2 = mean(digit2,2);

    Sw = 0; % within class variances
    for k = 1:n1
        Sw = Sw + (digit1(:,k) - m1) * (digit1(:,k) - m1)';
    end

    for k = 1:n2
        Sw =  Sw + (digit2(:,k) - m2) * (digit2(:,k) - m2)';
    end

    Sb = (m1-m2)*(m2-m1)'; % between class

    [V2, D] = eig(Sb,Sw); % linear disciminant analysis
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);

    v_digit1 = w'*digit1;
    v_digit2 = w'*digit2;

    if mean(v_digit1) > mean(v_digit2)
        w = -w;
        v_digit1 = -v_digit1;
        v_digit2 = -v_digit2;
    end

    sort_digit1 = sort(v_digit1);
    sort_digit2 = sort(v_digit2);
    t1 = length(sort_digit1);
    t2 = 1;
    
    while sort_digit1(t1) > sort_digit2(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    threshold = (sort_digit1(t1) + sort_digit2(t2)) / 2;
end

