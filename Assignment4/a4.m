%%
% clear workspace
close all; clear all; clc

% Reshape data (each column is a picture)
[trainingdata, traingnd] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
trainingdata = im2double(reshape(trainingdata, size(trainingdata,1) * size(trainingdata,2), []));
traingnd = double(traingnd);

[testdata, testgnd] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testdata = im2double(reshape(testdata, size(testdata,1) * size(testdata,2), []));
testgnd = double(testgnd);

% subtract mean
mn =  mean(trainingdata,2);
trainingdata = trainingdata - repmat(mn,1,60000);

testdata = testdata - repmat(mn,1,10000);
%%
[u,s,v] = svd(trainingdata, 'econ');
lambda = diag(s).^2;

figure(1)
plot(diag(s),'ko','Linewidth',2)
title("singular value diagram")
set(gca,'Fontsize',16,'Xlim',[0 500])
figure(2)
plot(1:784, lambda/sum(lambda), 'mo', 'Linewidth', 2);
title("Energy of each Diagonal Variance");
xlabel("Diagonal Variances"); ylabel("Energy Captured");

%%
proj = u(:,[2,3,5])'* trainingdata;

proj_0 = proj(:,traingnd == 0);
proj_1 = proj(:,traingnd == 1);
proj_2 = proj(:,traingnd == 2);
proj_3 = proj(:,traingnd == 3);
proj_4 = proj(:,traingnd == 4);
proj_5 = proj(:,traingnd == 5);
proj_6 = proj(:,traingnd == 6);
proj_7 = proj(:,traingnd == 7);
proj_8 = proj(:,traingnd == 8);
proj_9 = proj(:,traingnd == 9);

figure(3)
plot3(proj_0(1,:),proj_0(2,:),proj_0(3,:),"o");hold on
plot3(proj_1(1,:),proj_1(2,:),proj_1(3,:),"o");hold on
plot3(proj_2(1,:),proj_2(2,:),proj_2(3,:),"o");hold on
plot3(proj_3(1,:),proj_3(2,:),proj_3(3,:),"o");hold on
plot3(proj_4(1,:),proj_4(2,:),proj_4(3,:),"o");hold on
plot3(proj_5(1,:),proj_5(2,:),proj_5(3,:),"o");hold on
plot3(proj_6(1,:),proj_6(2,:),proj_6(3,:),"o");hold on
plot3(proj_7(1,:),proj_7(2,:),proj_7(3,:),"o");hold on
plot3(proj_8(1,:),proj_8(2,:),proj_8(3,:),"o");hold on
plot3(proj_9(1,:),proj_9(2,:),proj_9(3,:),"o");

title('3D projection selected V-modes')
legend('0','1','2','3','4','5','6','7','8','9');
%% pick two digits and build LDA
% selected_digits = randperm(10, 2) - 1;
highest_digit1 = 0;
highest_digit2 = 0;
highest_succRate = 0;

lowest_digit1 = 0;
lowest_digit2 = 0;
lowest_succRate = 2;

% try all combinations
for i = 0:8
    for j = i + 1:9
        selected_digits = [i j];
        digit1_indices = find(traingnd == selected_digits(1));
        digit1_train = trainingdata(:,digit1_indices');

        digit2_indices = find(traingnd == selected_digits(2));
        digit2_train = trainingdata(:,digit2_indices');

        [v_digit1, v_digit2, threshold, u, s, v, w, sort_digit1, sort_digit2] = digit_trainer(digit1_train, digit2_train);
        
        digit1_test_indices = find(testgnd == selected_digits(1));
        digit1_test = testdata(:,digit1_test_indices');
        n1_test = size(digit1_test, 2);
        digit2_test_indices = find(testgnd == selected_digits(2));
        digit2_test = testdata(:,digit2_test_indices');
        n2_test = size(digit2_test, 2);

        digits_test = [digit1_test digit2_test];
        digits_train = [digit1_train digit2_train];
        n1_train = size(digit1_train, 2);
        n2_train = size(digit2_train, 2);
        hiddenlabels = zeros(1, size(digits_train, 2));
        hiddenlabels(n1_train + 1:n1_train + n2_train) = 1;
        testNum = size(digits_train,2);
        testMat = u' * digits_train;
        pval = w' * testMat;

        % digit2 = 1, digit1 = 0
        ResVec = (pval > threshold);
        err = abs(ResVec - hiddenlabels);
        errNum = sum(err);
        sucRate = 1 - errNum/testNum;
        
        if sucRate > highest_succRate
            highest_succRate = sucRate;
            highest_digit1 = i;
            highest_digit2 = j;
        end
        
        if sucRate < lowest_succRate
            lowest_succRate = sucRate;
            lowest_digit1 = i;
            lowest_digit2 = j;
        end
    end
end

figure(3)
plot(v_digit1, zeros(length(v_digit1)),'ob','Linewidth',2)
hold on
plot(v_digit2, ones(length(v_digit2)),'dr','Linewidth',2)
ylim([0 1.2])
title("Overlap of two digits: 8,9")

figure(5)
subplot(1,2,1)
histogram(sort_digit1,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-10 10],'Ylim',[0 1000],'Fontsize',14)
title('the first digit')
subplot(1,2,2)
histogram(sort_digit2,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-10 10],'Ylim',[0 1000],'Fontsize',14)
title('the second digit')

%% Check accuracy for two digits on test set
digit1_test_indices = find(testgnd == selected_digits(1));
digit1_test = testdata(:,digit1_test_indices');
n1_test = size(digit1_test, 2);
digit2_test_indices = find(testgnd == selected_digits(2));
digit2_test = testdata(:,digit2_test_indices');
n2_test = size(digit2_test, 2);

digits_test = [digit1_test digit2_test];
hiddenlabels = zeros(1, size(digits_test, 2));
hiddenlabels(n1_test + 1:n1_test + n2_test) = 1;
testNum = size(digits_test,2);
testMat = u' * digits_test;
pval = w' * testMat;

% digit2 = 1, digit1 = 0
ResVec = (pval > threshold);
err = abs(ResVec - hiddenlabels);
errNum = sum(err);
sucRate = 1 - errNum/testNum;

%% Pick three digits
feature = 50;
% selected_digits = randperm(10, 3) - 1;
selected_digits = [1 2 3];
digit1_indices = find(traingnd == selected_digits(1));
digit1_train = trainingdata(:,digit1_indices');

digit2_indices = find(traingnd == selected_digits(2));
digit2_train = trainingdata(:,digit2_indices');

digit3_indices = find(traingnd == selected_digits(3));
digit3_train = trainingdata(:,digit3_indices');

data = [digit1_train digit2_train digit3_train];
[u,s,v] = svd(data, 'econ');

n1 = size(digit1_train, 2);
n2 = size(digit2_train, 2);
n3 = size(digit3_train, 2);

digits = s * v';

digit1 = digits(1:feature,1:n1);
digit2 = digits(1:feature,n1+1:n1+n2);
digit3 = digits(1:feature,n1+n2+1:n1+n2+n3);

%% LDA for three digits
m1 = mean(digit1, 2);
m2 = mean(digit2, 2);
m3 = mean(digit3, 2);
m = [m1 m2 m3];
overall_m = (m1 + m2 + m3) / 3;

Sw = 0; % within class variances
for k = 1:n1
    Sw = Sw + (digit1(:,k) - m1) * (digit1(:,k) - m1)';
end

for k = 1:n2
    Sw = Sw + (digit2(:,k) - m2) * (digit2(:,k) - m2)';
end

for k = 1:n3
    Sw = Sw + (digit3(:,k) - m3) * (digit3(:,k) - m3)';
end

Sb = 0; % between class
for i = 1:3
    Sb = Sb + (m(:, i) - overall_m) * (m(:, i) - overall_m)'; 
end

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

v_digit1 = w'*digit1;
v_digit2 = w'*digit2;
v_digit3 = w'*digit3;

if mean(v_digit1) > mean(v_digit2) 
    w = -w;
    v_digit1 = -v_digit1;
    v_digit2 = -v_digit2;
end

if mean(v_digit2) > mean(v_digit3)
    w = -w;
    v_digit2 = -v_digit2;
    v_digit3 = -v_digit3;
end

figure(3)
plot(v_digit1, zeros(length(v_digit1)),'ob','Linewidth',2)
hold on
plot(v_digit2, ones(length(v_digit2)),'dr','Linewidth',2)
hold on
plot(v_digit3, ones(length(v_digit3)) + 1,'g','Linewidth',2)
ylim([0 3])
title("Overlap of three digits: 1, 2 ,3")
%% Classification
sort_digit1 = sort(v_digit1);
sort_digit2 = sort(v_digit2);
sort_digit3 = sort(v_digit3);

% Compare between digit 1 and digit 2
t1 = length(sort_digit1);
t2 = 1;
while sort_digit1(t1) > sort_digit2(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort_digit1(t1) + sort_digit2(t2)) / 2;

figure(4)
subplot(1,2,1)
histogram(sort_digit1,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-8 4],'Ylim',[0 1000],'Fontsize',14)
title('the first digit')
subplot(1,2,2)
histogram(sort_digit2,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-8 4],'Ylim',[0 1000],'Fontsize',14)
title('the second digit')

% Compare between digit 2 and digit 3
t2 = length(sort_digit2);
t3 = 1;
while sort_digit2(t2) > sort_digit3(t3)
    t2 = t2 - 1;
    t3 = t3 + 1;
end
threshold = (sort_digit2(t2) + sort_digit3(t3)) / 2;

figure(5)
subplot(1,2,1)
histogram(sort_digit2,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-5 7],'Ylim',[0 1000],'Fontsize',14)
title('the second digit')
subplot(1,2,2)
histogram(sort_digit3,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-5 7],'Ylim',[0 1000],'Fontsize',14)
title('the third digit')

% Compare between digit 1 and digit 3
t1 = length(sort_digit1);
t3 = 1;
while sort_digit1(t1) > sort_digit3(t3)
    t1 = t1 - 1;
    t3 = t3 + 1;
end
threshold = (sort_digit1(t1) + sort_digit3(t3)) / 2;

figure(6)
subplot(1,2,1)
histogram(sort_digit1,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-6 7],'Ylim',[0 1000],'Fontsize',14)
title('the first digit')
subplot(1,2,2)
histogram(sort_digit3,30); hold on, plot([threshold threshold], [0 1000],'r')
set(gca,'Xlim',[-6 7],'Ylim',[0 1000],'Fontsize',14)
title('the second digit')

%% SVM and decision tree
tree = fitctree(trainingdata',traingnd, 'MaxNumSplits',10,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree, 'mode', 'individual');
[~, k] = min(classError);
testSetPredictions = predict(tree.Trained{k}, testdata');
err_tree = immse(testSetPredictions, testgnd);

diff = testSetPredictions - testgnd;
correct_pred = find(diff == 0);
sucRate_tree = length(correct_pred) / length(testgnd);

% SVM classifier with training data, labels and test set
Mdl = fitcecoc(trainingdata',traingnd);
testlabels = predict(Mdl,testdata');

diff_svm = testlabels - testgnd;
correct_pred = find(diff_svm == 0);
sucRate_svm = length(correct_pred) / length(testgnd);

%% Decision tree and svm on pairs of digits
easiest_digits = [7 9];

% Decision Tree
digit1_indices = find(traingnd == easiest_digits(1));
digit1_train = trainingdata(:,digit1_indices');
digit1_train_label = traingnd(digit1_indices);

digit2_indices = find(traingnd == easiest_digits(2));
digit2_train = trainingdata(:,digit2_indices');
digit2_train_label = traingnd(digit2_indices);

easiest_data = [digit1_train digit2_train];

easiest_test_digit1_ind = find(testgnd == easiest_digits(1));
digit1_test = testdata(:,easiest_test_digit1_ind');

easiest_test_digit2_ind = find(testgnd == easiest_digits(2));
digit2_test = testdata(:,easiest_test_digit2_ind');

digits_test = [digit1_test digit2_test];

tree = fitctree(easiest_data',[digit1_train_label; digit2_train_label], 'MaxNumSplits',10,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree, 'mode', 'individual');
[~, k] = min(classError);
testSetPredictions = predict(tree.Trained{k}, easiest_data');

train_label = [digit1_train_label; digit2_train_label];
easiest_test_label = [testgnd(easiest_test_digit1_ind); testgnd(easiest_test_digit2_ind)];

diff = testSetPredictions - train_label;
correct_pred = find(diff == 0);
sucRate_tree = length(correct_pred) / length(testSetPredictions);

% SVM
Mdl = fitcsvm(easiest_data',[digit1_train_label; digit2_train_label]);
testlabels = predict(Mdl,easiest_data');

diff_svm = testlabels - train_label;
correct_pred = find(diff_svm == 0);
sucRate_svm = length(correct_pred) / length(train_label);
