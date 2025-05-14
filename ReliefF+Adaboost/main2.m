clear; clc;
[current_dataset_data, current_dataset_head, xlsx_file_name] = read_dataset('/dataset/'); % 读入数据集

% 示例数据
X = current_dataset_data(:, 2:end); % 100 个样本，10 个特征
y = current_dataset_data(:, 1); % 3 个类别

classes=unique(y);


% 使用 relieff 算法进行特征选择
[idx, weights] = relieff(X, y, 10);

% 选择最重要的前 5 个特征
num_features = 5;
top_features = idx(1:num_features);
X_selected = X(:, top_features);

% Adaboost分类器及交叉验证
t = templateTree('MaxNumSplits', 5);
k = 10; % 10 折交叉验证
cv_indices = crossvalind('Kfold', y, k);

y_pred = zeros(size(y));
for i = 1:k
    test_idx = (cv_indices == i);
    train_idx = ~test_idx;
    if length(classes)==2
        Mdl = fitcensemble(X_selected(train_idx, :), y(train_idx), 'Method', 'AdaBoostM1', 'Learners', t, 'NumLearningCycles', 50);
    else
        Mdl = fitcensemble(X_selected(train_idx, :), y(train_idx), 'Method', 'AdaBoostM2', 'Learners', t, 'NumLearningCycles', 50);
    end
    
    y_pred(test_idx) = predict(Mdl, X_selected(test_idx, :));
end

% 评估模型性能
confMat = confusionmat(y, y_pred);

% 计算准确率
accuracy = sum(y_pred == y) / length(y);

% 计算精确率、召回率和 F1 分数
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1 = 2 * (precision .* recall) ./ (precision + recall);

% 计算宏平均、微平均和加权平均
macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1 = mean(f1);

micro_precision = sum(diag(confMat)) / sum(confMat(:));
micro_recall = micro_precision; % 在多类分类问题中，微平均的精确率和召回率相同
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall);

weights = sum(confMat, 2) / sum(confMat(:));
weighted_precision = sum(precision .* weights);
weighted_recall = sum(recall .* weights);
weighted_f1 = sum(f1 .* weights);

% 显示结果
disp('Selected Features using relieff:');
disp(X_selected);
disp(['Accuracy using Adaboost: ', num2str(accuracy)]);
disp('Confusion Matrix:');
disp(confMat);
disp('Precision:');
disp(precision);
disp('Recall:');
disp(recall);
disp('F1 Score:');
disp(f1);
disp(['Macro Precision: ', num2str(macro_precision)]);
disp(['Macro Recall: ', num2str(macro_recall)]);
disp(['Macro F1: ', num2str(macro_f1)]);
disp(['Micro Precision: ', num2str(micro_precision)]);
disp(['Micro Recall: ', num2str(micro_recall)]);
disp(['Micro F1: ', num2str(micro_f1)]);
disp(['Weighted Precision: ', num2str(weighted_precision)]);
disp(['Weighted Recall: ', num2str(weighted_recall)]);
disp(['Weighted F1: ', num2str(weighted_f1)]);

% 保存结果
temp = ['基于Relieff的特征选择与Adaboost', xlsx_file_name(1:end-5),num2str(month(now)), num2str(day(now)), num2str(hour(now)), num2str(minute(now)), num2str(second(now)), '.mat'];
save(temp);
