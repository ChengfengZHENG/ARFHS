clear; clc;
[current_dataset_data, current_dataset_head, xlsx_file_name] = read_dataset('/dataset/'); % 读入数据集

% 示例数据
X = current_dataset_data(:, 2:end); % 特征
y = current_dataset_data(:, 1); % 类别标签
classes=unique(y);

% 划分训练集和验证集
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_val = X(test(cv), :);
y_val = y(test(cv), :);

% 使用随机森林进行特征选择
t = templateTree('MaxNumSplits', 5);
model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'Learners', t, 'NumLearningCycles', 50);

% 获取特征重要性评分
importance = predictorImportance(model);

% 选择最重要的前 5 个特征
[~, top_features] = maxk(importance, 5);
X_train_selected = X_train(:, top_features);
X_val_selected = X_val(:, top_features);

% AdaBoost分类器
if length(classes)<=2
    adaMdl = fitcensemble(X_train_selected, y_train, 'Method', 'AdaBoostM1', 'Learners', t, 'NumLearningCycles', 50);
else
    adaMdl = fitcensemble(X_train_selected, y_train, 'Method', 'AdaBoostM2', 'Learners', t, 'NumLearningCycles', 50);
end
y_pred = predict(adaMdl, X_val_selected);

% 评估模型性能
confMat = confusionmat(y_val, y_pred);

% 计算准确率
accuracy = sum(y_pred == y_val) / length(y_val);

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
disp('Selected Features using Tree-based Feature Selection:');
disp(X_train_selected);
disp(['Accuracy: ', num2str(accuracy)]);
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
temp = ['基于随机树的特征选择与AdaBoost', xlsx_file_name(1:end-5), num2str(month(now)), num2str(day(now)), num2str(hour(now)), num2str(minute(now)), num2str(second(now)), '.mat'];
save(temp);
