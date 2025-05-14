clear; clc;
[current_dataset_data, current_dataset_head, xlsx_file_name] = read_dataset('/dataset/'); % 读入数据集

% 示例数据
X = current_dataset_data(:, 2:end); % 特征
y = current_dataset_data(:, 1); % 类别标签
classes = unique(y);

% 随机打乱数据
rng(1); % 固定随机种子以确保可重复性
shuffled_indices = randperm(length(y));
X = X(shuffled_indices, :);
y = y(shuffled_indices);

% 划分数据集为训练集和验证集（60% 训练，40% 验证）
train_ratio = 0.6;
num_train = round(train_ratio * length(y));
X_train = X(1:num_train, :);
y_train = y(1:num_train);
X_val = X(num_train+1:end, :);
y_val = y(num_train+1:end);

% 使用卡方检验进行特征选择
num_features = size(X_train, 2);
chi2_values = zeros(num_features, 1);
for i = 1:num_features
    tbl = crosstab(X_train(:, i), y_train);
    chi2_values(i) = chi2test(tbl); % 计算卡方统计量
end
[~, sorted_idx] = sort(chi2_values, 'descend');
top_features = sorted_idx(1:5); % 选择最重要的前 5 个特征
X_train_selected = X_train(:, top_features);
X_val_selected = X_val(:, top_features);

% 训练 FKNN 模型
num_neighbors = 5; % 设定K值
Mdl = fitcknn(X_train_selected, y_train, 'NumNeighbors', num_neighbors, 'Distance', 'euclidean', 'Standardize', true);

% 在验证集上进行预测
y_val_pred = predict(Mdl, X_val_selected);

% 评估模型性能
confMat = confusionmat(y_val, y_val_pred);

% 计算准确率
accuracy = sum(y_val_pred == y_val) / length(y_val);

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
disp('Selected Features using Chi-Square Test:');
disp(top_features);
disp(['Accuracy using FKNN: ', num2str(accuracy)]);
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
temp = ['基于卡方检验特征选择与FKNN', xlsx_file_name(1:end-5), num2str(month(now)), num2str(day(now)), num2str(hour(now)), num2str(minute(now)), num2str(second(now)), '.mat'];
save(temp);

% 卡方检验函数
function chi2 = chi2test(tbl)
    % 计算卡方统计量
    % tbl: 交叉表
    [R, C] = size(tbl);
    total = sum(tbl(:));
    expected = sum(tbl, 2) * sum(tbl, 1) / total;
    chi2 = sum((tbl(:) - expected(:)).^2 ./ expected(:));
end
