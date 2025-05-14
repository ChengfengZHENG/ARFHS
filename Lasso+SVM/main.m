clear; clc;
[current_dataset_data, current_dataset_head, xlsx_file_name] = read_dataset('/dataset/'); % 读入数据集

% 示例数据
X = current_dataset_data(:, 2:end); % 100 个样本，10 个特征
y = current_dataset_data(:, 1); % 3 个类别

% 将类别标签转换为二进制矩阵
Y = dummyvar(y);

% 初始化一些变量
numClasses = size(Y, 2);
selectedFeatures = false(size(X, 2), numClasses);

% 使用一对多策略进行 Lasso 特征选择
for i = 1:numClasses
    [B, FitInfo] = lasso(X, Y(:, i), 'CV', 10); 
    selectedFeatures(:, i) = B(:, FitInfo.Index1SE) ~= 0;
end

% 选择非零系数的特征
X_selected = X(:, any(selectedFeatures, 2));

% SVM分类器及交叉验证
cvMdl = fitcecoc(X_selected, y, 'Learners', 'linear', 'CrossVal', 'on', 'KFold', 10);
y_pred = kfoldPredict(cvMdl);

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
disp('Selected Features using Lasso:');
disp(X_selected);
disp(['Accuracy using SVM: ', num2str(accuracy)]);
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
temp = ['基于Lasso的特征选择与SVM', xlsx_file_name(1:end-5), num2str(month(now)), num2str(day(now)), num2str(hour(now)), num2str(minute(now)), num2str(second(now)), '.mat'];
save(temp);
