clear; clc;

% 固定随机种子以确保可重复性
rng(1);

% 定义参数
dimensions = [10, 50, 100]; % 低维、中维、高维
noise_levels = [0.1, 0.5, 1.0]; % 低、中、高噪声强度（标准差）
outlier_ratio = 0.05; % 离群值比例（5%）
num_samples = 1000; % 每组人工数据集的样本数
num_classes = 2; % 假设二分类，可根据需要调整为多分类

% 创建输出目录（确保保存路径存在）
output_dir = 'synthetic_datasets';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 生成人工数据集并保存为 CSV
for d = 1:length(dimensions)
    dim = dimensions(d);
    X_synthetic = randn(num_samples, dim);
    y_original = randi([0, 1], num_samples, 1); % 原始标签，1000 行
    
    for n = 1:length(noise_levels)
        noise_std = noise_levels(n);
        X_noisy = X_synthetic + noise_std * randn(size(X_synthetic)); % 添加噪声
        
        % 添加离群值
        num_outliers = round(outlier_ratio * num_samples);
        outlier_indices = randsample(num_samples, num_outliers);
        X_outliers = zeros(num_outliers, dim);
        y_outliers = zeros(num_outliers, 1);
        
        for k = 1:num_outliers
            i = outlier_indices(k);
            feature_idx = randi(dim);
            X_noisy(i, feature_idx) = X_noisy(i, feature_idx) * 10;
            X_outliers(k, :) = X_noisy(i, :);
            y_outliers(k) = y_original(i); % 使用原始标签
        end
        
        % 追加离群值
        X_noisy = [X_noisy; X_outliers]; % 现在 1050 行
        y_current = [y_original; y_outliers]; % 创建 y_current，1050 行
        
        % 调试输出
        fprintf('Dimension %d, Noise %.1f: X_noisy rows = %d, cols = %d; y_current rows = %d\n', ...
                dim, noise_std, size(X_noisy, 1), size(X_noisy, 2), size(y_current, 1));
        
        % 验证维度一致性
        if size(X_noisy, 1) ~= size(y_current, 1)
            error('Dimension mismatch: X_noisy has %d rows, y_current has %d rows', ...
                  size(X_noisy, 1), size(y_current, 1));
        end
        
        % 保存为 CSV
        dataset_name = fullfile(output_dir, sprintf('synthetic_dataset_dim%d_noise%.1f.csv', dim, noise_std));
        csv_data = [y_current, X_noisy];
        headers = [{'Label'}, arrayfun(@(x) sprintf('Feature%d', x), 1:dim, 'UniformOutput', false)];
        csv_cell = [headers; num2cell(csv_data)];
        writecell(csv_cell, dataset_name);
        disp(['Saved artificial dataset to: ', dataset_name]);
    end
end

