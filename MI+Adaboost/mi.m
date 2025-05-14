function I = mi(X, Y)
    % 计算 X 和 Y 之间的互信息
    % X: 特征向量
    % Y: 类别标签
    
    % 将 X 和 Y 转换为列向量
    X = X(:);
    Y = Y(:);
    
    % 获取唯一值和类别数
    unique_X = unique(X);
    unique_Y = unique(Y);
    num_X = length(unique_X);
    num_Y = length(unique_Y);
    
    % 计算联合概率分布
    joint_prob = histcounts2(X, Y, [num_X, num_Y], 'Normalization', 'probability');
    
    % 计算边缘概率分布
    pX = sum(joint_prob, 2);
    pY = sum(joint_prob, 1);
    
    % 创建网格以匹配联合概率分布的大小
    [pX_grid, pY_grid] = meshgrid(pY, pX);
    
    % 计算互信息
    non_zero_indices = joint_prob > 0;
    I = sum(joint_prob(non_zero_indices) .* log2(joint_prob(non_zero_indices) ./ (pX_grid(non_zero_indices) .* pY_grid(non_zero_indices))));
end
