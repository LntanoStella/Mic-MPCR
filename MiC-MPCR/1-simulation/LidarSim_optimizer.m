function LidarAnalysis_Optimizer()
    % LidarAnalysis_Optimizer - 仿真数据后处理与布站优化专家系统
    clc; clear; close all;
    
    % =====================================================================
    % 1. 配置评价权重 (根据实际工程需求调整)
    % =====================================================================
    weights = struct('w_eff', 0.5, 'w_red', 0.3, 'w_base', 0.2);
    
    fprintf('======================================================\n');
    fprintf('   Lidar 布站优化分析系统 (Y轴高度适配版)\n');
    fprintf('======================================================\n');

    % =====================================================================
    % 2. 数据加载与预处理
    % =====================================================================
    [file, path] = uigetfile('*.csv', '选择仿真日志文件 (BatchLog_*.csv)');
    if isequal(file, 0), return; end
    
    fprintf('>> 正在加载数据: %s ... ', file);
    try
        data = readtable(fullfile(path, file));
        fprintf('成功! (共 %d 组方案)\n', height(data));
    catch ME
        fprintf('失败!\n错误信息: %s\n', ME.message); return;
    end
    
    % 检查必要字段
    reqVars = {'Off_X', 'Off_Y', 'Off_Z', 'Cov_Base', 'Cov_Eff', 'Redundancy'};
    if ~all(ismember(reqVars, data.Properties.VariableNames))
        error('CSV文件格式不匹配，缺少必要的列。');
    end

    % =====================================================================
    % 3. 多维评价算法 (TOPSIS & Pareto)
    % =====================================================================
    fprintf('>> 执行 TOPSIS 综合评分算法...\n');
    
    D =[data.Cov_Base, data.Cov_Eff, data.Redundancy];
    [m, ~] = size(D);
    w_vec =[weights.w_base, weights.w_eff, weights.w_red];
    
    norm_D = D ./ sqrt(sum(D.^2, 1));
    V = norm_D .* w_vec;
    
    Z_pos = max(V); 
    Z_neg = min(V);
    
    D_pos = sqrt(sum((V - Z_pos).^2, 2));
    D_neg = sqrt(sum((V - Z_neg).^2, 2));
    
    Scores = D_neg ./ (D_pos + D_neg);
    data.Score = Scores * 100; 
    
    % =====================================================================
    % 4. 帕累托前沿分析 (Pareto Frontier)
    % =====================================================================
    fprintf('>> 识别帕累托非支配解...\n');
    xy = [data.Cov_Eff, data.Redundancy];
    isPareto = false(m, 1);
    
    for i = 1:m
        dominated = false;
        for j = 1:m
            if i == j, continue; end
            if (xy(j,1) >= xy(i,1) && xy(j,2) >= xy(i,2)) && ...
               (xy(j,1) > xy(i,1) || xy(j,2) > xy(i,2))
                dominated = true;
                break;
            end
        end
        if ~dominated, isPareto(i) = true; end
    end
    data.IsPareto = isPareto;
    fprintf('   -> 发现 %d 个非支配最优解。\n', sum(isPareto));

    % =====================================================================
    % 5. 生成分析报告
    % =====================================================================
    sortedData = sortrows(data, 'Score', 'descend');
    top5 = sortedData(1:5, :);
    
    fprintf('\n=======================================================================\n');
    fprintf('   最优布站方案推荐 (Top 5)\n');
    fprintf('=======================================================================\n');
    fprintf('排名 | 评分 | Off_X | Off_Y(高) | Off_Z | 基础覆盖%% | 有效覆盖%% | 冗余度%% | 备注\n');
    fprintf('-----|------|-------|-----------|-------|-----------|----------|--------|------\n');
    for i = 1:5
        isP = ''; if top5.IsPareto(i), isP = '(Pareto)'; end
        fprintf('  %d  | %4.1f | %5.1f | %9.1f | %5.1f | %9.2f | %8.2f | %6.2f | %s\n', ...
            i, top5.Score(i), top5.Off_X(i), top5.Off_Y(i), top5.Off_Z(i), ...
            top5.Cov_Base(i), top5.Cov_Eff(i), top5.Redundancy(i), isP);
    end
    
    % =====================================================================
    % 6. 可视化绘图
    % =====================================================================
    
    figure('Name', '布站参数性能分析 (Y-Up 坐标系适配)', 'Position',[50, 50, 1100, 750]);
    
    % 图1: 4D 性能全景图
    subplot(2,2,1);
    % 为了符合视觉习惯，将 Y 轴数据传给 scatter3 的第三个参数（视觉高度Z）
    scatter3(data.Off_X, data.Off_Z, data.Off_Y, 40, data.Score, 'filled');
    xlabel('X 扩张 (m)'); ylabel('Z 扩张 (m)'); zlabel('Y 高度偏移 (m)');
    cb = colorbar; cb.Label.String = '综合评分 (Score)';
    title('图1: 布站参数空间与综合评分');
    grid on; axis equal; view(45, 30);
    colormap(jet);
    
    % 图2: 帕累托前沿 (Trade-off)
    subplot(2,2,2);
    scatter(data.Cov_Eff, data.Redundancy, 20, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    scatter(data.Cov_Eff(isPareto), data.Redundancy(isPareto), 50, 'r', 'filled');
    plot(top5.Cov_Eff(1), top5.Redundancy(1), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y');
    xlabel('有效覆盖率 (%)'); ylabel('冗余度 (%)');
    title('图2: 帕累托前沿 - 性能权衡');
    legend('普通解', '非支配解(Pareto)', '综合最优(Top1)', 'Location', 'best');
    grid on;
    
    % 图3: 单因素敏感性分析
    subplot(2,2,3);
    analyzeSensitivity(data);
    title('图3: 单参数敏感性趋势');
    
    % 图4: 最佳高度(Y)下的切片热力图
    subplot(2,2,4);
    bestY = top5.Off_Y(1);
    tol = 1e-4;
    % 根据最佳高度 Y 进行切片
    sliceData = data(abs(data.Off_Y - bestY) < tol, :);
    if height(sliceData) >= 3
        try
            % 因为 Y 是高度，平面展开为 X-Z 平面
            [Xq, Zq] = meshgrid(unique(sliceData.Off_X), unique(sliceData.Off_Z));
            Vq = griddata(sliceData.Off_X, sliceData.Off_Z, sliceData.Score, Xq, Zq);
            contourf(Xq, Zq, Vq, 20, 'LineColor', 'none');
            colorbar;
            hold on;
            plot(top5.Off_X(1), top5.Off_Z(1), 'rp', 'MarkerSize', 10, 'MarkerFaceColor','w');
            xlabel('X 扩张 (m)'); ylabel('Z 扩张 (m)');
            title(sprintf('图4: 最佳高度 (Y=%.1fm) 下的热力图', bestY));
        catch
            text(0.5, 0.5, '数据不足以生成网格热力图', 'HorizontalAlignment', 'center');
        end
    else
        text(0.5, 0.5, '该高度下数据点过少', 'HorizontalAlignment', 'center');
    end
    
    % 图5: Top 3 方案雷达对比图
    drawRadarComparison(top5(1:3, :));
    
    fprintf('\n>> 分析完成! 图表已生成。\n');
end

% =========================================================================
% 辅助绘图函数
% =========================================================================

function analyzeSensitivity(data)
    % X Sensitivity
    [G_x, ID_x] = findgroups(data.Off_X);
    MeanScore_X = splitapply(@mean, data.Score, G_x);
    
    % Y Sensitivity (Height)
    [G_y, ID_y] = findgroups(data.Off_Y);
    MeanScore_Y = splitapply(@mean, data.Score, G_y);
    
    % Z Sensitivity
    [G_z, ID_z] = findgroups(data.Off_Z);
    MeanScore_Z = splitapply(@mean, data.Score, G_z);
    
    hold on;
    plot(ID_x, MeanScore_X, '-o', 'LineWidth', 1.5, 'DisplayName', 'X 扩张影响');
    plot(ID_z, MeanScore_Z, '-^', 'LineWidth', 1.5, 'DisplayName', 'Z 扩张影响');
    % 将高度 Y 的线型重点区分
    plot(ID_y, MeanScore_Y, '-s', 'LineWidth', 2.0, 'Color',[0.85 0.32 0.09], 'DisplayName', 'Y 高度偏移影响');
    
    xlabel('参数值 (m)'); ylabel('平均综合评分');
    grid on; legend('Location', 'best');
end

function drawRadarComparison(top3)
    metrics =[top3.Cov_Base, top3.Cov_Eff, top3.Redundancy];
    norm_metrics = metrics; 
    
    labels = {'基础覆盖', '有效覆盖(密度)', '冗余度'};
    nVars = 3;
    
    figure('Name', 'Top 3 方案雷达对比', 'Position', [600, 200, 500, 400]);
    
    theta = linspace(0, 2*pi, nVars+1);
    rho_ticks =[20, 40, 60, 80, 100];
    hold on; axis equal;
    for r = rho_ticks
        plot(r*cos(theta), r*sin(theta), 'k:', 'Color', [0.8 0.8 0.8]);
    end
    for i = 1:nVars
        plot([0 100*cos(theta(i))],[0 100*sin(theta(i))], 'k-', 'Color', [0.5 0.5 0.5]);
        text(110*cos(theta(i)), 110*sin(theta(i)), labels{i}, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    colors = {'r', 'g', 'b'};
    lineStyles = {'-', '--', ':'};
    
    for i = 1:height(top3)
        vals =[norm_metrics(i, :), norm_metrics(i, 1)]; 
        x = vals .* cos(theta);
        y = vals .* sin(theta);
        plot(x, y, 'LineWidth', 2, 'Color', colors{i}, 'LineStyle', lineStyles{i}, ...
            'DisplayName', sprintf('Rank %d (Score: %.1f)', i, top3.Score(i)));
        fill(x, y, colors{i}, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    end
    
    legend('Location', 'southoutside', 'Orientation', 'horizontal');
    title('Top 3 方案多维能力对比');
    axis off;
end