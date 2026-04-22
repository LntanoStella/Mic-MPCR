function LidarBatchRunner_Pro()
    % LidarBatchRunner_Pro - 工程级批量仿真工具
    clc; clear; close all;
    
    % --- 全局数据 ---
    app = struct();
    app.fig = [];
    app.geoData = []; 
    
    % 默认基准 (会被导入覆盖)
    app.baseConfig = struct('pos', [], 'targetPos', []);
    % 默认给一个矩形阵列防止空指针
    app.baseConfig.pos = [-5 -5 4; -5 5 4; 5 -5 4; 5 5 4];
    for k=1:4, app.baseConfig.targetPos(k,:) = [0 0 0]; end
    
    % 雷达与物理参数
    app.lidarParams = struct('maxRange',60, 'minRange',0.5, 'VFOV',40, 'HFOV',120, ...
                             'Channels',64, 'HRes',0.2, 'distNoise',0.02, 'angleNoise',0.05);
    app.physicsParams = struct('maxIncidence',80, 'densityThresh',10);

    createGUI();

    %% ====================================================================
    %  GUI 构建
    % ====================================================================
    function createGUI()
        app.fig = uifigure('Name', '批量仿真 Pro (增量保存 & 详细日志)', 'Position', [100, 100, 1350, 850]);
        
        pnl = uipanel(app.fig, 'Position', [10, 10, 420, 830], 'Title', '参数配置');
        y = 750; dy = 30;

        % --- 1. 基础资源 ---
        uilabel(pnl, 'Position', [10, y, 300, 20], 'Text', '1. 基础资源', 'FontWeight', 'bold'); y=y-dy;
        uibutton(pnl, 'Position', [10, y, 110, 25], 'Text', '加载 STL', 'ButtonPushedFcn', @loadSTL);
        app.lblModel = uilabel(pnl, 'Position', [130, y, 250, 25], 'Text', '未加载', 'FontColor',[0.5 0.5 0.5]); y=y-dy;
        
        uibutton(pnl, 'Position', [10, y, 180, 25], 'Text', '导入传感器配置(基准)', ...
            'ButtonPushedFcn', @loadBaseConfig, 'BackgroundColor', [0.9 0.95 1.0], 'FontWeight', 'bold'); 
        app.lblConfig = uilabel(pnl, 'Position', [200, y, 200, 25], 'Text', '未导入(使用默认)', 'FontColor',[0.5 0.5 0.5]); y=y-dy-10;

        % --- 2. 搜索空间设置 ---
        uilabel(pnl, 'Position', [10, y, 400, 20], 'Text', '2. 相对搜索空间 (基于基准坐标)', 'FontWeight', 'bold'); y=y-25;
        
        % 表头
        headers = {'Min偏移', 'Max偏移', '步长(Step)'};
        x_pos = [100, 200, 300];
        for i=1:3, uilabel(pnl, 'Position', [x_pos(i), y, 90, 20], 'Text', headers{i}, 'FontSize', 11); end
        y=y-25;
        
        % X 轴 (扩张)
        uilabel(pnl, 'Position', [10, y, 80, 20], 'Text', 'X 扩张(m):', 'FontWeight', 'bold');
        app.efX_Min  = uieditfield(pnl, 'numeric', 'Position', [100, y, 80, 20], 'Value', -2);
        app.efX_Max  = uieditfield(pnl, 'numeric', 'Position', [200, y, 80, 20], 'Value', 2);
        app.efX_Step = uieditfield(pnl, 'numeric', 'Position', [300, y, 80, 20], 'Value', 1); y=y-dy;
        
        % Y 轴 (扩张)
        uilabel(pnl, 'Position', [10, y, 80, 20], 'Text', 'Y 扩张(m):', 'FontWeight', 'bold');
        app.efY_Min  = uieditfield(pnl, 'numeric', 'Position', [100, y, 80, 20], 'Value', -2);
        app.efY_Max  = uieditfield(pnl, 'numeric', 'Position', [200, y, 80, 20], 'Value', 2);
        app.efY_Step = uieditfield(pnl, 'numeric', 'Position', [300, y, 80, 20], 'Value', 1); y=y-dy;
        
        % Z 轴 (升降)
        uilabel(pnl, 'Position', [10, y, 80, 20], 'Text', 'Z 偏移(m):', 'FontWeight', 'bold');
        app.efZ_Min  = uieditfield(pnl, 'numeric', 'Position', [100, y, 80, 20], 'Value', -1);
        app.efZ_Max  = uieditfield(pnl, 'numeric', 'Position', [200, y, 80, 20], 'Value', 1);
        app.efZ_Step = uieditfield(pnl, 'numeric', 'Position', [300, y, 80, 20], 'Value', 0.5); y=y-dy-10;
        
        app.lblCount = uilabel(pnl, 'Position', [10, y, 380, 20], 'Text', '预计实验组数: 0', 'FontColor', 'blue'); y=y-dy;
        uibutton(pnl, 'Position', [10, y, 380, 30], 'Text', '预览搜索范围 (Preview)', ...
            'ButtonPushedFcn', @previewLayout, 'BackgroundColor', [0.9 0.9 0.8]); y=y-dy*1.5;

        % --- 3. 仿真参数 ---
        uilabel(pnl, 'Position', [10, y, 300, 20], 'Text', '3. 仿真核心参数', 'FontWeight', 'bold'); y=y-dy;
        app.efLines = createField(pnl, '线数:', 64, [10,y,60,20], [80,y,50,20]);
        app.efHRes  = createField(pnl, 'H Res(°):', 0.2, [150,y,60,20], [220,y,50,20]); y=y-dy;
        app.efRange = createField(pnl, 'Range(m):', 60, [10,y,60,20], [80,y,50,20]); 
        app.chkPhysics = uicheckbox(pnl, 'Position', [150, y, 200, 20], 'Text', '启用物理模型', 'Value', true); y=y-dy-15;
        
        % --- 4. 执行 ---
        uilabel(pnl, 'Position', [10, y, 300, 20], 'Text', '4. 执行', 'FontWeight', 'bold'); y=y-dy;
        app.btnRun = uibutton(pnl, 'Position', [10, y, 380, 40], 'Text', '开始批量仿真 (Batch Run)', ...
            'BackgroundColor', [0.2 0.5 0.3], 'FontWeight', 'bold', 'FontColor', 'w', ...
            'Enable', 'off', 'ButtonPushedFcn', @startBatchRun);

        % 右侧：3D预览区
        app.ax = uiaxes(app.fig, 'Position', [440, 10, 900, 830]);
        app.ax.Title.String = '布局预览';
        app.ax.XLabel.String = 'X'; app.ax.YLabel.String = 'Y'; app.ax.ZLabel.String = 'Z';
        grid(app.ax, 'on'); axis(app.ax, 'equal'); view(app.ax, 45, 30);
        lighting(app.ax, 'gouraud'); camlight(app.ax, 'headlight');
    end

    %% ====================================================================
    %  回调函数
    % ====================================================================

    function loadSTL(~, ~)
        [f, p] = uigetfile({'*.stl;*.STL'}, '选择模型');
        if f == 0, return; end
        try
            gm = stlread(fullfile(p, f));
            V = gm.Points / 1000.0; F = gm.ConnectivityList;
            V1=V(F(:,1),:); V2=V(F(:,2),:); V3=V(F(:,3),:);
            cp = cross(V2-V1, V3-V1, 2);
            area = 0.5 * vecnorm(cp, 2, 2);
            normals = normalize(cp, 2, 'norm');
            centers = (V1+V2+V3)/3;
            
            app.geoData = struct('vertices',V, 'faces',F, 'areas',area, 'normals',normals, 'centers',centers, 'totalArea',sum(area));
            
            app.lblModel.Text = f; app.btnRun.Enable = 'on';
            previewLayout();
        catch ME, uialert(app.fig, ME.message, '错误'); end
    end

    function loadBaseConfig(~, ~)
        [f, p] = uigetfile('*.mat', '选择传感器配置');
        if f == 0, return; end
        try
            d = load(fullfile(p, f));
            if isfield(d, 'sensors'), src = d.sensors;
            elseif isfield(d, 'sensorConfig'), src = d.sensorConfig;
            else, error('文件无效'); end
            
            % 将 struct array 转换为 matrix 方便处理
            % app.baseConfig.pos: 4x3
            % app.baseConfig.targetPos: 4x3
            numS = min(length(src), 4);
            for k=1:numS
                app.baseConfig.pos(k,:) = src(k).pos;
                app.baseConfig.targetPos(k,:) = src(k).targetPos;
            end
            
            app.lblConfig.Text = f; app.lblConfig.FontColor = 'black';
            previewLayout();
            uialert(app.fig, '配置已导入。', '成功', 'Icon', 'success');
        catch ME, uialert(app.fig, ME.message, '导入错误'); end
    end

    function previewLayout(~, ~)
        if isempty(app.geoData), return; end
        
        dx_vec = app.efX_Min.Value : app.efX_Step.Value : app.efX_Max.Value;
        dy_vec = app.efY_Min.Value : app.efY_Step.Value : app.efY_Max.Value;
        dz_vec = app.efZ_Min.Value : app.efZ_Step.Value : app.efZ_Max.Value;
        
        total = length(dx_vec) * length(dy_vec) * length(dz_vec);
        app.lblCount.Text = sprintf('预计实验组数: %d (耗时约 %.1f 分钟)', total, total*0.5/60);
        
        % 绘图
        ax = app.ax; cla(ax); hold(ax, 'on');
        patch(ax, 'Vertices', app.geoData.vertices, 'Faces', app.geoData.faces, ...
              'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceLighting', 'gouraud', 'AmbientStrength', 0.5);
        
        center = mean(app.baseConfig.pos, 1);
        scatter3(ax, center(1), center(2), center(3), 50, 'k', 'Marker', '+');
        
        cols = [0.9 0.3 0.1; 0.2 0.8 0.3; 0.2 0.4 0.9; 0.8 0.2 0.8];
        
        for k=1:4
            p = app.baseConfig.pos(k,:);
            t = app.baseConfig.targetPos(k,:);
            col = cols(k, :);
            
            scatter3(ax, p(1), p(2), p(3), 80, col, 'filled', 'MarkerEdgeColor', 'k');
            plot3(ax, [p(1) t(1)], [p(2) t(2)], [p(3) t(3)], '--', 'Color', [col 0.5]);
            text(ax, p(1), p(2), p(3)+0.5, sprintf('S%d', k), 'FontSize', 8);
            
            % 搜索范围框
            dirX = sign(p(1) - center(1)); if dirX==0, dirX=1; end
            dirY = sign(p(2) - center(2)); if dirY==0, dirY=1; end
            
            x_range = p(1) + [min(dx_vec), max(dx_vec)] * dirX;
            y_range = p(2) + [min(dy_vec), max(dy_vec)] * dirY;
            z_range = p(3) + [min(dz_vec), max(dz_vec)];
            
            drawBox(ax, sort(x_range), sort(y_range), sort(z_range), col);
        end
        title(ax, '布局预览 (实心:基准 / 线框:参数搜索范围)');
        hold(ax, 'off');
    end

    function drawBox(ax, x, y, z, col)
        plot3(ax, [x(1) x(2) x(2) x(1) x(1)], [y(1) y(1) y(2) y(2) y(1)], [z(1) z(1) z(1) z(1) z(1)], 'Color', col);
        plot3(ax, [x(1) x(2) x(2) x(1) x(1)], [y(1) y(1) y(2) y(2) y(1)], [z(2) z(2) z(2) z(2) z(2)], 'Color', col);
        for i=1:2, for j=1:2, plot3(ax, [x(i) x(i)], [y(j) y(j)], [z(1) z(2)], 'Color', col); end; end
    end

    function startBatchRun(~, ~)
        if isempty(app.geoData), return; end
        
        % 1. 生成实验列表
        dx_vec = app.efX_Min.Value : app.efX_Step.Value : app.efX_Max.Value;
        dy_vec = app.efY_Min.Value : app.efY_Step.Value : app.efY_Max.Value;
        dz_vec = app.efZ_Min.Value : app.efZ_Step.Value : app.efZ_Max.Value;
        
        [DX, DY, DZ] = meshgrid(dx_vec, dy_vec, dz_vec);
        expList = [DX(:), DY(:), DZ(:)];
        numExp = size(expList, 1);
        
        if numExp == 0, uialert(app.fig,'搜索空间为空','Err'); return; end
        
        selection = uiconfirm(app.fig, sprintf('运行 %d 组仿真？\n(数据将实时保存至 CSV)', numExp), '确认', 'Icon', 'info');
        if ~strcmp(selection, 'OK'), return; end
        
        % 2. 准备 CSV 文件和表头 (增量保存的核心)
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        csvFileName = sprintf('BatchLog_%s.csv', timestamp);
        matFileName = sprintf('BatchData_%s.mat', timestamp);
        
        % 定义列名 (18列 + 3指标 = 21列)
        colNames = {'Off_X', 'Off_Y', 'Off_Z', ...
                    'S1_X', 'S1_Y', 'S1_Z', 'S2_X', 'S2_Y', 'S2_Z', ...
                    'S3_X', 'S3_Y', 'S3_Z', 'S4_X', 'S4_Y', 'S4_Z', ...
                    'Cov_Base', 'Cov_Eff', 'Redundancy'};
        
        % 写入表头 (WriteMode: overwrite for first time)
        % 创建一个空表头用于初始化文件
        T_head = table('Size', [0, length(colNames)], 'VariableNames', colNames, ...
             'VariableTypes', repmat({'double'}, 1, length(colNames)));
        writetable(T_head, csvFileName); 
        
        % 3. 准备内存大表 (用于最后保存 MAT)
        % 严格预分配，修复 Warning 1
        results = table('Size', [numExp, length(colNames)], ...
            'VariableNames', colNames, ...
            'VariableTypes', repmat({'double'}, 1, length(colNames)));
        
        % 参数准备
        app.lidarParams.Channels = app.efLines.Value;
        app.lidarParams.HRes     = app.efHRes.Value;
        app.lidarParams.maxRange = app.efRange.Value;
        usePhy = app.chkPhysics.Value;
        
        hWait = waitbar(0, '初始化仿真内核...', 'WindowStyle', 'modal');
        tStart = tic;
        
        center = mean(app.baseConfig.pos, 1);
        
        fprintf('\n======== 开始批量仿真 (共 %d 组) ========\n', numExp);
        fprintf('数据实时写入: %s\n', csvFileName);
        
        try
            for i = 1:numExp
                off_x = expList(i, 1);
                off_y = expList(i, 2);
                off_z = expList(i, 3);
                
                % 生成当前传感器位置
                currentPos = zeros(4, 3);
                currentSensors = struct('pos',{},'d',{},'u',{},'v',{});
                
                for k=1:4
                    baseP = app.baseConfig.pos(k,:);
                    targetP = app.baseConfig.targetPos(k,:);
                    
                    dirX = sign(baseP(1) - center(1)); if dirX==0, dirX=1; end
                    dirY = sign(baseP(2) - center(2)); if dirY==0, dirY=1; end
                    
                    newP = baseP + [off_x*dirX, off_y*dirY, off_z];
                    currentPos(k, :) = newP;
                    
                    % 向量
                    diff = targetP - newP; d=diff/norm(diff);
                    if abs(d(3))<0.99, tmp=[0 0 1]; else, tmp=[0 1 0]; end
                    u=cross(d,tmp); u=u/norm(u); v=cross(d,u);
                    
                    currentSensors(k).pos = newP;
                    currentSensors(k).d = d; currentSensors(k).u = u; currentSensors(k).v = v;
                end
                
                % 运行核心 (无头)
                metrics = runHeadlessSim(app.geoData, currentSensors, app.lidarParams, app.physicsParams, usePhy);
                
                % 4. 构造单行数据 (用于增量写入)
                rowData = {off_x, off_y, off_z, ...
                           currentPos(1,1), currentPos(1,2), currentPos(1,3), ...
                           currentPos(2,1), currentPos(2,2), currentPos(2,3), ...
                           currentPos(3,1), currentPos(3,2), currentPos(3,3), ...
                           currentPos(4,1), currentPos(4,2), currentPos(4,3), ...
                           metrics.base, metrics.eff, metrics.red};
                
                % 填入大表 (内存)
                results(i, :) = rowData;
                
                % 增量写入 CSV (磁盘)
                T_row = table(rowData{:}, 'VariableNames', colNames);
                writetable(T_row, csvFileName, 'WriteMode', 'Append', 'WriteVariableNames', false);
                
                % 5. 日志与进度
                tElap = toc(tStart);
                if mod(i, 5) == 0 || i==1 || i==numExp
                    avgT = tElap / i;
                    remT = avgT * (numExp - i);
                    
                    % 控制台日志
                    fprintf('[%d/%d] Off:[%.1f, %.1f, %.1f] -> Cov: %.2f%% | Eff: %.2f%% | Red: %.2f%%\n', ...
                        i, numExp, off_x, off_y, off_z, metrics.base, metrics.eff, metrics.red);
                    
                    % 进度条更新
                    waitbar(i/numExp, hWait, sprintf('进度: %d/%d (%.1f%%)\n剩余时间: %.0f 秒', ...
                        i, numExp, (i/numExp)*100, remT));
                end
            end
            delete(hWait);
            
            % 保存完整 MAT
            save(matFileName, 'results', 'app');
            uialert(app.fig, sprintf('全部完成！\nCSV: %s\nMAT: %s', csvFileName, matFileName), '成功');
            
        catch ME
            if isvalid(hWait), delete(hWait); end
            fprintf('!!! 运行中断: %s\n', ME.message);
            uialert(app.fig, ME.message, '运行错误');
        end
    end

    %% ====================================================================
    %  核心计算 (无头模式)
    % ====================================================================
    function metrics = runHeadlessSim(geo, sensors, lP, pP, usePhysics)
        numFaces = size(geo.faces, 1);
        hitCounts = zeros(numFaces, 1);
        sensorMap = false(numFaces, length(sensors));
        
        el_vec = linspace(-lP.VFOV/2, lP.VFOV/2, lP.Channels);
        az_vec = -lP.HFOV/2 : lP.HRes : lP.HFOV/2;
        [AZ, EL] = meshgrid(deg2rad(az_vec), deg2rad(el_vec));
        
        if usePhysics
            nScale = deg2rad(lP.angleNoise);
            AZ = AZ + randn(size(AZ)) * nScale;
            EL = EL + randn(size(EL)) * nScale;
        end
        Lx = cos(EL).*cos(AZ); Ly = cos(EL).*sin(AZ); Lz = sin(EL);
        raysLocal = [Lx(:), Ly(:), Lz(:)];
        
        for sIdx = 1:length(sensors)
            sen = sensors(sIdx);
            R = [sen.d(:), sen.u(:), sen.v(:)];
            raysGlobal = raysLocal * R';
            
            v2f = geo.centers - sen.pos; d2f = vecnorm(v2f, 2, 2); dn = v2f ./ (d2f + 1e-6);
            mask1 = d2f < lP.maxRange + 5;
            mask2 = (dn * sen.d(:)) > cos(deg2rad(max(lP.HFOV, lP.VFOV)/2 + 20));
            mask3 = sum(dn .* geo.normals, 2) < 0;
            cIdx = find(mask1 & mask2 & mask3);
            if isempty(cIdx), continue; end
            
            [ht, hID] = rayCast(sen.pos, raysGlobal, geo.vertices, geo.faces(cIdx,:), cIdx, lP.maxRange);
            
            valid = ~isinf(ht);
            if any(valid)
                if usePhysics
                    rt = ht(valid); rd = raysGlobal(valid,:); rid = hID(valid);
                    n = geo.normals(rid, :);
                    cosInc = dot(n, -rd, 2);
                    keep = cosInc > cos(deg2rad(pP.maxIncidence));
                    prob = cosInc .* (1 - (rt/lP.maxRange).^2);
                    keep = keep & (rand(size(rt)) < (0.8*prob+0.2));
                    finID = rid(keep);
                else
                    finID = hID(valid);
                end
                for k=1:length(finID)
                    fid = finID(k);
                    hitCounts(fid) = hitCounts(fid) + 1;
                    sensorMap(fid, sIdx) = true;
                end
            end
        end
        
        metrics.base = sum(geo.areas(hitCounts>0)) / geo.totalArea * 100;
        dens = hitCounts ./ (geo.areas + 1e-9);
        metrics.eff  = sum(geo.areas(dens > pP.densityThresh)) / geo.totalArea * 100;
        metrics.red  = sum(geo.areas(sum(sensorMap,2)>=2)) / geo.totalArea * 100;
    end

    function [mt, mid] = rayCast(org, dirs, V, subF, origID, maxR)
        nR = size(dirs,1); nF = size(subF,1);
        mt = inf(nR,1); mid = zeros(nR,1);
        V1=V(subF(:,1),:); V2=V(subF(:,2),:); V3=V(subF(:,3),:);
        E1=V2-V1; E2=V3-V1; TB=org-V1;
        bs = 50000;
        for i=1:bs:nR
            ie = min(i+bs-1, nR); chunk = dirs(i:ie,:); nc = size(chunk,1);
            cmt = inf(nc,1); cmid = zeros(nc,1);
            for f=1:nF
                ce1=E1(f,:); ce2=E2(f,:); ct=TB(f,:);
                P = [chunk(:,2)*ce2(3)-chunk(:,3)*ce2(2), chunk(:,3)*ce2(1)-chunk(:,1)*ce2(3), chunk(:,1)*ce2(2)-chunk(:,2)*ce2(1)];
                det = P * ce1'; mask = abs(det)>1e-6;
                if ~any(mask), continue; end
                invD = 1./det(mask); cP=P(mask,:); cR=chunk(mask,:);
                u = (cP*ct') .* invD; mU = u>=0 & u<=1;
                if ~any(mU), continue; end
                Q = cross(ct, ce1); v = (cR(mU,:)*Q') .* invD(mU);
                mV = v>=0 & (u(mU)+v<=1);
                if ~any(mV), continue; end
                t = dot(ce2, Q) * invD(mU); t = t(mV);
                idx = find(mask); idx=idx(mU); idx=idx(mV);
                vt = t>1e-6 & t<maxR;
                if any(vt)
                    ui = idx(vt); tv = t(vt);
                    closer = tv < cmt(ui);
                    if any(closer), fi = ui(closer); cmt(fi) = tv(closer); cmid(fi) = origID(f); end
                end
            end
            mt(i:ie)=cmt; mid(i:ie)=cmid;
        end
    end

    function ef = createField(p, txt, val, posL, posE)
        uilabel(p, 'Position', posL, 'Text', txt);
        ef = uieditfield(p, 'numeric', 'Position', posE, 'Value', val);
    end
end