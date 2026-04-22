function LidarSim_RealScan()
    % LidarSim_RealScan
    clc; clear; close all;
    
    % --- 全局数据结构 ---
    app = struct();
    app.fig = [];
    app.objectGeometry =[]; 
    app.geoCache = struct('faceNormals', [], 'faceCenters', [], 'faceAreas',[], 'totalArea', 0);
    
    % 真实雷达参数配平 (基于 96线, 垂向分辨率 ~0.94度)
    % VFOV = (96-1) * 0.94 = 89.3度
    app.lidarParams = struct(...
        'maxRange', 25, ...       % 测试时调小射程可看清近处线束
        'minRange', 0.1, ...
        'VFOV_deg', 89.3, ...     % <--- 根据 0.94 * 95 计算得出
        'HFOV_deg', 120, ...      
        'Channels', 96, ...       
        'H_Res_deg', 0.2, ...     % 水平分辨率
        'distNoiseStd', 0.01, ... % 测距噪声，几厘米是很真实的
        'azNoiseStd', 0.05, ...   % 水平电机微小误差
        'elNoiseStd', 0.00 ...    % <--- 绝对不能大！必须接近0，否则线束会被抖散
    );
    
    app.physicsParams = struct('maxIncidenceAngle', 80, 'outlierRatio', 0.0001, 'densityThreshold', 10);

    % 传感器初始化
    app.sensors = struct();
    defaultPos =[-8, -8, 5; -8, 8, 5; 15, -8, 5; 15, 8, 5];
    defaultColor =[0.9 0.3 0.1; 0.2 0.8 0.3; 0.2 0.4 0.9; 0.8 0.2 0.8];
    for i = 1:4
        app.sensors(i).pos = defaultPos(i,:);
        app.sensors(i).targetPos = [0, 0, 0];
        app.sensors(i).showRange = true;
        app.sensors(i).color = defaultColor(i,:);
        app.sensors(i).d = [1 0 0]; app.sensors(i).u = [0 1 0]; app.sensors(i).v =[0 0 1]; 
    end
    
    app.selectedSensorIndex = 1;
    app.simulationResults = struct('sensorPoints', {cell(1,4)}, 'faceHitCounts', [], 'faceSensorMap',[]);
    app.epsilon = 1e-6;
    
    createGUI();
    fprintf('>> LidarSim 真实扫描版就绪。已修复垂直噪声参数，还原完美圆锥线束。\n');

    % =====================================================================
    % GUI 构建
    % =====================================================================
    function createGUI()
        app.fig = uifigure('Name', 'LidarSim - 真实扫描纹理与圆锥视场', 'Position', [50, 50, 1400, 900]);
        
        pnl = uipanel(app.fig, 'Position',[10, 10, 320, 880], 'Title', '控制面板');
        y = 830; dy = 30;
        
        % 1. 模型
        uilabel(pnl, 'Position', [10, y, 300, 20], 'Text', '1. 场景模型', 'FontWeight', 'bold'); y=y-dy;
        uibutton(pnl, 'Position', [10, y, 140, 25], 'Text', '加载STL', 'ButtonPushedFcn', @loadSTLCallback);
        app.lblModelName = uilabel(pnl, 'Position',[160, y, 150, 25], 'Text', '未加载', 'FontColor',[0.5 0.5 0.5]); y=y-dy-10;
        
        % 2. 雷达参数
        uilabel(pnl, 'Position',[10, y, 300, 20], 'Text', '2. 雷达参数', 'FontWeight', 'bold'); y=y-dy;
        app.efChannels = createEditField(pnl, '线数 (Line):', app.lidarParams.Channels, y, @updateParams); y=y-dy;
        app.efHRes = createEditField(pnl, '水平分辨率(°):', app.lidarParams.H_Res_deg, y, @updateParams); y=y-dy;
        app.efVFOV = createEditField(pnl, '垂直 FOV(°):', app.lidarParams.VFOV_deg, y, @updateParams); y=y-dy;
        app.efHFOV = createEditField(pnl, '水平 FOV(°):', app.lidarParams.HFOV_deg, y, @updateParams); y=y-dy;
        app.efRange = createEditField(pnl, '最大射程(m):', app.lidarParams.maxRange, y, @updateParams); y=y-dy-10;

        % 3. 传感器
        uilabel(pnl, 'Position', [10, y, 300, 20], 'Text', '3. 传感器布站', 'FontWeight', 'bold'); y=y-dy;
        app.ddSensor = uidropdown(pnl, 'Position',[10, y, 100, 20], 'Items', {'Lidar 1', 'Lidar 2', 'Lidar 3', 'Lidar 4'}, 'ValueChangedFcn', @selectSensorCallback);
        app.chkRange = uicheckbox(pnl, 'Position', [120, y, 150, 20], 'Text', '显示真实圆锥视场', 'Value', true, 'ValueChangedFcn', @updateParams); y=y-dy;
        
        uilabel(pnl, 'Position',[10, y, 30, 20], 'Text', 'Pos:');
        app.efPx = uieditfield(pnl, 'numeric', 'Position', [40, y, 50, 20], 'Value', 0, 'ValueChangedFcn', @sensorPropChanged);
        app.efPy = uieditfield(pnl, 'numeric', 'Position', [95, y, 50, 20], 'Value', 0, 'ValueChangedFcn', @sensorPropChanged);
        app.efPz = uieditfield(pnl, 'numeric', 'Position',[150, y, 50, 20], 'Value', 0, 'ValueChangedFcn', @sensorPropChanged); y=y-dy;
        
        uilabel(pnl, 'Position',[10, y, 30, 20], 'Text', 'Tar:');
        app.efTx = uieditfield(pnl, 'numeric', 'Position', [40, y, 50, 20], 'Value', 0, 'ValueChangedFcn', @sensorPropChanged);
        app.efTy = uieditfield(pnl, 'numeric', 'Position', [95, y, 50, 20], 'Value', 0, 'ValueChangedFcn', @sensorPropChanged);
        app.efTz = uieditfield(pnl, 'numeric', 'Position',[150, y, 50, 20], 'Value', 0, 'ValueChangedFcn', @sensorPropChanged); 
        uibutton(pnl, 'Position', [210, y, 80, 20], 'Text', '全应用', 'ButtonPushedFcn', @applyGlobalTarget); y=y-dy-15;
        
        % 4. 仿真模式
        uilabel(pnl, 'Position',[10, y, 300, 20], 'Text', '4. 仿真控制', 'FontWeight', 'bold'); y=y-dy;
        app.chkPhysics = uicheckbox(pnl, 'Position', [10, y, 300, 20], 'Text', '启用物理模型 (噪声/衰减/丢点)', 'Value', true); y=y-35;
        app.btnRun = uibutton(pnl, 'Position',[10, y, 300, 35], 'Text', '运行高保真仿真', ...
            'BackgroundColor', [0.2 0.4 0.7], 'FontWeight', 'bold', 'FontColor', 'w', ...
            'Enable', 'off', 'ButtonPushedFcn', @runSimulation); y=y-50;

        % 5. 统计
        app.pnlStats = uipanel(pnl, 'Position',[5, 10, 310, y-10], 'Title', '统计信息');
        app.txtStats = uitextarea(app.pnlStats, 'Position', [5, 5, 300, y-35], 'Editable', 'off', 'FontName', 'Monospaced', 'FontSize', 9);
        
        % 3D View
        app.ax = uiaxes(app.fig, 'Position',[340, 10, 1050, 880]);
        app.ax.Title.String = '激光雷达真实线束扫描仿真';
        grid(app.ax, 'on'); axis(app.ax, 'equal'); view(app.ax, 45, 30);
        lighting(app.ax, 'gouraud'); camlight(app.ax, 'headlight');
        
        updateSensorGUI(); plotScene();
    end

    function ef = createEditField(parent, txt, val, y, cb)
        uilabel(parent, 'Position',[10, y, 120, 20], 'Text', txt);
        ef = uieditfield(parent, 'numeric', 'Position', [130, y, 60, 20], 'Value', val, 'ValueChangedFcn', cb);
    end

    % =====================================================================
    % 核心仿真逻辑
    % =====================================================================
    function runSimulation(~, ~)
        if isempty(app.objectGeometry), return; end
        
        usePhysics = app.chkPhysics.Value;
        numSensors = 4;
        sensorPoints = cell(1, numSensors);
        numFaces = size(app.objectGeometry.faces, 1);
        app.simulationResults.faceHitCounts = zeros(numFaces, 1);
        app.simulationResults.faceSensorMap = false(numFaces, numSensors);
        
        for k=1:numSensors, updateSensorVectors(k); end
        
        waitH = waitbar(0, '运行物理级光线追踪...', 'WindowStyle', 'modal');
        
        try
            for sIdx = 1:numSensors
                sensor = app.sensors(sIdx);
                
                % 生成射线：通过网格生成绝对平滑的环线
                el_vec = linspace(-app.lidarParams.VFOV_deg/2, app.lidarParams.VFOV_deg/2, app.lidarParams.Channels);
                az_vec = -app.lidarParams.HFOV_deg/2 : app.lidarParams.H_Res_deg : app.lidarParams.HFOV_deg/2;
                [AZ, EL] = meshgrid(deg2rad(az_vec), deg2rad(el_vec));
                
                % 仅在水平方向加极细微噪点，垂直方向严格保持零或极小以维持线束！
                if usePhysics
                    AZ = AZ + randn(size(AZ)) * deg2rad(app.lidarParams.azNoiseStd);
                    EL = EL + randn(size(EL)) * deg2rad(app.lidarParams.elNoiseStd);
                end
                
                L_x = cos(EL) .* cos(AZ); L_y = cos(EL) .* sin(AZ); L_z = sin(EL);
                raysLocal = [L_x(:), L_y(:), L_z(:)];
                numRays = size(raysLocal, 1);
                
                waitbar((sIdx-1)/numSensors, waitH, sprintf('Lidar %d: 处理中 (%d 条射线)...', sIdx, numRays));
                
                R = [sensor.d(:), sensor.u(:), sensor.v(:)];
                raysGlobal = raysLocal * R';
                
                % 视锥剔除
                v2f = app.geoCache.faceCenters - sensor.pos;
                distToFace = vecnorm(v2f, 2, 2);
                dirNorm = v2f ./ (distToFace + 1e-6);
                
                maskDist = (distToFace < app.lidarParams.maxRange + 5);
                minCos = cos(deg2rad(max(app.lidarParams.HFOV_deg, app.lidarParams.VFOV_deg)/2 + 10));
                maskAngle = (dirNorm * sensor.d(:)) > minCos;
                maskFacing = sum(dirNorm .* app.geoCache.faceNormals, 2) < 0; 
                
                candidateIdx = find(maskDist & maskAngle & maskFacing);
                if isempty(candidateIdx), sensorPoints{sIdx} =[]; continue; end
                
                % 几何求交
                [hits_t, hits_faceID] = rayCastMesh(sensor.pos, raysGlobal, ...
                    app.objectGeometry.vertices, app.objectGeometry.faces(candidateIdx,:), ...
                    candidateIdx, app.epsilon, app.lidarParams.maxRange);
                
                % 物理处理
                validMask = ~isinf(hits_t);
                if any(validMask)
                    rawT = hits_t(validMask);
                    rawDirs = raysGlobal(validMask, :);
                    rawFaceIDs = hits_faceID(validMask);
                    
                    if usePhysics[finalT, finalDirs, finalFaceIDs] = applyPhysicsModel(rawT, rawDirs, rawFaceIDs, app);
                    else
                        rangeMask = rawT <= app.lidarParams.maxRange;
                        finalT = rawT(rangeMask); finalDirs = rawDirs(rangeMask, :); finalFaceIDs = rawFaceIDs(rangeMask);
                    end
                    
                    if ~isempty(finalT)
                        pts = sensor.pos + finalDirs .* finalT;
                        
                        % 离群点物理模拟：仅依附于激光线产生多径拖尾
                        if usePhysics
                            numOutliers = floor(size(pts,1) * app.physicsParams.outlierRatio);
                            if numOutliers > 0
                                idx = randi(size(finalDirs, 1), numOutliers, 1);
                                outlierDirs = finalDirs(idx, :);
                                outlierOffsets = (rand(numOutliers, 1) * 2.5) - 0.5; 
                                outlierDist = finalT(idx) + outlierOffsets;
                                outlierDist = max(app.lidarParams.minRange, min(app.lidarParams.maxRange, outlierDist));
                                pts = [pts; sensor.pos + outlierDirs .* outlierDist];
                            end
                        end
                        
                        sensorPoints{sIdx} = pts;
                        for fi = 1:length(finalFaceIDs)
                            fid = finalFaceIDs(fi);
                            app.simulationResults.faceHitCounts(fid) = app.simulationResults.faceHitCounts(fid) + 1;
                            app.simulationResults.faceSensorMap(fid, sIdx) = true;
                        end
                    end
                end
            end
            
            delete(waitH);
            app.simulationResults.sensorPoints = sensorPoints;
            updateStats(usePhysics);
            plotScene();
            uialert(app.fig, '仿真完成。线束效果已还原。', '成功');
            
        catch ME
            if isvalid(waitH), delete(waitH); end
            uialert(app.fig, ME.message, 'Error');
        end
    end

    function[filtT, filtDirs, filtIDs] = applyPhysicsModel(t, dirs, faceIDs, app)
        normals = app.geoCache.faceNormals(faceIDs, :);
        cosIncidence = dot(normals, -dirs, 2);
        
        minCos = cos(deg2rad(app.physicsParams.maxIncidenceAngle));
        keepMask = (cosIncidence > minCos);
        
        probKeep = cosIncidence .* (1 - (t / app.lidarParams.maxRange).^2);
        probKeep = max(0, probKeep); 
        keepMask = keepMask & (rand(length(t), 1) < (0.9 * probKeep + 0.1)); 
        
        noise = app.lidarParams.distNoiseStd * randn(length(t), 1);
        t = t + noise;
        
        filtT = t(keepMask); filtDirs = dirs(keepMask, :); filtIDs = faceIDs(keepMask);
    end

    function[min_t, min_faceID] = rayCastMesh(origin, rays, vertices, faces, originalFaceIDs, eps, maxRange)
        numRays = size(rays, 1); numFaces = size(faces, 1);
        min_t = inf(numRays, 1); min_faceID = zeros(numRays, 1);
        V1=vertices(faces(:,1),:); V2=vertices(faces(:,2),:); V3=vertices(faces(:,3),:);
        E1=V2-V1; E2=V3-V1; TB=origin-V1;
        
        bs = 50000;
        for rStart = 1:bs:numRays
            rEnd = min(rStart+bs-1, numRays); chunkRays = rays(rStart:rEnd, :);
            chunk_min_t = inf(size(chunkRays,1), 1); chunk_face_id = zeros(size(chunkRays,1), 1);
            
            for f = 1:numFaces
                ce1=E1(f,:); ce2=E2(f,:); ct=TB(f,:);
                P =[chunkRays(:,2)*ce2(3)-chunkRays(:,3)*ce2(2), chunkRays(:,3)*ce2(1)-chunkRays(:,1)*ce2(3), chunkRays(:,1)*ce2(2)-chunkRays(:,2)*ce2(1)];
                det = P * ce1'; mask = abs(det)>eps; if ~any(mask), continue; end
                invDet = 1./det(mask); currP=P(mask,:); currRays=chunkRays(mask,:);
                u = (currP*ct').*invDet; maskU = u>=0 & u<=1; if ~any(maskU), continue; end
                Q = cross(ct, ce1); v=(currRays(maskU,:)*Q').*invDet(maskU);
                maskV = v>=0 & (u(maskU)+v<=1); if ~any(maskV), continue; end
                t = dot(ce2, Q)*invDet(maskU); t = t(maskV);
                
                idx = find(mask); idx=idx(maskU); idx=idx(maskV);
                vt = t>eps & t<maxRange;
                if any(vt)
                    upd=idx(vt); val=t(vt); closer=val<chunk_min_t(upd);
                    if any(closer), fi=upd(closer); chunk_min_t(fi)=val(closer); chunk_face_id(fi)=originalFaceIDs(f); end
                end
            end
            min_t(rStart:rEnd)=chunk_min_t; min_faceID(rStart:rEnd)=chunk_face_id;
        end
    end

    function updateStats(isPhysicsMode)
        if isempty(app.geoCache.faceAreas), return; end
        counts = app.simulationResults.faceHitCounts;
        areas = app.geoCache.faceAreas;
        densities = counts ./ (areas + 1e-9);
        effectiveArea = sum(areas(densities > app.physicsParams.densityThreshold));
        totalArea = app.geoCache.totalArea;
        
        totalPts = 0;
        for k=1:4, if ~isempty(app.simulationResults.sensorPoints{k}), totalPts=totalPts+size(app.simulationResults.sensorPoints{k},1); end; end
        
        txt = {
            sprintf('模式: %s', string(isPhysicsMode));
            sprintf('有效覆盖率: %.2f %%', (effectiveArea/totalArea)*100);
            sprintf('冗余覆盖率: %.2f %%', (sum(areas(sum(app.simulationResults.faceSensorMap,2)>=2))/totalArea)*100);
            ' ';
            sprintf('总点云数: %d', totalPts);
        };
        app.txtStats.Value = txt;
    end

    function loadSTLCallback(~, ~)
        [f, p] = uigetfile({'*.stl;*.STL'}, '选择STL');
        if isequal(f, 0), return; end
        try
            gm = stlread(fullfile(p, f));
            app.objectGeometry.vertices = gm.Points / 1000.0; app.objectGeometry.faces = gm.ConnectivityList;
            V=app.objectGeometry.vertices; F=app.objectGeometry.faces;
            V1=V(F(:,1),:); V2=V(F(:,2),:); V3=V(F(:,3),:); cp=cross(V2-V1, V3-V1, 2);
            app.geoCache.faceAreas = 0.5*vecnorm(cp,2,2); app.geoCache.totalArea = sum(app.geoCache.faceAreas);
            app.geoCache.faceNormals = normalize(cp,2,'norm'); app.geoCache.faceCenters = (V1+V2+V3)/3;
            
            center = mean(V, 1);
            for k=1:4, app.sensors(k).targetPos = center; updateSensorVectors(k); end
            app.lblModelName.Text = f; app.btnRun.Enable = 'on';
            updateSensorGUI(); plotScene();
        catch ME, uialert(app.fig, ME.message, '错误'); end
    end

    function plotScene()
        if isempty(app.fig) || ~isvalid(app.fig), return; end
        ax = app.ax; cla(ax); hold(ax, 'on');
        
        if ~isempty(app.objectGeometry)
            faceColors = repmat([0.7 0.7 0.7], size(app.objectGeometry.faces, 1), 1);
            patch(ax, 'Vertices', app.objectGeometry.vertices, 'Faces', app.objectGeometry.faces, ...
                  'FaceVertexCData', faceColors, 'FaceColor', 'flat', 'EdgeColor', 'none', ...
                  'FaceLighting', 'gouraud', 'AmbientStrength', 0.5);
        end
        
        for k=1:4
            s = app.sensors(k);
            if ~isfield(s, 'color') || isempty(s.color), s.color = [0 0.5 0.5]; end
            
            pts = app.simulationResults.sensorPoints{k};
            if ~isempty(pts)
                % 完全不降采样，以极小的点渲染，保证线束纹理清晰
                scatter3(ax, pts(:,1), pts(:,2), pts(:,3), 1.5, s.color, 'filled'); 
            end
            
            ec = 'none'; sz = 50; if k==app.selectedSensorIndex, ec='k'; sz=80; end
            scatter3(ax, s.pos(1), s.pos(2), s.pos(3), sz, s.color, 'filled', 'MarkerEdgeColor', ec);
            if s.showRange && app.chkRange.Value, drawConicalFrustum(ax, s); end
        end
        hold(ax, 'off');
    end

    % 物理级真实圆锥边界渲染
    function drawConicalFrustum(ax, s)
        range = app.lidarParams.maxRange;
        numPts = 40;
        h_vec = linspace(-app.lidarParams.HFOV_deg/2, app.lidarParams.HFOV_deg/2, numPts);
        v_top = app.lidarParams.VFOV_deg/2;
        v_bot = -app.lidarParams.VFOV_deg/2;
        
        % 生成圆锥面上下弧线 (Local)
        dt =[cosd(v_top)*cosd(h_vec)', cosd(v_top)*sind(h_vec)', repmat(sind(v_top),numPts,1)];
        db =[cosd(v_bot)*cosd(h_vec)', cosd(v_bot)*sind(h_vec)', repmat(sind(v_bot),numPts,1)];
        
        % 转全局
        R =[s.d(:) s.u(:) s.v(:)];
        pt = s.pos + (dt * R') * range;
        pb = s.pos + (db * R') * range;
        
        col = [s.color 0.4];
        % 侧边
        plot3(ax, [s.pos(1) pt(1,1)],[s.pos(2) pt(1,2)], [s.pos(3) pt(1,3)], 'Color', col, 'LineWidth', 1);
        plot3(ax, [s.pos(1) pt(end,1)],[s.pos(2) pt(end,2)], [s.pos(3) pt(end,3)], 'Color', col, 'LineWidth', 1);
        plot3(ax,[s.pos(1) pb(1,1)], [s.pos(2) pb(1,2)],[s.pos(3) pb(1,3)], 'Color', col, 'LineWidth', 1);
        plot3(ax,[s.pos(1) pb(end,1)], [s.pos(2) pb(end,2)],[s.pos(3) pb(end,3)], 'Color', col, 'LineWidth', 1);
        
        % 圆锥曲线边
        plot3(ax, pt(:,1), pt(:,2), pt(:,3), 'Color', col, 'LineWidth', 1);
        plot3(ax, pb(:,1), pb(:,2), pb(:,3), 'Color', col, 'LineWidth', 1);
        plot3(ax,[pt(1,1) pb(1,1)],[pt(1,2) pb(1,2)], [pt(1,3) pb(1,3)], 'Color', col, 'LineWidth', 1);
        plot3(ax, [pt(end,1) pb(end,1)],[pt(end,2) pb(end,2)], [pt(end,3) pb(end,3)], 'Color', col, 'LineWidth', 1);
    end

    function updateParams(~, ~)
        app.lidarParams.Channels = app.efChannels.Value; app.lidarParams.H_Res_deg = app.efHRes.Value;
        app.lidarParams.VFOV_deg = app.efVFOV.Value; app.lidarParams.HFOV_deg = app.efHFOV.Value;
        app.lidarParams.maxRange = app.efRange.Value; plotScene();
    end

    function selectSensorCallback(~, ~)
        app.selectedSensorIndex = str2double(erase(app.ddSensor.Value, 'Lidar '));
        updateSensorGUI(); plotScene();
    end

    function sensorPropChanged(~, ~)
        idx = app.selectedSensorIndex;
        app.sensors(idx).pos =[app.efPx.Value, app.efPy.Value, app.efPz.Value];
        app.sensors(idx).targetPos = [app.efTx.Value, app.efTy.Value, app.efTz.Value];
        updateSensorVectors(idx); plotScene();
    end
    
    function applyGlobalTarget(~, ~)
        t = app.sensors(app.selectedSensorIndex).targetPos;
        for k=1:4, app.sensors(k).targetPos = t; updateSensorVectors(k); end
        plotScene();
    end

    function updateSensorVectors(k)
        diff = app.sensors(k).targetPos - app.sensors(k).pos; dist = norm(diff);
        if dist < 1e-6, dir=[1 0 0]; else, dir=diff/dist; end
        app.sensors(k).d = dir;
        if abs(dir(3)) < 0.99, temp=[0 0 1]; else, temp=[0 1 0]; end
        u = cross(dir, temp); app.sensors(k).u = normalize(u, 'norm');
        app.sensors(k).v = cross(dir, app.sensors(k).u);
    end

    function updateSensorGUI()
        s = app.sensors(app.selectedSensorIndex);
        app.efPx.Value=s.pos(1); app.efPy.Value=s.pos(2); app.efPz.Value=s.pos(3);
        app.efTx.Value=s.targetPos(1); app.efTy.Value=s.targetPos(2); app.efTz.Value=s.targetPos(3);
        app.ddSensor.Value = sprintf('Lidar %d', app.selectedSensorIndex);
    end
end