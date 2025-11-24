%% 完整流程：洛伦兹系统ESN预测 + 算法指定Wout扰动（N+2维随机变量）+ 相关性筛选
%RC超参数如N，W谱半径，range_b等需根据实际情况设定
clear; clc; 
% =========================================================================
% 1. 生成洛伦兹系统数据（纯混沌稳态，避免初始瞬态干扰）
% =========================================================================
sigma_true = 10;    % 真实洛伦兹参数
beta_true = 8/3;    
rho_true = 28;      
dt = 0.01;          % 时间步
washout = 1000;     % 洗出期（去除初始瞬态）
train_len = 8000;   % 训练数据长度
test_len = 1000;     % 筛选用测试数据长度
long_test_len = 1000;  % 最终长期预测长度
total_len = washout + train_len + test_len + long_test_len;  % 总数据长度

% 洛伦兹微分方程
lorenz_ode = @(t, x) [sigma_true*(x(2)-x(1)); 
                      x(1)*(rho_true - x(3)) - x(2); 
                      x(1)*x(2) - beta_true*x(3)];

% 生成数据（ode45求解）
t_span = linspace(0, (total_len-1)*dt, total_len);
x0 = [1; 1; 1];  % 初始状态
[t_sol, X_sol] = ode45(lorenz_ode, t_span, x0);
X = X_sol';  % 维度：3×total_len（3行：x/y/z，列：时间步）

% 数据划分（跳过洗出期）
X_train = X(:, washout+1 : washout+train_len);               
X_test = X(:, washout+train_len+1 : washout+train_len+test_len);  
X_long_test = X(:, washout+train_len+test_len+1 : end); 

fprintf('数据生成完成：\n');
fprintf('  训练数据（U）：3×%d\n  筛选测试数据：3×%d\n  长期预测数据：3×%d\n', ...
    train_len, test_len, long_test_len);

% =========================================================================
% 2. 初始化ESN储备池网络（算法第1步：X = [1; U; r]，储备池状态包含r）
% =========================================================================
N = 500;             % 储备池节点数（算法中的N）
in_dim = 3;          % 输入维度（x/y/z）
out_dim = 3;         % 输出维度（x/y/z）
leak_rate = 0.3;     % 泄漏率
lambda = 2e-4;       % 岭回归正则化系数（算法中的ε）
sparsity = 0.05;     % 储备池稀疏度
spect_rad = 0.9;     % 谱半径

% 输入权重矩阵（N×3，算法中U的输入权重）
Win = randn(N, in_dim) * 0.5;

% 储备池连接矩阵（N×N，算法中的r相关连接）
W = randn(N, N);
W(rand(N, N) > sparsity) = 0;  % 稀疏化
max_eig = max(abs(eig(W)));
W = W * (spect_rad / max_eig);  % 调整谱半径

% 偏置项（N×1，算法中的常数项1的权重）
bias = randn(N, 1) * 0.2;

fprintf('\n储备池初始化完成：节点数N=%d，谱半径=%.2f，泄漏率=%.2f\n', ...
    N, spect_rad, leak_rate);

% =========================================================================
% 3. 训练储备池状态（算法第1步：生成储备池状态X）
% =========================================================================
R = zeros(N, train_len);       % 储备池状态矩阵
x_train_final = zeros(N, 1);  % 训练最终状态

for t = 1:train_len
    u = X_train(:, t);  % 当前训练输入（U的第t步，3×1）
    x_train_final = (1 - leak_rate) * x_train_final + leak_rate * tanh(W * x_train_final + Win * u + bias);
    R(:, t) = x_train_final;
end

fprintf('储备池训练完成：状态矩阵（X）维度=%d×%d\n', N, train_len);

% =========================================================================
% 4. 岭回归计算初始输出权重Wout_initial（算法第2步）
% =========================================================================
Y_train = X_train(:, 2:end);   
R_train = R(:, 1:end-1);       
Wout_initial = Y_train * R_train' / (R_train * R_train' + lambda * eye(N));

fprintf('初始Wout计算完成：维度=3×%d（out_dim×N）\n', N);
fprintf('Wout_initial维度验证：%d行×%d列\n', size(Wout_initial,1), size(Wout_initial,2));

% =========================================================================
% 5. 生成候选Wout（算法指定：N+2维随机变量→单位矩阵→rs缩放扰动）
% =========================================================================
num_candidates = ;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 扰动组数需视实际情况而定
candidate_Wout = cell(num_candidates, 1);  % 存储所有候选Wout

% 算法核心：扰动向量与Wout同维度
Wout_rows = size(Wout_initial, 1);  
Wout_cols = size(Wout_initial, 2);  

fprintf('\n开始生成%d个候选Wout（算法指定：N+2维随机变量+单位矩阵+rs扰动）...\n', num_candidates);
for s = 1:num_candidates
    % --------------------------- 算法第8-13步：严格按矩阵维度实现 ---------------------------
    % 第8-9步：生成N+2维随机变量→扩展为与Wout同维度的随机矩阵（3×500）
    z = randn(Wout_rows, Wout_cols);  % 3×500随机矩阵（元素服从标准正态分布）
    
    % 第10步：计算矩阵的Frobenius范数
    norm_z = norm(z, 'fro');
    
    % 第11步：归一化得到单位矩阵d（Frobenius范数=1）
    d = z / norm_z;

    % 第12步：采样rs~Uniform(0,1)（均匀分布随机数）
    rs = rand(); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%rs范围视实际情况而定

    % 第13步：应用扰动（矩阵加法，维度完全匹配）
    Wout_candidate = Wout_initial + rs * d;
    
    % 存储候选Wout
    candidate_Wout{s} = Wout_candidate;
    
    % 进度输出
    if mod(s, 50) == 0
        fprintf('  已生成%d/%d个候选Wout（当前rs=%.4f）\n', s, num_candidates, rs);
    end
end

fprintf('候选Wout生成完成：共%d个，每个维度=3×%d\n', num_candidates, N);

% =========================================================================
% 6. 相关性筛选（算法第4-5步：CheckCorrelation验证动态同步性）
% =========================================================================
corr_len = 500;  % 用于相关性验证的预测长度
train_tail = X_train(:, end-corr_len+1:end);  % 训练数据尾部（算法中的U，3×500）

% --- 计算初始Wout的预测序列和相关性（算法第3步：初始预测） ---
x_pred_init = x_train_final;  % 继承训练最终状态
pred_initial = zeros(out_dim, corr_len);  % 初始Wout预测序列（3×500）

for t = 1:corr_len
    % 自反馈输入（算法第6步：闭环预测）
    if t == 1
        u = X_train(:, end);  % 初始输入=训练集最后一个点
    else
        u = pred_initial(:, t-1);  % 后续输入=前一步预测结果
    end
    u = reshape(u, in_dim, 1);  % 确保输入是3×1列向量
    
    % 储备池状态更新
    x_pred_init = (1 - leak_rate) * x_pred_init + leak_rate * tanh(W * x_pred_init + Win * u + bias);
    
    % 预测输出
    pred_initial(:, t) = reshape(Wout_initial * x_pred_init, out_dim, 1);
end

% 计算初始Wout的相关性（算法第4步：CheckCorrelation）
corr_init_x = corr(pred_initial(1, :)', train_tail(1, :)');
corr_init_y = corr(pred_initial(2, :)', train_tail(2, :)');
corr_init_z = corr(pred_initial(3, :)', train_tail(3, :)');
score_initial = mean([corr_init_x, corr_init_y, corr_init_z]);

fprintf('\n初始Wout验证完成：平均相关性=%.4f\n', score_initial);

% --- 计算所有候选Wout的预测序列和相关性 ---
candidate_corr_score = zeros(num_candidates, 1);  % 每个候选的相关性得分
candidate_pred = cell(num_candidates, 1);  % 每个候选的预测序列

fprintf('\n开始验证候选Wout的相关性（算法CheckCorrelation）...\n');
for s = 1:num_candidates
    current_Wout = candidate_Wout{s};
    x_pred = x_train_final;  % 继承相同的训练最终状态（公平对比）
    current_pred = zeros(out_dim, corr_len);  % 3×500预测序列
    
    % 闭环预测（与初始Wout逻辑完全一致）
    for t = 1:corr_len
        if t == 1
            u = X_train(:, end);
        else
            u = current_pred(:, t-1);
        end
        u = reshape(u, in_dim, 1);
        
        x_pred = (1 - leak_rate) * x_pred + leak_rate * tanh(W * x_pred + Win * u + bias);
        current_pred(:, t) = reshape(current_Wout * x_pred, out_dim, 1);
    end
    candidate_pred{s} = current_pred;
    
    % 算法第4步：CheckCorrelation（计算与U的互相关）
    corr_x = corr(current_pred(1, :)', train_tail(1, :)');
    corr_y = corr(current_pred(2, :)', train_tail(2, :)');
    corr_z = corr(current_pred(3, :)', train_tail(3, :)');
    candidate_corr_score(s) = mean([corr_x, corr_y, corr_z]);  % 平均相关性得分
    
    % 进度输出
    if mod(s, 50) == 0
        current_best = max(candidate_corr_score(1:s));
        fprintf('  已验证%d/%d个候选Wout，当前最优相关性=%.4f\n', ...
            s, num_candidates, current_best);
    end
end

% --- 筛选最优Wout（算法第15-16步：返回相关性最高的Wout） ---
[best_corr_score, best_idx] = max(candidate_corr_score);
best_Wout = candidate_Wout{best_idx};
best_pred = candidate_pred{best_idx};

% 输出算法筛选结果
fprintf('\n=== 算法筛选结果 ===\n');
fprintf('初始Wout平均相关性：%.4f\n', score_initial);
fprintf('最优候选Wout平均相关性：%.4f（提升：%.2f%%）\n', ...
    best_corr_score, (best_corr_score - score_initial)/score_initial*100);
fprintf('最优候选索引：%d\n', best_idx);
fprintf('各分量相关性对比：\n');
fprintf('  x分量：初始=%.4f | 最优=%.4f\n', corr_init_x, corr(best_pred(1,:)', train_tail(1,:)'));
fprintf('  y分量：初始=%.4f | 最优=%.4f\n', corr_init_y, corr(best_pred(2,:)', train_tail(2,:)'));
fprintf('  z分量：初始=%.4f | 最优=%.4f\n', corr_init_z, corr(best_pred(3,:)', train_tail(3,:)'));

% =========================================================================
% 7. 可视化：输入数据（U）+ 初始Wout输出 + 最优Wout输出
% =========================================================================
figure('Position', [10, 10, 800, 800]);
time = 1:corr_len;

% x分量对比
subplot(3,1,1);
plot(time, train_tail(1,:), 'b-', 'LineWidth', 2.0, 'DisplayName', '输入数据U（训练集尾部）'); hold on;
plot(time, pred_initial(1,:), 'c--', 'LineWidth', 1.5, 'DisplayName', '初始Wout预测');
plot(time, best_pred(1,:), 'r-.', 'LineWidth', 1.8, 'DisplayName', '最优Wout预测');
title('x分量：输入数据 vs 初始Wout预测 vs 最优Wout预测');
ylabel('x'); legend('Location', 'best'); grid on;

% y分量对比
subplot(3,1,2);
plot(time, train_tail(2,:), 'b-', 'LineWidth', 2.0); hold on;
plot(time, pred_initial(2,:), 'c--', 'LineWidth', 1.5);
plot(time, best_pred(2,:), 'r-.', 'LineWidth', 1.8);
title('y分量'); ylabel('y'); grid on;

% z分量对比
subplot(3,1,3);
plot(time, train_tail(3,:), 'b-', 'LineWidth', 2.0); hold on;
plot(time, pred_initial(3,:), 'c--', 'LineWidth', 1.5);
plot(time, best_pred(3,:), 'r-.', 'LineWidth', 1.8);
title(sprintf('z分量（初始相关性：%.4f | 最优相关性：%.4f）', score_initial, best_corr_score));
xlabel('时间步'); ylabel('z'); grid on;

sgtitle('算法筛选结果对比：输入数据 vs 初始Wout vs 最优Wout');

% 3D相空间对比（直观展示混沌动态同步性）
figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1);
plot3(train_tail(1,:), train_tail(2,:), train_tail(3,:), 'b-', 'LineWidth', 1.2);
title('输入数据U的相空间'); xlabel('x'); ylabel('y'); zlabel('z'); grid on;
subplot(1,3,2);
plot3(pred_initial(1,:), pred_initial(2,:), pred_initial(3,:), 'c-', 'LineWidth', 1.2);
title('初始Wout预测的相空间'); xlabel('x'); ylabel('y'); zlabel('z'); grid on;
subplot(1,3,3);
plot3(best_pred(1,:), best_pred(2,:), best_pred(3,:), 'r-', 'LineWidth', 1.2);
title('最优Wout预测的相空间'); xlabel('x'); ylabel('y'); zlabel('z'); grid on;

sgtitle('3D相空间对比（动态同步性验证）');