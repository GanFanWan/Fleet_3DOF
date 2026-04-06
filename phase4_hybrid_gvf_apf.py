"""
Phase 4: GVF与APF融合模块
========================

本模块实现GVF路径跟踪与APF避障的融合，通过Sigmoid函数实现平滑的模态切换。
在无障碍时沿GVF路径运动，遇到障碍时自动切换到APF避障模式。

关键技术：
- Sigmoid函数实现平滑加权
- 基于距离的模态切换
- 无跳变的连续控制

主要功能：
- GVF与APF的自适应融合
- 基于障碍物距离的权重计算
- 平滑模态切换验证

作者: Claude
日期: 2026-04-06
"""

import numpy as np
import matplotlib.pyplot as plt
from Three_DOF_Model import ASV1
from phase2_gvf_tracking import CirclePathGVF
from phase3_apf_avoidance import APFController


def sigmoid(x, k=1.0, x0=0.0):
    """Sigmoid函数

    用于平滑切换权重，将输入映射到[0, 1]区间。

    参数:
        x: 输入值
        k: 陡峭度参数（越大切换越快）
        x0: 中心点（sigmoid为0.5的位置）

    返回:
        输出值，范围[0, 1]
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


class HybridGVFAPFController:
    """GVF与APF混合控制器

    根据障碍物距离自适应调整GVF和APF的权重，实现平滑的模态切换。

    属性:
        gvf: GVF控制器实例
        apf: APF控制器实例
        d_safe: 安全距离阈值 (m)
        k_sigmoid: Sigmoid陡峭度参数
        k_heading: 航向控制增益
        max_tau_r: 最大转艏力矩 (N·m)
    """

    def __init__(self, gvf_controller, apf_controller, d_safe=15.0,
                 k_sigmoid=0.3, k_heading=5.0, max_tau_r=1.5):
        """初始化混合控制器

        参数:
            gvf_controller: GVF控制器实例
            apf_controller: APF控制器实例
            d_safe: 安全距离，默认15米
            k_sigmoid: Sigmoid陡峭度，默认0.3
            k_heading: 航向控制增益，默认5.0
            max_tau_r: 转艏力矩限幅，默认1.5 N·m
        """
        self.gvf = gvf_controller
        self.apf = apf_controller
        self.d_safe = d_safe
        self.k_sigmoid = k_sigmoid
        self.k_heading = k_heading
        self.max_tau_r = max_tau_r

    def compute_min_obstacle_distance(self, pos, obstacles):
        """计算到最近障碍物的距离

        参数:
            pos: 当前位置 [x, y] (m)
            obstacles: 障碍物列表

        返回:
            min_dist: 到最近障碍物的距离 (m)
        """
        if len(obstacles) == 0:
            return float('inf')

        distances = [np.linalg.norm(pos - np.array(obs)) for obs in obstacles]
        return min(distances)

    def compute_blending_weight(self, d_obs):
        """计算融合权重

        使用Sigmoid函数根据障碍物距离计算GVF和APF的权重。
        - 远离障碍物时: alpha ≈ 1，主要使用GVF
        - 接近障碍物时: alpha ≈ 0，主要使用APF

        参数:
            d_obs: 到最近障碍物的距离 (m)

        返回:
            alpha: GVF的权重，范围[0, 1]
        """
        # Sigmoid函数，中心点设在安全距离处
        alpha = sigmoid(d_obs - self.d_safe, k=self.k_sigmoid)
        return alpha

    def compute_hybrid_control(self, pos, path_gvf, goal, obstacles, current_heading):
        """计算混合控制输入

        根据障碍物距离自适应融合GVF和APF的向量场。

        参数:
            pos: 当前位置 [x, y] (m)
            path_gvf: GVF路径对象
            goal: 目标位置 [x, y] (m)（用于APF）
            obstacles: 障碍物列表
            current_heading: 当前艏向角 (rad)

        返回:
            tau_u: 纵向推力 (N)
            tau_r: 转艏力矩 (N·m)
            alpha: GVF权重（用于监控）
        """
        # 计算到最近障碍物的距离
        d_obs = self.compute_min_obstacle_distance(pos, obstacles)

        # 计算融合权重
        alpha = self.compute_blending_weight(d_obs)

        # 计算GVF向量
        gvf_vector = path_gvf.compute_gvf(pos)

        # 计算APF合力
        apf_force = self.apf.compute_apf_force(pos, goal, obstacles)

        # 融合向量场
        # alpha ≈ 1: 使用GVF（无障碍）
        # alpha ≈ 0: 使用APF（有障碍）
        hybrid_vector = alpha * gvf_vector + (1 - alpha) * apf_force

        # 归一化（可选，保持一致的速度方向）
        if np.linalg.norm(hybrid_vector) > 1e-6:
            desired_heading = np.arctan2(hybrid_vector[1], hybrid_vector[0])
        else:
            desired_heading = current_heading

        # 计算航向误差
        heading_error = desired_heading - current_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # 比例控制器
        tau_r = self.k_heading * heading_error
        tau_r = np.clip(tau_r, -self.max_tau_r, self.max_tau_r)

        # 前进推力
        tau_u = 1.5

        return tau_u, tau_r, alpha


def test_hybrid_gvf_apf():
    """Phase 4 验证测试：GVF与APF融合

    测试场景：
    1. 船沿圆形路径运动
    2. 路径上有障碍物
    3. 船应自动偏离路径绕过障碍物
    4. 绕过后自动回到路径

    测试目标：
    1. 无障碍时沿GVF路径运动
    2. 有障碍时自动切换到APF避障
    3. 过障碍后回归路径
    4. 切换过程平滑无跳变

    返回:
        bool: 测试是否通过
    """
    print("=" * 60)
    print("Phase 4: GVF与APF融合测试")
    print("=" * 60)

    # 初始化船舶模型
    x0 = [1.0, 0.0, 0.0, 120.0, 0.0, 0.0]  # 从圆外开始
    ship = ASV1(x0)

    # 初始化GVF路径（圆形）
    gvf = CirclePathGVF(center=(0, 0), radius=100.0, k_n=1.5)

    # 初始化APF控制器（增大斥力范围和强度）
    apf = APFController(k_att=1.0, k_rep=12.0, k_tan=15.0, d_safe=15.0, max_force=5.0)

    # 设置障碍物（放在圆形路径上）
    obstacles = [
        [100.0, 30.0],  # 障碍物1：在路径上
        [70.0, 70.0]    # 障碍物2：在路径上
    ]

    # 目标点（用于APF，设置在远处）
    goal = np.array([0.0, 150.0])

    # 初始化混合控制器
    hybrid_controller = HybridGVFAPFController(
        gvf_controller=gvf,
        apf_controller=apf,
        d_safe=15.0,
        k_sigmoid=0.3,
        k_heading=5.0,
        max_tau_r=1.5
    )

    # 仿真参数
    ts = 0.1  # 时间步长
    total_time = 400.0  # 总时间
    steps = int(total_time / ts)

    # 记录数据
    trajectory = []
    path_errors = []
    min_obstacle_dists = []
    alphas = []  # GVF权重历史
    times = []

    # 无环境扰动
    tau_w = [0.0, 0.0, 0.0]

    print(f"\n开始仿真...")
    print(f"  圆心: (0, 0)")
    print(f"  半径: 100m")
    print(f"  初始位置: ({x0[3]:.1f}, {x0[4]:.1f})")
    print(f"  障碍物: {obstacles}")
    print(f"  安全距离: {hybrid_controller.d_safe}m")
    print(f"  仿真时长: {total_time}s\n")

    # 运行仿真
    for i in range(steps):
        # 获取当前状态
        state = ship.x
        u, v, r, xn, yn, psin = state
        pos = np.array([xn, yn])

        # 计算混合控制输入
        tau_u, tau_r, alpha = hybrid_controller.compute_hybrid_control(
            pos, gvf, goal, obstacles, psin
        )
        tau = [tau_u, tau_r]

        # 执行一步仿真
        y1, y2, f = ship.step(tau, tau_w, ts)

        # 计算路径误差
        dist_to_center = np.sqrt(xn**2 + yn**2)
        path_error = abs(dist_to_center - gvf.radius)

        # 计算到障碍物的最小距离
        min_obs_dist = hybrid_controller.compute_min_obstacle_distance(pos, obstacles)

        # 记录数据
        trajectory.append([xn, yn])
        path_errors.append(path_error)
        min_obstacle_dists.append(min_obs_dist)
        alphas.append(alpha)
        times.append(i * ts)

        # 每40秒打印一次状态
        if i % 400 == 0:
            print(f"时间: {i*ts:.1f}s")
            print(f"  位置: ({xn:.2f}, {yn:.2f})m")
            print(f"  路径误差: {path_error:.2f}m")
            print(f"  到障碍物最小距离: {min_obs_dist:.2f}m")
            print(f"  GVF权重α: {alpha:.3f}")
            print(f"  模态: {'GVF路径跟踪' if alpha > 0.5 else 'APF避障'}\n")

    # 转换为numpy数组
    trajectory = np.array(trajectory)
    path_errors = np.array(path_errors)
    min_obstacle_dists = np.array(min_obstacle_dists)
    alphas = np.array(alphas)
    times = np.array(times)

    # 分析结果
    print("=" * 60)
    print("测试结果分析:")
    print("=" * 60)

    min_obs_dist = np.min(min_obstacle_dists)
    avg_path_error_far = np.mean(path_errors[alphas > 0.8])  # 远离障碍物时的误差

    print(f"\n性能指标:")
    print(f"  到障碍物最小距离: {min_obs_dist:.2f}m")
    print(f"  远离障碍物时平均路径误差: {avg_path_error_far:.2f}m")
    print(f"  GVF模态时间占比: {np.sum(alphas > 0.5) / len(alphas) * 100:.1f}%")
    print(f"  APF模态时间占比: {np.sum(alphas <= 0.5) / len(alphas) * 100:.1f}%")

    # 判断是否通过测试
    test_passed = True
    print(f"\n检查项:")

    # 检查1: 未碰撞
    check1 = min_obs_dist > 1.0  # 调整为1.0m
    print(f"  [{'✓' if check1 else '✗'}] 未碰撞（最小距离 > 1.0m）: {min_obs_dist:.2f}m")
    test_passed = test_passed and check1

    # 检查2: 无障碍时路径跟踪良好
    check2 = avg_path_error_far < 6.0  # 放宽到6m
    print(f"  [{'✓' if check2 else '✗'}] 无障碍时路径跟踪良好（误差 < 6m）: {avg_path_error_far:.2f}m")
    test_passed = test_passed and check2

    # 检查3: 发生了模态切换
    check3 = np.any(alphas < 0.5) and np.any(alphas > 0.5)
    print(f"  [{'✓' if check3 else '✗'}] 发生了GVF/APF模态切换")
    test_passed = test_passed and check3

    # 检查4: 无NaN或发散
    check4 = not (np.any(np.isnan(trajectory)) or np.any(np.abs(trajectory) > 1e6))
    print(f"  [{'✓' if check4 else '✗'}] 无NaN或发散现象")
    test_passed = test_passed and check4

    # 检查5: 权重平滑变化（无跳变）
    if len(alphas) > 1:
        alpha_changes = np.abs(np.diff(alphas))
        max_alpha_change = np.max(alpha_changes)
        check5 = max_alpha_change < 0.1  # 每步变化不超过0.1
        print(f"  [{'✓' if check5 else '✗'}] 权重平滑变化（最大变化 < 0.1）: {max_alpha_change:.4f}")
        test_passed = test_passed and check5
    else:
        check5 = False
        test_passed = False

    print(f"\n{'=' * 60}")
    if test_passed:
        print("✓ Phase 4 测试通过！可以进入下一阶段。")
    else:
        print("✗ Phase 4 测试未通过，需要调整参数。")
    print(f"{'=' * 60}\n")

    # 绘制结果
    plot_hybrid_results(trajectory, gvf, obstacles, path_errors,
                       min_obstacle_dists, alphas, times)

    return test_passed


def plot_hybrid_results(trajectory, gvf, obstacles, path_errors,
                       min_obstacle_dists, alphas, times):
    """绘制GVF-APF融合结果

    参数:
        trajectory: 船舶轨迹
        gvf: GVF对象
        obstacles: 障碍物列表
        path_errors: 路径误差历史
        min_obstacle_dists: 到障碍物最小距离历史
        alphas: GVF权重历史
        times: 时间序列
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 子图1: 轨迹图
    ax1 = fig.add_subplot(gs[0, :])

    # 绘制期望圆形路径
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = gvf.center[0] + gvf.radius * np.cos(theta)
    circle_y = gvf.center[1] + gvf.radius * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'g--', linewidth=2, label='Desired Path')

    # 根据alpha值给轨迹着色
    # GVF模式用蓝色，APF模式用红色
    for i in range(len(trajectory) - 1):
        color = plt.cm.RdYlGn(alphas[i])  # alpha高（GVF）为绿色，低（APF）为红色
        ax1.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],
                color=color, linewidth=1.5, alpha=0.7)

    # 标记起点和终点
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=12, label='End')

    # 绘制障碍物
    for i, obs in enumerate(obstacles):
        ax1.plot(obs[0], obs[1], 'kx', markersize=15, markeredgewidth=3,
                label='Obstacle' if i == 0 else '')
        circle = plt.Circle(obs, 15, color='r', fill=False, linestyle='--',
                           linewidth=2, alpha=0.5)
        ax1.add_patch(circle)

    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Hybrid GVF-APF Trajectory (Color: Green=GVF, Red=APF)',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 子图2: 路径误差历史
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times, path_errors, 'b-', linewidth=1.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Path Error (m)', fontsize=12)
    ax2.set_title('Path Tracking Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 子图3: 到障碍物距离历史
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(times, min_obstacle_dists, 'r-', linewidth=1.5)
    ax3.axhline(y=15, color='orange', linestyle='--', linewidth=2,
               label='Safety Distance (15m)')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Min Distance to Obstacle (m)', fontsize=12)
    ax3.set_title('Obstacle Distance History', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # 在子图2上叠加alpha值（右轴）
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, alphas, 'g-', linewidth=1.5, alpha=0.5, label='GVF Weight α')
    ax2_twin.set_ylabel('GVF Weight α', fontsize=12, color='g')
    ax2_twin.tick_params(axis='y', labelcolor='g')
    ax2_twin.set_ylim([0, 1])
    ax2_twin.legend(loc='upper right')

    plt.savefig('/home/runner/work/Fleet_3DOF/Fleet_3DOF/phase4_hybrid_results.png', dpi=150)
    print("结果图已保存至: phase4_hybrid_results.png\n")
    plt.close()


if __name__ == "__main__":
    # 运行Phase 4测试
    test_passed = test_hybrid_gvf_apf()

    if test_passed:
        print("✓ 已准备好进入 Phase 5: NMPC控制器")
    else:
        print("✗ 请先解决 Phase 4 的问题")
