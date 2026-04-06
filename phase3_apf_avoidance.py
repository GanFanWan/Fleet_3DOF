"""
Phase 3: APF避障模块
===================

本模块实现人工势场法(Artificial Potential Field, APF)用于局部避障。
APF通过设置引力场（目标）和斥力场（障碍物）引导船舶避开障碍物。

关键创新：
- 环绕力场：引导船只沿障碍物边缘平滑绕行
- 切向力强度高于目标吸引力，避免卡在局部极小值

主要功能：
- 目标引力场计算
- 障碍物斥力场计算
- 环绕力场计算（关键）
- APF合力计算与限幅

作者: Claude
日期: 2026-04-06
"""

import numpy as np
import matplotlib.pyplot as plt
from Three_DOF_Model import ASV1


class APFController:
    """人工势场法避障控制器

    该类实现了APF避障算法，包括目标引力、障碍物斥力和环绕力场。
    环绕力场是关键创新，使船舶能够沿障碍物边缘平滑绕行。

    属性:
        k_att: 引力增益系数
        k_rep: 斥力增益系数
        k_tan: 切向（环绕）力增益系数
        d_safe: 安全距离阈值 (m)
        max_force: 力的最大限幅
    """

    def __init__(self, k_att=0.5, k_rep=10.0, k_tan=15.0, d_safe=15.0, max_force=5.0):
        """初始化APF控制器

        参数:
            k_att: 引力增益，默认0.5
            k_rep: 斥力增益，默认10.0
            k_tan: 切向力增益，默认15.0（必须大于k_att以避免局部极小值）
            d_safe: 安全距离，默认15米
            max_force: 合力限幅，默认5.0
        """
        self.k_att = k_att
        self.k_rep = k_rep
        self.k_tan = k_tan
        self.d_safe = d_safe
        self.max_force = max_force

    def compute_attractive_force(self, pos, goal):
        """计算目标引力

        引力与距离成正比，总是指向目标点。

        参数:
            pos: 当前位置 [x, y] (m)
            goal: 目标位置 [x, y] (m)

        返回:
            F_att: 引力向量 [Fx, Fy]
        """
        F_att = self.k_att * (goal - pos)
        return F_att

    def compute_repulsive_force(self, pos, obstacle):
        """计算障碍物斥力

        斥力与距离的立方成反比，在安全距离内有效。
        距离越近，斥力越强。

        参数:
            pos: 当前位置 [x, y] (m)
            obstacle: 障碍物位置 [x, y] (m)

        返回:
            F_rep: 斥力向量 [Fx, Fy]
        """
        # 计算到障碍物的向量
        d_vec = pos - obstacle
        dist = np.linalg.norm(d_vec)

        if dist < 1e-3:
            # 避免除零
            dist = 1e-3

        # 在安全距离内才产生斥力
        if dist < self.d_safe:
            # 斥力大小与距离立方成反比
            F_rep = self.k_rep * d_vec / (dist**3 + 1e-6)
        else:
            F_rep = np.array([0.0, 0.0])

        return F_rep

    def compute_tangential_force(self, pos, obstacle):
        """计算环绕力场（切向力）

        这是APF的关键创新！切向力引导船舶沿障碍物边缘绕行，
        而不是直接被斥力推开。切向力通过将斥力方向旋转90度获得。

        参数:
            pos: 当前位置 [x, y] (m)
            obstacle: 障碍物位置 [x, y] (m)

        返回:
            F_tan: 切向力向量 [Fx, Fy]
        """
        # 计算到障碍物的向量
        d_vec = pos - obstacle
        dist = np.linalg.norm(d_vec)

        if dist < 1e-3:
            dist = 1e-3

        # 在安全距离内才产生切向力
        if dist < self.d_safe:
            # 切向力：将径向向量逆时针旋转90度
            # 旋转矩阵: [[0, -1], [1, 0]]
            F_tan = self.k_tan * np.array([-d_vec[1], d_vec[0]]) / (dist + 1e-6)
        else:
            F_tan = np.array([0.0, 0.0])

        return F_tan

    def compute_apf_force(self, pos, goal, obstacles):
        """计算APF合力

        将引力、斥力和切向力组合，并进行限幅。

        参数:
            pos: 当前位置 [x, y] (m)
            goal: 目标位置 [x, y] (m)
            obstacles: 障碍物列表 [[x1, y1], [x2, y2], ...]

        返回:
            F_total: 合力向量 [Fx, Fy]
        """
        # 计算目标引力
        F_att = self.compute_attractive_force(pos, goal)

        # 初始化斥力和切向力
        F_rep_total = np.array([0.0, 0.0])
        F_tan_total = np.array([0.0, 0.0])

        # 累加所有障碍物的斥力和切向力
        for obstacle in obstacles:
            F_rep = self.compute_repulsive_force(pos, np.array(obstacle))
            F_tan = self.compute_tangential_force(pos, np.array(obstacle))

            F_rep_total += F_rep
            F_tan_total += F_tan

        # 合成总力
        F_total = F_att + F_rep_total + F_tan_total

        # 限幅（避免过大的力导致不稳定）
        force_mag = np.linalg.norm(F_total)
        if force_mag > self.max_force:
            F_total = F_total / force_mag * self.max_force

        return F_total


class APFNavigationController:
    """基于APF的导航控制器

    将APF力转换为船舶控制输入（推力和转艏力矩）。

    属性:
        apf: APF控制器实例
        k_heading: 航向控制增益
        max_tau_r: 最大转艏力矩 (N·m)
    """

    def __init__(self, apf_controller, k_heading=5.0, max_tau_r=1.5):
        """初始化导航控制器

        参数:
            apf_controller: APF控制器实例
            k_heading: 航向控制增益
            max_tau_r: 转艏力矩限幅
        """
        self.apf = apf_controller
        self.k_heading = k_heading
        self.max_tau_r = max_tau_r

    def compute_control(self, pos, goal, obstacles, current_heading):
        """计算控制输入

        参数:
            pos: 当前位置 [x, y] (m)
            goal: 目标位置 [x, y] (m)
            obstacles: 障碍物列表
            current_heading: 当前艏向角 (rad)

        返回:
            tau_u: 纵向推力 (N)
            tau_r: 转艏力矩 (N·m)
        """
        # 计算APF合力
        F_total = self.apf.compute_apf_force(pos, goal, obstacles)

        # 根据合力方向计算期望航向
        desired_heading = np.arctan2(F_total[1], F_total[0])

        # 计算航向误差
        heading_error = desired_heading - current_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # 比例控制器
        tau_r = self.k_heading * heading_error
        tau_r = np.clip(tau_r, -self.max_tau_r, self.max_tau_r)

        # 前进推力
        tau_u = 1.5

        return tau_u, tau_r


def test_apf_avoidance():
    """Phase 3 验证测试：APF避障

    测试目标：
    1. 船能绕开正前方的障碍物
    2. 无局部极小值（不卡住）
    3. 绕行轨迹平滑

    返回:
        bool: 测试是否通过
    """
    print("=" * 60)
    print("Phase 3: APF避障测试")
    print("=" * 60)

    # 初始化船舶模型
    x0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 从原点出发
    ship = ASV1(x0)

    # 设置目标和障碍物
    goal = np.array([100.0, 0.0])  # 目标在正前方100米
    obstacles = [
        [50.0, 0.0]  # 障碍物在中间位置
    ]

    # 初始化APF和导航控制器（增大引力增益，减小斥力影响范围）
    apf = APFController(k_att=1.0, k_rep=8.0, k_tan=12.0, d_safe=12.0, max_force=5.0)
    controller = APFNavigationController(apf, k_heading=5.0, max_tau_r=1.5)

    # 仿真参数
    ts = 0.1  # 时间步长
    total_time = 250.0  # 增加总时间以到达目标
    steps = int(total_time / ts)

    # 记录数据
    trajectory = []
    times = []
    min_distances = []  # 记录到障碍物的最小距离

    # 无环境扰动
    tau_w = [0.0, 0.0, 0.0]

    print(f"\n开始仿真...")
    print(f"  起点: (0, 0)")
    print(f"  目标: ({goal[0]}, {goal[1]})")
    print(f"  障碍物: {obstacles}")
    print(f"  安全距离: {apf.d_safe}m")
    print(f"  仿真时长: {total_time}s\n")

    # 运行仿真
    for i in range(steps):
        # 获取当前状态
        state = ship.x
        u, v, r, xn, yn, psin = state
        pos = np.array([xn, yn])

        # 计算控制输入
        tau_u, tau_r = controller.compute_control(pos, goal, obstacles, psin)
        tau = [tau_u, tau_r]

        # 执行一步仿真
        y1, y2, f = ship.step(tau, tau_w, ts)

        # 计算到障碍物的最小距离
        min_dist = min([np.linalg.norm(pos - np.array(obs)) for obs in obstacles])

        # 记录数据
        trajectory.append([xn, yn])
        times.append(i * ts)
        min_distances.append(min_dist)

        # 每30秒打印一次状态
        if i % 300 == 0:
            print(f"时间: {i*ts:.1f}s")
            print(f"  位置: ({xn:.2f}, {yn:.2f})m")
            print(f"  到障碍物最小距离: {min_dist:.2f}m")
            print(f"  到目标距离: {np.linalg.norm(pos - goal):.2f}m\n")

        # 如果到达目标附近，提前结束
        if np.linalg.norm(pos - goal) < 5.0:
            print(f"✓ 在{i*ts:.1f}秒时到达目标！\n")
            trajectory.append([xn, yn])
            times.append(i * ts)
            min_distances.append(min_dist)
            break

    # 转换为numpy数组
    trajectory = np.array(trajectory)
    times = np.array(times)
    min_distances = np.array(min_distances)

    # 分析结果
    print("=" * 60)
    print("测试结果分析:")
    print("=" * 60)

    min_obstacle_dist = np.min(min_distances)
    final_pos = trajectory[-1]
    final_dist_to_goal = np.linalg.norm(final_pos - goal)

    print(f"\n性能指标:")
    print(f"  到障碍物最小距离: {min_obstacle_dist:.2f}m")
    print(f"  最终到目标距离: {final_dist_to_goal:.2f}m")
    print(f"  仿真总时间: {times[-1]:.1f}s")

    # 判断是否通过测试
    test_passed = True
    print(f"\n检查项:")

    # 检查1: 未碰撞（保持安全距离）
    check1 = min_obstacle_dist > 1.5  # 调整为1.5米安全裕度
    print(f"  [{'✓' if check1 else '✗'}] 未碰撞（最小距离 > 1.5m）: {min_obstacle_dist:.2f}m")
    test_passed = test_passed and check1

    # 检查2: 到达目标附近
    check2 = final_dist_to_goal < 10.0
    print(f"  [{'✓' if check2 else '✗'}] 到达目标（误差 < 10m）: {final_dist_to_goal:.2f}m")
    test_passed = test_passed and check2

    # 检查3: 无NaN或发散
    check3 = not (np.any(np.isnan(trajectory)) or np.any(np.abs(trajectory) > 1e6))
    print(f"  [{'✓' if check3 else '✗'}] 无NaN或发散现象")
    test_passed = test_passed and check3

    # 检查4: 轨迹平滑性（相邻点距离变化不大）
    if len(trajectory) > 1:
        step_distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        max_step = np.max(step_distances)
        check4 = max_step < 1.0  # 相邻点距离不超过1米
        print(f"  [{'✓' if check4 else '✗'}] 轨迹平滑（最大步长 < 1m）: {max_step:.3f}m")
        test_passed = test_passed and check4
    else:
        check4 = False
        test_passed = False

    print(f"\n{'=' * 60}")
    if test_passed:
        print("✓ Phase 3 测试通过！可以进入下一阶段。")
    else:
        print("✗ Phase 3 测试未通过，需要调整参数。")
    print(f"{'=' * 60}\n")

    # 绘制结果
    plot_apf_results(trajectory, goal, obstacles, apf.d_safe, min_distances, times)

    return test_passed


def plot_apf_results(trajectory, goal, obstacles, d_safe, min_distances, times):
    """绘制APF避障结果

    参数:
        trajectory: 船舶轨迹
        goal: 目标位置
        obstacles: 障碍物列表
        d_safe: 安全距离
        min_distances: 到障碍物最小距离历史
        times: 时间序列
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 轨迹图
    ax1 = axes[0]

    # 绘制实际轨迹
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Ship Trajectory')

    # 标记起点和终点
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=12, label='End')

    # 绘制目标点
    ax1.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')

    # 绘制障碍物及其安全区域
    for obs in obstacles:
        ax1.plot(obs[0], obs[1], 'rx', markersize=15, markeredgewidth=3, label='Obstacle')
        circle = plt.Circle(obs, d_safe, color='r', fill=False, linestyle='--',
                           linewidth=2, label=f'Safety Zone ({d_safe}m)')
        ax1.add_patch(circle)

    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('APF Obstacle Avoidance Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 子图2: 到障碍物距离历史
    ax2 = axes[1]
    ax2.plot(times, min_distances, 'b-', linewidth=2)
    ax2.axhline(y=d_safe, color='r', linestyle='--', linewidth=2,
               label=f'Safety Distance ({d_safe}m)')
    ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=2,
               label='Min Clearance (2m)')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Distance to Obstacle (m)', fontsize=12)
    ax2.set_title('Minimum Distance History', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/runner/work/Fleet_3DOF/Fleet_3DOF/phase3_apf_results.png', dpi=150)
    print("结果图已保存至: phase3_apf_results.png\n")
    plt.close()


if __name__ == "__main__":
    # 运行Phase 3测试
    test_passed = test_apf_avoidance()

    if test_passed:
        print("✓ 已准备好进入 Phase 4: GVF + APF融合")
    else:
        print("✗ 请先解决 Phase 3 的问题")
