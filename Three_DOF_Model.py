import numpy as np


def euler2(dx, x, ts):
    """与原Matlab完全一致的欧拉积分函数，支持标量和向量输入"""
    return x + dx * ts


class ASV1:
    """三自由度自主水面船(ASV)动力学模型
    完全复现原Matlab ASV1.m的所有逻辑，包括执行器延迟、输入限幅和水动力计算
    """

    def __init__(self, x0):
        """初始化模型
        Args:
            x0: 初始状态向量 [u, v, r, xn, yn, psin]
                u: 纵向速度(m/s)
                v: 横向速度(m/s)
                r: 艏向角速度(rad/s)
                xn: 北向位置(m)
                yn: 东向位置(m)
                psin: 艏向角(rad)
        """
        # 持久化状态变量（对应原Matlab的persistent）
        self.x = np.array(x0, dtype=np.float64)
        self.tu1 = 0.0  # 推力执行器一阶惯性状态
        self.tr1 = 0.0  # 转艏力矩执行器一阶惯性状态

        # 船舶固定动力学参数（与原代码完全一致）
        self.m11 = 25.8
        self.m22 = 33.8
        self.m33 = 2.76

        # 控制输入限幅参数
        self.tu_max = 2.0
        self.tr_max = 1.5

        # 执行器一阶惯性时间常数
        self.actuator_tau = 0.1

    def step(self, tau, tau_w, ts):
        """单步仿真
        Args:
            tau: 控制输入向量 [tu, tr]
                tu: 期望推力(N)
                tr: 期望转艏力矩(N·m)
            tau_w: 环境扰动向量 [tu_w, tv_w, tr_w]
                tu_w: 纵向扰动力(N)
                tv_w: 横向扰动力(N)
                tr_w: 艏向扰动力矩(N·m)
            ts: 仿真时间步长(s)
        Returns:
            y1: 当前状态向量 [u, v, r, xn, yn, psin]
            y2: 实际输出控制量 [tu_actual, tr_actual]（经过执行器延迟）
            f: 加速度向量 [u_acc, r_acc]
        """
        # 提取环境扰动分量
        tu_w, tv_w, tr_w = tau_w

        # 原代码注释掉的艏向角速度限幅，需要时可打开
        # if abs(self.x[2]) >= 0.8:
        #     self.x[2] = np.sign(self.x[2]) * 0.8

        # 控制输入限幅（与原代码完全一致：推力非对称0~2，力矩对称±1.5）
        tu, tr = tau
        tu = np.clip(tu, 0, self.tu_max)
        tr = np.clip(tr, -self.tr_max, self.tr_max)

        # 执行器一阶惯性环节（模拟响应延迟）
        tu1_dot = (tu - self.tu1) / self.actuator_tau
        tr1_dot = (tr - self.tr1) / self.actuator_tau
        self.tu1 = euler2(tu1_dot, self.tu1, ts)
        self.tr1 = euler2(tr1_dot, self.tr1, ts)
        tu_actual = self.tu1
        tr_actual = self.tr1

        # 提取当前状态分量
        u, v, r, xn, yn, psin = self.x

        # 水动力计算（与原代码公式完全一致）
        fu = (-5.87 * u**3
              - 1.33 * abs(u) * u
              - 0.72 * u
              + self.m22 * v * r
              + 1.0948 * r**2)

        fv = (-36.5 * abs(v) * v
              - 0.8896 * v
              - 0.805 * v * abs(r)
              - self.m11 * u * r)

        fr = (-0.75 * abs(r) * r
              - 1.90 * r
              + 0.08 * abs(v) * r
              + (self.m11 - self.m22) * u * v
              - 1.0948 * u * r)

        # 旋转矩阵（北东坐标系到船体坐标系）
        R = np.array([
            [np.cos(psin), -np.sin(psin), 0],
            [np.sin(psin), np.cos(psin), 0],
            [0, 0, 1]
        ])

        # 状态导数计算
        x1 = np.array([u, v, r])
        x2_dot = R @ x1  # 位置和艏向导数

        x_dot = np.array([
            (fu + tu_actual + tu_w) / self.m11,  # 纵向加速度
            (fv + tv_w) / self.m22,              # 横向加速度
            (fr + tr_actual + tr_w) / self.m33,  # 艏向角加速度
            x2_dot[0],                           # 北向速度
            x2_dot[1],                           # 东向速度
            x2_dot[2]                            # 艏向角速度
        ])

        # 欧拉积分更新状态
        self.x = euler2(x_dot, self.x, ts)

        # 构造返回值（与原Matlab输出完全一致）
        y1 = self.x.copy()
        y2 = np.array([tu_actual, tr_actual])
        f = np.array([
            (fu + tu_w) / self.m11,
            (fr + tr_w) / self.m33
        ])

        return y1, y2, f


# ------------------------------
# 使用示例（与原Matlab调用方式对应）
# ------------------------------
if __name__ == "__main__":
    # 初始化模型（初始状态全零）
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    asv_model = ASV1(x0)

    # 仿真参数
    ts = 0.01  # 时间步长10ms
    total_time = 10.0  # 总仿真时间10s
    steps = int(total_time / ts)

    # 控制输入和扰动
    tau = [1.0, 0.2]  # 期望推力1N，期望转艏力矩0.2N·m
    tau_w = [0.0, 0.0, 0.0]  # 无环境扰动

    # 运行仿真
    for i in range(steps):
        y1, y2, f = asv_model.step(tau, tau_w, ts)

        # 每100步打印一次状态
        if i % 100 == 0:
            print(f"时间: {i*ts:.1f}s")
            print(f"  位置: ({y1[3]:.2f}, {y1[4]:.2f})m")
            print(f"  艏向角: {np.rad2deg(y1[5]):.1f}°")
            print(f"  实际控制量: 推力={y2[0]:.2f}N, 力矩={y2[1]:.2f}N·m\n")
