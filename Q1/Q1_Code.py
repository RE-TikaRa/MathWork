import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from typing import Optional, Tuple
import matplotlib

# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['font.monospace'] = ['Microsoft YaHei', 'SimHei', 'Fira Code', 'Consolas', 'DejaVu Sans Mono']
matplotlib.rcParams['axes.unicode_minus'] = False


class RealTimeProjection:
    """
    Q1 场景：
    - M1 以 300 m/s 朝 FO 直线运动。
    - RO 抽象为球：中心(0,200,5)，半径√74。
    - FY1 投放烟雾干扰：半径固定 10 m，t_det=5.1 s 起爆，起爆后下沉 3 m/s 持续 20 s。
    - 实时可视化切锥、烟团、以及“完全遮蔽”时间段，并给出关键几何数据。
    """

    def __init__(self) -> None:
        # 场景定义
        self.M1_start = np.array([20000.0, 0.0, 2000.0])
        self.FO = np.array([0.0, 0.0, 0.0])
        self.RO_center = np.array([0.0, 200.0, 5.0])
        self.RO_radius = float(np.sqrt(74.0))
        self.FY1 = np.array([17800.0, 0.0, 1800.0])

        # 运动（M1 直线朝 FO）
        self.velocity = 300.0
        self.direction = (self.FO - self.M1_start) / np.linalg.norm(self.FO - self.M1_start)

        # 时间
        self.total_time = float(np.linalg.norm(self.FO - self.M1_start) / self.velocity)
        self.dt = 0.1

        # 图与控件
        self.fig = None
        self.ax3d = None
        self.ax_area = None
        self.ax_dist = None
        self.ax_info = None
        self.btn_play = None
        self.btn_pause = None
        self.btn_reset = None
        self.slider = None
        self.time_cursor_area = None
        self.time_cursor_dist = None
        self.ani = None
        self.running = True
        self.current_frame = 0
        self._updating_slider = False

        # 分析曲线缓存
        self._times = None
        self._areas = None
        self._dists = None

        # 显示与样式
        self.show_cone = True
        self.show_sphere = True
        self.show_axis = True
        self.show_rim = True
        self.show_smoke = True
        self.show_fy1 = True
        self.cone_alpha = 0.6
        # 3D 叠加信息开关
        self.show_overlay = True

        # 视角与播放控制
        self._default_view = (30.0, -60.0)
        self._preserve_view = True  # 播放时保留用户视角
        self.play_speed = 1.0       # 播放速度倍率
        self._frame_accum = 0.0     # 累积步进，支持非整数速度
        # 额外控件句柄
        self.btn_lock_view = None
        self.btn_view_reset = None
        self.speed_slider = None

        # 信息面板句柄
        self.info_text = None
        self.status_text = None

        # —— 烟团与完全遮蔽设置 ——
        self.g = 9.8  # m/s^2
        self.smoke_radius = 10.0  # m 固定
        self.fy_speed = 120.0  # m/s
        self.t_drop = 1.5  # s
        self.t_det = self.t_drop + 3.6  # s（起爆）
        self.smoke_active_duration = 20.0  # 起爆后有效 20 s

        # FY1 航迹（等高度指向原点）
        self.FY1_start = self.FY1.copy()
        fy_target_same_alt = np.array([0.0, 0.0, self.FY1_start[2]])
        fy_dir = fy_target_same_alt - self.FY1_start
        fy_dir = fy_dir / np.linalg.norm(fy_dir)
        self.fy_dir = fy_dir

        # 投放瞬间位置 & 起爆位置（投放后自由落体 + 初速为 FY1 速度）
        self.pos_drop = self.FY1_start + self.fy_dir * (self.fy_speed * self.t_drop)
        # 起爆时刻位置（投弹后 3.6s）
        det_dt = self.t_det - self.t_drop
        v0 = self.fy_speed * self.fy_dir
        self.pos_det = self.pos_drop + v0 * det_dt + np.array([0.0, 0.0, -0.5 * self.g * det_dt * det_dt])

        # 遮蔽缓存
        self._occluded_ts = None
        self._occluded_flags = None
        self._occluded_total = 0.0

    def get_M1_position(self, t: float) -> np.ndarray:
        return self.M1_start + self.direction * (self.velocity * float(t))

    def get_sphere_silhouette(self, view_point: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        to_center = self.RO_center - view_point
        dist = float(np.linalg.norm(to_center))
        if dist <= self.RO_radius:
            return None, None
        sin_theta = self.RO_radius / dist
        cos_theta = float(np.sqrt(max(0.0, 1.0 - sin_theta**2)))
        center = view_point + to_center * cos_theta
        radius = self.RO_radius * sin_theta
        view_dir = to_center / dist
        if abs(view_dir[2]) < 0.9:
            v1 = np.cross(view_dir, np.array([0.0, 0.0, 1.0]))
        else:
            v1 = np.cross(view_dir, np.array([1.0, 0.0, 0.0]))
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(view_dir, v1)
        v2 = v2 / np.linalg.norm(v2)
        theta = np.linspace(0.0, 2.0 * np.pi, 180)
        pts = center + radius * (np.cos(theta)[:, None] * v1 + np.sin(theta)[:, None] * v2)
        return pts, center

    def _smoke_center(self, t: float) -> Optional[np.ndarray]:
        if t < self.t_det or t > self.t_det + self.smoke_active_duration:
            return None
        dz = -3.0 * (t - self.t_det)
        return self.pos_det + np.array([0.0, 0.0, dz])

    @staticmethod
    def _ray_sphere_near(P: np.ndarray, u: np.ndarray, C: np.ndarray, r: float) -> Optional[float]:
        oc = P - C
        b = 2.0 * float(np.dot(u, oc))
        c = float(np.dot(oc, oc)) - r * r
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        sqrt_disc = float(np.sqrt(max(0.0, disc)))
        s1 = (-b - sqrt_disc) / 2.0
        s2 = (-b + sqrt_disc) / 2.0
        candidates = [s for s in (s1, s2) if s >= 0.0]
        if not candidates:
            return None
        return float(min(candidates))

    @staticmethod
    def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
        a = u / np.linalg.norm(u)
        b = v / np.linalg.norm(v)
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.arccos(dot))

    def is_fully_occluded(self, t: float, samples: int = 48, eps: float = 1e-6) -> bool:
        """
        解析法（无采样）：
        记 apex=P=M1(t)，目标球心 C，半径 R；烟团中心 S，半径 r。
        设 d=|C-P|，α=arcsin(R/d)（切锥半角），L=√(d^2−R^2)（至切点的母线长度）。
        设 ds=|S-P|，β=arcsin(r/ds)（烟团角半径），θ=∠(C-P, S-P)。
        完全遮蔽的必要且充分条件：
          1) θ+α ≤ β（锥体全边界方向均与烟团相交）；且
          2) 在最不利边界方向 ψ=θ+α 处的最近交点距离 t_sm(ψ) ≤ L。
        其中 t_sm(ψ)= ds·[cosψ − √( (r/ds)^2 − sin^2ψ )]（ψ≤β 时定义）。
        补充充分条件：P 在烟内 或 目标球完全包裹于烟内。
        """
        S = self._smoke_center(t)
        if S is None:
            return False
        P = self.get_M1_position(t)
        C = self.RO_center
        R = self.RO_radius
        r = self.smoke_radius

        # 充分情形
        if np.linalg.norm(P - S) <= r + eps:
            return True
        if np.linalg.norm(C - S) + R <= r + eps:
            return True

        # 目标切锥参数
        d = float(np.linalg.norm(C - P))
        if d <= R + eps:
            return False  # 位于目标球内，无切锥意义
        alpha = float(np.arcsin(R / d))
        L = float(np.sqrt(max(0.0, d * d - R * R)))

        # 烟团角度与中心相对参数
        ds = float(np.linalg.norm(S - P))
        if ds <= r + eps:  # apex 在烟内
            return True
        rho = float(r / ds)
        rho = min(1.0, max(0.0, rho))
        beta = float(np.arcsin(rho))
        theta = self._angle_between(C - P, S - P)

        # 角域完全覆盖：确保边界圆上所有方向都与烟团相交
        if theta + alpha > beta + 1e-9:
            return False

        # 最不利边界方向 ψ=θ+α（此时交点最远），计算 t_sm(ψ)
        psi = theta + alpha
        # 数值稳健：限制在 [0, beta]
        psi = min(max(0.0, psi), beta)
        cos_psi = float(np.cos(psi))
        sin_psi = float(np.sin(psi))
        under = rho * rho - sin_psi * sin_psi
        if under < -1e-12:
            return False  # 理论不应发生（因 ψ≤β），数值兜底
        under = max(0.0, under)
        t_sm = ds * (cos_psi - float(np.sqrt(under)))
        # 要求：烟团最近交点距离不大于切点距离 L
        return t_sm <= L + eps

    def analyze_full_occlusion(self, ts: np.ndarray):
        flags = np.array([self.is_fully_occluded(float(t)) for t in ts], dtype=bool)
        total = float(np.sum(flags) * self.dt)
        return ts, flags, total

    @staticmethod
    def _shade_occlusion(ax, ts: np.ndarray, flags: np.ndarray) -> None:
        if ts is None or flags is None:
            return
        on = False
        t_start = None
        for i in range(len(ts)):
            if flags[i] and not on:
                on = True
                t_start = ts[i]
            if (not flags[i] and on) or (on and i == len(ts) - 1):
                t_end = ts[i]
                ax.axvspan(t_start, t_end, color='green', alpha=0.15)
                on = False

    def _draw_scene(self, ax, t: float) -> None:
    # 在清空前保存用户当前视角（避免拖拽被覆盖）
        try:
            elev, azim = float(getattr(ax, 'elev', 30.0)), float(getattr(ax, 'azim', -60.0))
        except Exception:
            elev, azim = 30.0, -60.0

        ax.clear()
        M1 = self.get_M1_position(t)

        # 固定对象
        ax.scatter(*self.FO, color='red', s=80, label='FO')
        ax.scatter(*self.RO_center, color='green', s=80, label='RO Center')

        # 球体线框
        if self.show_sphere:
            u = np.linspace(0, 2 * np.pi, 24)
            v = np.linspace(0, np.pi, 16)
            x = self.RO_center[0] + self.RO_radius * np.outer(np.cos(u), np.sin(v))
            y = self.RO_center[1] + self.RO_radius * np.outer(np.sin(u), np.sin(v))
            z = self.RO_center[2] + self.RO_radius * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(x, y, z, color='green', alpha=0.2)  # type: ignore[attr-defined]

        # M1 位置
        ax.scatter(*M1, color='orange', s=100, label=f'M1 t={t:.1f}s')

        # FY1 无人机运动与投弹/弹体轨迹
        if self.show_fy1:
            # FY1 当前位置与轨迹（保持等高度朝向原点）
            fy_t = float(max(0.0, t))
            fy_pos = self.FY1_start + self.fy_dir * (self.fy_speed * fy_t)
            # FY1 轨迹从起点到当前
            tt_fy = np.linspace(0.0, fy_t, 50)
            traj_fy = self.FY1_start + self.fy_dir[None, :] * (self.fy_speed * tt_fy[:, None])
            ax.plot(traj_fy[:, 0], traj_fy[:, 1], traj_fy[:, 2], color='blue', alpha=0.6, lw=1.2, label='FY1 航迹')
            ax.scatter(*fy_pos, color='blue', s=40, label='FY1')
            # 投弹点
            ax.scatter(*self.pos_drop, color='purple', s=30, label='投弹点 t=1.5s')
            # 弹体抛物线（投弹后至当前或起爆时刻）
            if t >= self.t_drop:
                t0 = self.t_drop
                t1 = min(t, self.t_det)
                ts_seg = np.linspace(t0, t1, 50)
                dt_seg = ts_seg - t0
                pos_seg = self.pos_drop[None, :] + (self.fy_speed * self.fy_dir)[None, :] * dt_seg[:, None] \
                          + np.array([0.0, 0.0, -0.5 * self.g])[None, :] * (dt_seg[:, None] ** 2)
                ax.plot(pos_seg[:, 0], pos_seg[:, 1], pos_seg[:, 2], color='black', ls='--', lw=1.0, alpha=0.8, label='弹体轨迹')

        # 烟团球
        S = self._smoke_center(t)
        if self.show_smoke and S is not None:
            u_s = np.linspace(0, 2 * np.pi, 24)
            v_s = np.linspace(0, np.pi, 16)
            xs = S[0] + self.smoke_radius * np.outer(np.cos(u_s), np.sin(v_s))
            ys = S[1] + self.smoke_radius * np.outer(np.sin(u_s), np.sin(v_s))
            zs = S[2] + self.smoke_radius * np.outer(np.ones_like(u_s), np.cos(v_s))
            ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.5)  # type: ignore[attr-defined]
            ax.scatter(*self.pos_det, color='black', s=20, label='烟团起爆点')

        # 切锥（M1->RO）
        to_center = self.RO_center - M1
        dist = float(np.linalg.norm(to_center))
        if dist > self.RO_radius + 1e-9:
            view_dir = to_center / dist
            half_angle = float(np.arcsin(self.RO_radius / dist))
            v1 = np.cross(view_dir, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(v1) < 1e-9:
                v1 = np.cross(view_dir, np.array([1.0, 0.0, 0.0]))
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(view_dir, v1)
            v2 = v2 / np.linalg.norm(v2)
            h_max = dist * float(np.cos(half_angle))
            center_rim = M1 + view_dir * h_max
            rim_radius = self.RO_radius
            h = np.linspace(0.0, h_max, 24)
            uu = np.linspace(0.0, 2.0 * np.pi, 90)
            H, U = np.meshgrid(h, uu, indexing='ij')
            R_h = H * float(np.tan(half_angle))
            X = M1[0] + view_dir[0] * H + R_h * (np.cos(U) * v1[0] + np.sin(U) * v2[0])
            Y = M1[1] + view_dir[1] * H + R_h * (np.cos(U) * v1[1] + np.sin(U) * v2[1])
            Z = M1[2] + view_dir[2] * H + R_h * (np.cos(U) * v1[2] + np.sin(U) * v2[2])
            if self.show_cone:
                ax.plot_surface(X, Y, Z, color='red', alpha=self.cone_alpha, shade=False, linewidth=0)  # type: ignore[attr-defined]
            theta = np.linspace(0.0, 2.0 * np.pi, 180)
            rim = center_rim + rim_radius * (np.cos(theta)[:, None] * v1 + np.sin(theta)[:, None] * v2)
            if self.show_rim:
                ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], 'r-', lw=2, label='切面圆')
                ax.scatter(*center_rim, color='red', s=40, zorder=5)
            if self.show_cone:
                for i in range(0, len(theta), 10):
                    p = center_rim + rim_radius * (np.cos(theta[i]) * v1 + np.sin(theta[i]) * v2)
                    ax.plot([M1[0], p[0]], [M1[1], p[1]], [M1[2], p[2]], color='orange', alpha=0.7, lw=0.9)
        else:
            ax.text2D(0.02, 0.95, 'M1 位于球体内部：切锥无定义', transform=ax.transAxes, color='crimson')

    # 轴线
        if self.show_axis:
            ax.plot([M1[0], self.RO_center[0]], [M1[1], self.RO_center[1]], [M1[2], self.RO_center[2]],
                    color='crimson', ls='--', lw=1.2, alpha=0.8, label='轴线')

        # 轨迹
        t2 = min(float(t) + 1.0, self.total_time)
        traj_t = np.linspace(max(0.0, t2 - 1.0), t2, 50)
        traj = np.array([self.get_M1_position(tt) for tt in traj_t])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '--', color='orange', alpha=0.7, label='轨迹')

        # 视图/坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')  # type: ignore[attr-defined]
        ax.set_title(f'M1 视角下 RO 切面圆锥 (t={t:.1f}s)')
        # 3D 叠加关键参数（便于“实时展示”直观查看）
        if self.show_overlay:
            to_center = self.RO_center - M1
            d = float(np.linalg.norm(to_center))
            occluded = self.is_fully_occluded(t)
            if d > self.RO_radius:
                alpha = float(np.arcsin(self.RO_radius / d))
                alpha_deg = float(np.degrees(alpha))
                apex_deg = 2.0 * alpha_deg
                overlay = (
                    f"t={t:.2f}s  d={d:.2f}m  α={alpha_deg:.2f}°  顶角={apex_deg:.2f}°  "
                    f"遮蔽={'YES' if occluded else 'NO'}"
                )
            else:
                overlay = f"t={t:.2f}s  M1位于球内  遮蔽={'YES' if occluded else 'NO'}"
            try:
                ax.text2D(0.02, 0.98, overlay, transform=ax.transAxes, va='top', ha='left',
                          fontsize=9, family='Microsoft YaHei', bbox=dict(fc='white', alpha=0.6, ec='none'))
            except Exception:
                pass
        # 图例去重（避免重复项堆积）
        try:
            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            new_h, new_l = [], []
            for h, lb in zip(handles, labels):
                if lb not in seen and lb.strip() != '':
                    new_h.append(h)
                    new_l.append(lb)
                    seen.add(lb)
            if new_h:
                # 将图例放在参数面板内部的空白区域（红框位置）
                ax.legend(new_h, new_l, loc='center', fontsize=7, framealpha=0.95, 
                         bbox_to_anchor=(1.35, 0.6), bbox_transform=ax.transAxes)
        except Exception:
            pass

        # 恢复用户拖拽的视角（不再每帧写死）
        if self._preserve_view:
            try:
                ax.view_init(elev=elev, azim=azim)  # type: ignore[attr-defined]
            except Exception:
                pass
        # 固定显示范围
        ax.set_xlim(-1000, 21000)
        ax.set_ylim(-1000, 2000)
        ax.set_zlim(-1000, 3000)  # type: ignore[attr-defined]
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    def _build_layout(self) -> None:
        # 模仿提供的界面布局：左上3D查看器，右上参数面板，中间两个图表，底部控件
        self.fig = plt.figure(figsize=(16, 12))
        
        # 创建网格布局：3行2列
        # 第1行：左边3D查看器，右边参数面板
        # 第2行：两个图表（图1和图2）
        # 第3行：播放控制区域
        gs = self.fig.add_gridspec(3, 2, 
                                   height_ratios=[2.8, 1.8, 0.6], 
                                   width_ratios=[2.2, 1],
                                   hspace=0.25, wspace=0.15)
        
        # 左上：3D查看器
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        
        # 右上：参数信息面板
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_info.axis('off')
        
        # 中下：图1和图2
        self.ax_area = self.fig.add_subplot(gs[1, 0])
        self.ax_dist = self.fig.add_subplot(gs[1, 1])
        
        # 底部控制区域（跨两列）
        self.ax_control = self.fig.add_subplot(gs[2, :])
        self.ax_control.axis('off')
        
        try:
            # 调整边距，避免元素重叠
            self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15, hspace=0.25, wspace=0.15)
        except Exception:
            pass

        # 分析曲线 + 遮蔽预计算
        ts, angles_deg, dists = self.analyze_projection_area()
        self._times, self._areas, self._dists = ts, angles_deg, dists
        self._occluded_ts, self._occluded_flags, self._occluded_total = self.analyze_full_occlusion(ts)

        # 图1：视线半角变化
        self.ax_area.plot(ts, angles_deg, 'r-', linewidth=2)
        self.ax_area.set_title('图1 - 视线半角变化', fontsize=12, family='Microsoft YaHei')
        self.ax_area.set_xlabel('时间 (s)', fontsize=10)
        self.ax_area.set_ylabel('半角 (°)', fontsize=10)
        self.ax_area.grid(True, alpha=0.3)
        self._shade_occlusion(self.ax_area, ts, self._occluded_flags)

        # 图2：距离变化
        self.ax_dist.plot(ts, dists, 'b-', linewidth=2)
        self.ax_dist.set_title('图2 - M1到RO距离', fontsize=12, family='Microsoft YaHei')
        self.ax_dist.set_xlabel('时间 (s)', fontsize=10)
        self.ax_dist.set_ylabel('距离 (m)', fontsize=10)
        self.ax_dist.grid(True, alpha=0.3)
        self._shade_occlusion(self.ax_dist, ts, self._occluded_flags)

        self.time_cursor_area = self.ax_area.axvline(0.0, color='k', ls='--', alpha=0.7)
        self.time_cursor_dist = self.ax_dist.axvline(0.0, color='k', ls='--', alpha=0.7)

        # 设置3D查看器标题
        self.ax3d.set_title('3D Viewer - M1视角下的RO切锥投影', fontsize=14, family='Microsoft YaHei', pad=20)

        # 初始化右上角参数面板 - 调整字体大小避免重叠
        self.info_text = self.ax_info.text(0.02, 0.98, self._compose_info_text(0.0), 
                                          va='top', ha='left', fontsize=8, 
                                          family='Microsoft YaHei', 
                                          transform=self.ax_info.transAxes,
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        # 初始 3D 视角
        try:
            self.ax3d.view_init(elev=self._default_view[0], azim=self._default_view[1])  # type: ignore[attr-defined]
        except Exception:
            pass

        # 初始化一帧，立即填充信息面板/时间游标
        try:
            self._update_frame(0)
        except Exception:
            pass

        # —— 交互控件（放在底部控制区域）——
        try:
            # 播放控制按钮组 - 调整位置避免重叠
            btn_width = 0.07
            btn_height = 0.035
            btn_y = 0.05  # 提高按钮位置
            
            ax_play = self.fig.add_axes((0.08, btn_y, btn_width, btn_height))
            ax_pause = self.fig.add_axes((0.16, btn_y, btn_width, btn_height))
            ax_reset = self.fig.add_axes((0.24, btn_y, btn_width, btn_height))
            
            self.btn_play = Button(ax_play, '播放', color='lightgreen')
            self.btn_pause = Button(ax_pause, '暂停', color='lightcoral')
            self.btn_reset = Button(ax_reset, '重置', color='lightblue')
            
            self.btn_play.on_clicked(self._on_play)
            self.btn_pause.on_clicked(self._on_pause)
            self.btn_reset.on_clicked(self._on_reset)

            # 时间滑块 - 调整位置和大小
            ax_slider = self.fig.add_axes((0.35, btn_y + 0.015, 0.4, 0.025))
            self.slider = Slider(ax_slider, '时间进度', 0.0, self.total_time, valinit=0.0, valstep=self.dt)
            self.slider.on_changed(self._on_slider)

            # 倍速滑块 - 分开位置避免重叠
            ax_speed = self.fig.add_axes((0.35, btn_y - 0.025, 0.25, 0.025))
            self.speed_slider = Slider(ax_speed, '倍速', 0.1, 4.0, valinit=self.play_speed, valstep=0.1)
            self.speed_slider.on_changed(self._on_speed_slider)

            # 视角控制按钮 - 调整到右侧
            ax_lock = self.fig.add_axes((0.78, btn_y + 0.015, 0.08, btn_height))
            ax_vreset = self.fig.add_axes((0.87, btn_y + 0.015, 0.08, btn_height))
            
            self.btn_lock_view = Button(ax_lock, f"锁定视角", color='lightyellow')
            self.btn_view_reset = Button(ax_vreset, '重置视角', color='lightgray')
            
            self.btn_lock_view.on_clicked(self._on_toggle_view_lock)
            self.btn_view_reset.on_clicked(self._on_reset_view)

            # 键盘事件
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        except Exception as e:
            print(f"控件创建失败: {e}")
            pass

    def _compose_info_text(self, t: float) -> str:
        """生成右上角参数面板的详细信息显示"""
        M1 = self.get_M1_position(t)
        to_center = self.RO_center - M1
        d = float(np.linalg.norm(to_center))
        R = self.RO_radius
        occluded = self.is_fully_occluded(t)
        total = float(self._occluded_total)
        S = self._smoke_center(t)
        
        # 切锥几何参数
        if d > R:
            alpha = float(np.arcsin(R / d))
            alpha_deg = float(np.degrees(alpha))
            apex_deg = 2.0 * alpha_deg
            h_max = d * float(np.cos(alpha))
            rim_radius = R
            d_tangent = float(np.sqrt(max(0.0, d*d - R*R)))
            
            geom_info = (
                f"切锥几何参数\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"距离: {d:.1f} m\n"
                f"半角α: {alpha_deg:.2f}°\n"
                f"顶角: {apex_deg:.2f}°\n"
                f"切面高: {h_max:.1f} m\n"
                f"切面半径: {rim_radius:.1f} m\n"
            )
        else:
            geom_info = (
                f"切锥几何参数\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⚠️ M1位于球体内部\n"
                f"距离: {d:.1f} m\n"
            )
        
        # 运动状态参数
        velocity_kmh = self.velocity * 3.6  # 转换为km/h
        remaining_dist = float(np.linalg.norm(self.FO - M1))
        remaining_time = remaining_dist / self.velocity
        
        motion_info = (
            f"\n运动状态\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"时间: {t:.2f} s\n"
            f"速度: {self.velocity:.0f} m/s\n"
            f"位置: ({M1[0]:.0f},{M1[1]:.0f},{M1[2]:.0f})\n"
            f"剩余: {remaining_dist:.0f} m\n"
        )
        
        # FY1和烟雾状态
        fy_pos = self.FY1_start + self.fy_dir * (self.fy_speed * max(0.0, t))
        
        if t < self.t_drop:
            fy_status = f"FY1: ({fy_pos[0]:.0f},{fy_pos[1]:.0f},{fy_pos[2]:.0f})"
            grenade_status = f"投弹: {self.t_drop - t:.1f} s"
        elif t < self.t_det:
            dtg = t - self.t_drop
            gpos = self.pos_drop + (self.fy_speed * self.fy_dir) * dtg + np.array([0.0, 0.0, -0.5 * self.g * dtg * dtg])
            fy_status = f"FY1: ({fy_pos[0]:.0f},{fy_pos[1]:.0f},{fy_pos[2]:.0f})"
            grenade_status = f"弹体: ({gpos[0]:.0f},{gpos[1]:.0f},{gpos[2]:.0f})\n爆炸: {self.t_det - t:.1f} s"
        else:
            fy_status = f"FY1: ({fy_pos[0]:.0f},{fy_pos[1]:.0f},{fy_pos[2]:.0f})"
            if S is not None:
                grenade_status = f"烟雾: ({S[0]:.0f},{S[1]:.0f},{S[2]:.0f})\n半径: {self.smoke_radius:.0f} m"
            else:
                grenade_status = "烟雾已消散"
        
        fy_info = (
            f"\nFY1干扰状态\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{fy_status}\n"
            f"{grenade_status}\n"
        )
        
        # 遮蔽分析结果
        occlusion_info = (
            f"\n遮蔽分析\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"状态: {'● 完全遮蔽' if occluded else '○ 未遮蔽'}\n"
            f"累计: {total:.2f} s\n"
            f"效率: {(total/self.total_time)*100:.1f}%\n"
        )
        
        return geom_info + motion_info + fy_info + occlusion_info

    # 事件与交互
    def _on_play(self, _event=None):
        self.running = True

    def _on_pause(self, _event=None):
        self.running = False

    def _on_reset(self, _event=None):
        self.running = False
        self.current_frame = 0
        if self.slider is not None:
            self._updating_slider = True
            try:
                self.slider.set_val(0.0)
            finally:
                self._updating_slider = False

    def _on_slider(self, value):
        if self._updating_slider:
            return
        t = float(value)
        self.current_frame = int(round(t / self.dt))
        # 立即刷新一帧，确保参数即时更新
        try:
            self._update_frame(self.current_frame)
        except Exception:
            pass

    def _on_toggle_view_lock(self, _event=None):
        self._preserve_view = not self._preserve_view
        try:
            label = '锁定视角: 开' if self._preserve_view else '锁定视角: 关'
            # 更新按钮文字
            if self.btn_lock_view is not None:
                self.btn_lock_view.label.set_text(label)
            if self.fig is not None:
                self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _on_reset_view(self, _event=None):
        if self.ax3d is None:
            return
        try:
            self.ax3d.view_init(elev=self._default_view[0], azim=self._default_view[1])  # type: ignore[attr-defined]
            if self.fig is not None:
                self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _on_speed_slider(self, value):
        try:
            self.play_speed = max(0.1, float(value))
            # 动态更新标签为“X倍速”
            if self.speed_slider is not None:
                self.speed_slider.label.set_text(f"{self.play_speed:.0f}倍速")
                if self.fig is not None:
                    self.fig.canvas.draw_idle()
        except Exception:
            self.play_speed = 1.0

    def _on_key(self, event):
        if event.key == ' ':
            self.running = not self.running
        elif event.key == 'right':
            self.current_frame = min(self.current_frame + 1, int(self.total_time / self.dt))
        elif event.key == 'left':
            self.current_frame = max(self.current_frame - 1, 0)
        elif event.key in ('r', 'R'):
            self._on_reset()
        elif event.key in ('c', 'C'):
            self.show_cone = not self.show_cone
        elif event.key in ('s', 'S'):
            self.show_sphere = not self.show_sphere
        elif event.key in ('a', 'A'):
            self.show_axis = not self.show_axis
        elif event.key in ('b', 'B'):
            self.show_rim = not self.show_rim
        elif event.key in ('m', 'M'):
            self.show_smoke = not self.show_smoke
        elif event.key in ('o', 'O'):
            self.show_overlay = not self.show_overlay
        elif event.key == '+':
            self.cone_alpha = min(1.0, self.cone_alpha + 0.05)
        elif event.key == '-':
            self.cone_alpha = max(0.05, self.cone_alpha - 0.05)

    def _update_frame(self, frame: int):
        if self.ax3d is None:
            return
        t = min(frame * self.dt, self.total_time)
        self._draw_scene(self.ax3d, t)
        if self.time_cursor_area is not None:
            self.time_cursor_area.set_xdata([t, t])
        if self.time_cursor_dist is not None:
            self.time_cursor_dist.set_xdata([t, t])
        if self.info_text is not None:
            self.info_text.set_text(self._compose_info_text(t))
        if self.status_text is not None:
            try:
                occluded = self.is_fully_occluded(t)
                self.status_text.set_text(f"时间: {t:.1f} s    完全遮蔽: {'YES' if occluded else 'NO'}")
            except Exception:
                pass
        if self.slider is not None and not self._updating_slider:
            # 仅在与当前值显著不同时更新，避免递归触发回调
            try:
                cur = float(self.slider.val)
            except Exception:
                cur = None
            if cur is None or abs(cur - t) > 1e-9:
                self._updating_slider = True
                try:
                    self.slider.set_val(t)
                finally:
                    self._updating_slider = False

    def analyze_projection_area(self):
        ts = np.arange(0.0, self.total_time + 1e-9, self.dt)
        half_angles_deg = []
        dists = []
        for t in ts:
            M1 = self.get_M1_position(float(t))
            d = float(np.linalg.norm(self.RO_center - M1))
            dists.append(d)
            if d > self.RO_radius:
                half_angles_deg.append(np.degrees(np.arcsin(self.RO_radius / d)))
            else:
                half_angles_deg.append(np.nan)
        return ts, np.array(half_angles_deg), np.array(dists)

    def run_interactive(self):
        self._build_layout()
        assert self.fig is not None, "Figure not initialized"
        total_frames = int(self.total_time / self.dt) + 1

        def _animate(_i):
            if self.running:
                # 支持倍速播放（非整数速度用累积步进）
                self._frame_accum += float(self.play_speed)
                step = int(self._frame_accum)
                if step >= 1:
                    self.current_frame = min(self.current_frame + step, total_frames - 1)
                    self._frame_accum -= step
            self._update_frame(self.current_frame)
            return []

        self.ani = animation.FuncAnimation(self.fig, _animate, frames=total_frames, interval=int(self.dt * 1000), blit=False)
        # 移除 tight_layout，避免与 3D Axes 冲突的警告
        plt.show()


if __name__ == '__main__':
    RealTimeProjection().run_interactive()
