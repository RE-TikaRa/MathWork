import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from typing import Optional, Tuple
import matplotlib

# 设置莫兰迪色系主题和中文字体
plt.style.use('dark_background')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['font.monospace'] = ['Microsoft YaHei', 'SimHei', 'Fira Code', 'DejaVu Sans Mono']
matplotlib.rcParams['axes.unicode_minus'] = False

# 莫兰迪色系配色方案
matplotlib.rcParams['axes.facecolor'] = '#2F2F2F'      # 深灰背景
matplotlib.rcParams['figure.facecolor'] = '#3A3A3A'    # 图形背景
matplotlib.rcParams['grid.color'] = '#5A5A5A'          # 网格颜色
matplotlib.rcParams['grid.alpha'] = 0.5
matplotlib.rcParams['text.color'] = '#E5E5E5'          # 文字颜色


class RealTimeProjection:
    """
    Q1 场景：
    - M1 以 300 m/s 朝 FO 直线运动。
    - RO 抽象为球：中心(0,200,5)，半径√74。
    - FY1 投放烟雾干扰：半径固定 10 m，t_det=5.1 s 起爆，起爆后下沉 3 m/s 持续 20 s。
    - 实时可视化切锥、烟团、以及"完全遮蔽"时间段，并给出关键几何数据。
    """

    def __init__(self) -> None:
        # 莫兰迪色系配色定义
        self.colors = {
            'morandi_sage': '#9CAF88',      # 莫兰迪鼠尾草绿 - 用于RO目标
            'morandi_dusty_rose': '#D4A5A5', # 莫兰迪玫瑰粉 - 用于FO假目标
            'morandi_warm_gray': '#A8A5A0',  # 莫兰迪暖灰 - 用于烟团
            'morandi_soft_blue': '#8FA5C7',  # 莫兰迪柔和蓝 - 用于FY1
            'morandi_mauve': '#B09FAC',      # 莫兰迪淡紫 - 用于M1导弹
            'morandi_beige': '#C7B299',      # 莫兰迪米色 - 用于轨迹
            'morandi_lavender': '#A5A2C7',   # 莫兰迪薰衣草 - 用于切锥
            'morandi_cream': '#E5D5C8',      # 莫兰迪奶油色 - 用于文字背景
            'background_dark': '#3A3A3A',    # 深色背景
            'text_light': '#E5E5E5',         # 浅色文字
        }
        
        # 场景定义
        self.M1_start = np.array([20000.0, 0.0, 2000.0])
        self.FO = np.array([0.0, 0.0, 0.0])
        self.RO_center = np.array([0.0, 200.0, 5.0])
        self.RO_radius = float(np.sqrt(74.0))
        self.FY1_start = np.array([17800.0, 0.0, 1800.0])

        # 基础物理参数
        self.v_missile = 300.0
        self.fy_speed = 120.0
        self.g = 9.8
        
        # 时间节点
        self.t_drop = 1.5
        self.t_det = 5.1
        self.smoke_duration = 20.0
        self.total_time = float(np.linalg.norm(self.FO - self.M1_start) / self.v_missile)
        self.dt = 0.01
        
        # 运动参数
        self.smoke_v_down = 3.0
        self.smoke_radius = 10.0
        self.R_smoke = 10.0
        
        # FY1飞行方向：朝向假目标FO
        self.fy_dir = self.FO - self.FY1_start
        self.fy_dir = self.fy_dir / np.linalg.norm(self.fy_dir)
        
        # 关键位置预计算
        self.pos_drop = self.FY1_start + self.fy_dir * (self.fy_speed * self.t_drop)
        det_dt = self.t_det - self.t_drop
        v0 = self.fy_speed * self.fy_dir
        self.pos_det = self.pos_drop + v0 * det_dt + np.array([0.0, 0.0, -0.5 * self.g * det_dt * det_dt])
        
        # 界面控制
        self.show_sphere = True
        self.show_fy1 = True
        self.show_smoke = True
        self.show_cone = True
        self.show_rim = True
        self.show_axis = True
        self.show_overlay = True
        self.cone_alpha = 0.3
        self._preserve_view = True
        self._default_view = (30.0, -60.0)
        
        # 动画控制
        self.running = False
        self.current_frame = 0
        self.play_speed = 1.0
        self._frame_accum = 0.0
        self._updating_slider = False
        
        # 界面组件
        self.fig = None
        self.ax3d = None
        self.ax_info = None
        self.ax_area = None
        self.ax_dist = None
        self.ax_control = None
        self.info_text = None
        self.status_text = None
        self.slider = None
        self.ani = None
        
        # 预计算数据缓存
        self._times = None
        self._areas = None
        self._dists = None
        self._occluded_ts = None
        self._occluded_flags = None
        self._occluded_total = None

    def get_M1_position(self, t: float) -> np.ndarray:
        direction = self.FO - self.M1_start
        direction = direction / np.linalg.norm(direction)
        return self.M1_start + direction * self.v_missile * t

    def _smoke_center(self, t: float) -> Optional[np.ndarray]:
        if t < self.t_det:
            return None
        dt = t - self.t_det
        if dt > self.smoke_duration:
            return None
        z_offset = -self.smoke_v_down * dt
        return self.pos_det + np.array([0.0, 0.0, z_offset])

    def is_fully_occluded(self, t: float) -> bool:
        M1 = self.get_M1_position(t)
        smoke_center = self._smoke_center(t)
        if smoke_center is None:
            return False
        
        to_center = self.RO_center - M1
        dist = float(np.linalg.norm(to_center))
        if dist <= self.RO_radius:
            return True
        
        view_dir = to_center / dist
        apex_to_smoke = smoke_center - M1
        proj_length = float(np.dot(apex_to_smoke, view_dir))
        
        if proj_length <= 0:
            return False
        
        half_angle = float(np.arcsin(self.RO_radius / dist))
        cone_radius_at_smoke = proj_length * float(np.tan(half_angle))
        lateral_distance = float(np.linalg.norm(apex_to_smoke - proj_length * view_dir))
        
        return lateral_distance + self.R_smoke <= cone_radius_at_smoke

    def analyze_full_occlusion(self, ts: np.ndarray):
        flags = np.array([self.is_fully_occluded(float(t)) for t in ts])
        occluded_ts = ts[flags]
        total_time = float(np.sum(np.diff(ts)[:-1][flags[1:]])) if len(occluded_ts) > 1 else 0.0
        return occluded_ts, flags, total_time

    @staticmethod
    def _shade_occlusion(ax, ts: np.ndarray, flags: np.ndarray) -> None:
        """为遮蔽时间段添加莫兰迪风格的阴影显示"""
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
                # 使用莫兰迪鼠尾草绿阴影表示遮蔽区域
                ax.axvspan(t_start, t_end, color='#9CAF88', alpha=0.3, 
                          label='遮蔽时段' if t_start == ts[flags].min() else '')
                on = False

    def _draw_scene(self, ax, t: float) -> None:
        # 在清空前保存用户当前视角
        try:
            elev, azim = float(getattr(ax, 'elev', 30.0)), float(getattr(ax, 'azim', -60.0))
        except Exception:
            elev, azim = 30.0, -60.0

        ax.clear()
        
        # 设置莫兰迪风格的3D场景外观
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False  
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#5A5A5A')
        ax.yaxis.pane.set_edgecolor('#5A5A5A')
        ax.zaxis.pane.set_edgecolor('#5A5A5A')
        ax.grid(True, alpha=0.4, color='#5A5A5A')
        
        M1 = self.get_M1_position(t)
        occluded = self.is_fully_occluded(t)

        # 固定对象：使用莫兰迪配色，增大尺寸避免重叠
        ax.scatter(*self.FO, color=self.colors['morandi_dusty_rose'], s=180, 
                  label='FO (假目标)', marker='*', edgecolors=self.colors['text_light'], linewidth=2)
        ax.scatter(*self.RO_center, color=self.colors['morandi_sage'], s=180, 
                  label='RO (真目标)', marker='o', edgecolors=self.colors['text_light'], linewidth=2)

        # 球体线框：使用莫兰迪配色，减少密度避免视觉混乱
        if self.show_sphere:
            u = np.linspace(0, 2 * np.pi, 24)
            v = np.linspace(0, np.pi, 16)
            x = self.RO_center[0] + self.RO_radius * np.outer(np.cos(u), np.sin(v))
            y = self.RO_center[1] + self.RO_radius * np.outer(np.sin(u), np.sin(v))
            z = self.RO_center[2] + self.RO_radius * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(x, y, z, color=self.colors['morandi_sage'], alpha=0.4, linewidth=1)

        # M1 位置：根据遮蔽状态动态改变颜色
        m1_color = self.colors['morandi_mauve'] if not occluded else self.colors['morandi_dusty_rose']
        m1_marker = 'D' if not occluded else '^'
        m1_size = 200 if not occluded else 220
        ax.scatter(*M1, color=m1_color, s=m1_size, label=f'M1 t={t:.1f}s', marker=m1_marker,
                  edgecolors=self.colors['text_light'], linewidth=2)

        # FY1 无人机运动与投弹/弹体轨迹：使用莫兰迪配色
        if self.show_fy1:
            fy_t = float(max(0.0, t))
            fy_pos = self.FY1_start + self.fy_dir * (self.fy_speed * fy_t)
            tt_fy = np.linspace(0.0, fy_t, 50)
            traj_fy = self.FY1_start + self.fy_dir[None, :] * (self.fy_speed * tt_fy[:, None])
            ax.plot(traj_fy[:, 0], traj_fy[:, 1], traj_fy[:, 2], color=self.colors['morandi_soft_blue'], 
                   alpha=0.8, linewidth=3, label='FY1 航迹')
            ax.scatter(*fy_pos, color=self.colors['morandi_soft_blue'], s=120, label='FY1', marker='s',
                      edgecolors=self.colors['text_light'], linewidth=2)
            ax.scatter(*self.pos_drop, color=self.colors['morandi_lavender'], s=100, 
                      label='投弹点 t=1.5s', marker='v', edgecolors=self.colors['text_light'], linewidth=2)
            
            if t >= self.t_drop:
                t0 = self.t_drop
                t1 = min(t, self.t_det)
                ts_seg = np.linspace(t0, t1, 50)
                dt_seg = ts_seg - t0
                pos_seg = self.pos_drop[None, :] + (self.fy_speed * self.fy_dir)[None, :] * dt_seg[:, None] \
                          + np.array([0.0, 0.0, -0.5 * self.g])[None, :] * (dt_seg[:, None] ** 2)
                ax.plot(pos_seg[:, 0], pos_seg[:, 1], pos_seg[:, 2], color=self.colors['morandi_beige'], 
                       linestyle='--', linewidth=2.5, alpha=0.9, label='弹体轨迹')

        # 烟团球：使用莫兰迪配色
        S = self._smoke_center(t)
        if self.show_smoke and S is not None:
            u_s = np.linspace(0, 2 * np.pi, 20)
            v_s = np.linspace(0, np.pi, 15)
            xs = S[0] + self.smoke_radius * np.outer(np.cos(u_s), np.sin(v_s))
            ys = S[1] + self.smoke_radius * np.outer(np.sin(u_s), np.sin(v_s))
            zs = S[2] + self.smoke_radius * np.outer(np.ones_like(u_s), np.cos(v_s))
            
            smoke_color = self.colors['morandi_warm_gray'] if not occluded else self.colors['morandi_dusty_rose']
            smoke_alpha = 0.6 if not occluded else 0.8
            ax.plot_wireframe(xs, ys, zs, color=smoke_color, alpha=smoke_alpha, linewidth=1.2)
            ax.scatter(*self.pos_det, color=self.colors['morandi_cream'], s=80, label='烟团起爆点', 
                      marker='*', edgecolors='#3A3A3A', linewidth=1.5)

        # 切锥：使用莫兰迪配色
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
            h = np.linspace(0.0, h_max, 20)
            uu = np.linspace(0.0, 2.0 * np.pi, 60)
            H, U = np.meshgrid(h, uu, indexing='ij')
            R_h = H * float(np.tan(half_angle))
            X = M1[0] + view_dir[0] * H + R_h * (np.cos(U) * v1[0] + np.sin(U) * v2[0])
            Y = M1[1] + view_dir[1] * H + R_h * (np.cos(U) * v1[1] + np.sin(U) * v2[1])
            Z = M1[2] + view_dir[2] * H + R_h * (np.cos(U) * v1[2] + np.sin(U) * v2[2])
            
            cone_color = self.colors['morandi_lavender'] if not occluded else self.colors['morandi_dusty_rose']
            cone_alpha = 0.3 if not occluded else 0.5
            
            if self.show_cone:
                ax.plot_surface(X, Y, Z, color=cone_color, alpha=cone_alpha, shade=True, linewidth=0)
            theta = np.linspace(0.0, 2.0 * np.pi, 120)
            rim = center_rim + rim_radius * (np.cos(theta)[:, None] * v1 + np.sin(theta)[:, None] * v2)
            if self.show_rim:
                rim_color = self.colors['morandi_lavender'] if not occluded else self.colors['morandi_dusty_rose']
                ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=rim_color, linewidth=3, label='切面圆')
                ax.scatter(*center_rim, color=rim_color, s=80, zorder=5, marker='o', 
                          edgecolors=self.colors['text_light'], linewidth=1.5)

        # 轴线：使用莫兰迪配色
        if self.show_axis:
            axis_color = self.colors['morandi_lavender'] if not occluded else self.colors['morandi_dusty_rose']
            ax.plot([M1[0], self.RO_center[0]], [M1[1], self.RO_center[1]], [M1[2], self.RO_center[2]],
                    color=axis_color, linestyle='-.', linewidth=2.5, alpha=0.9, label='视线轴线')

        # M1轨迹：使用莫兰迪配色
        t2 = min(float(t) + 1.0, self.total_time)
        traj_t = np.linspace(max(0.0, t2 - 1.0), t2, 50)
        traj = np.array([self.get_M1_position(tt) for tt in traj_t])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '--', color=self.colors['morandi_beige'], 
               alpha=0.8, linewidth=2.5, label='M1轨迹')

        # 坐标轴标签：使用浅色文字
        ax.set_xlabel('X (m)', fontsize=12, color=self.colors['text_light'], weight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, color=self.colors['text_light'], weight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, color=self.colors['text_light'], weight='bold')
        
        # 动态标题：根据遮蔽状态变化颜色
        title_color = self.colors['morandi_dusty_rose'] if occluded else self.colors['morandi_sage']
        occlusion_status = "完全遮蔽" if occluded else "无遮蔽"
        ax.set_title(f'烟幕干扰三维场景 - {occlusion_status} (t={t:.1f}s)', 
                    fontsize=16, family='Microsoft YaHei', color=title_color, weight='bold', pad=20)
        
        # 3D 叠加关键参数
        if self.show_overlay:
            to_center = self.RO_center - M1
            d = float(np.linalg.norm(to_center))
            if d > self.RO_radius:
                alpha = float(np.arcsin(self.RO_radius / d))
                alpha_deg = float(np.degrees(alpha))
                apex_deg = 2.0 * alpha_deg
                overlay = (
                    f"时间: {t:.2f}s  距离: {d:.1f}m  半角: {alpha_deg:.2f}°  "
                    f"顶角: {apex_deg:.2f}°  遮蔽: {'完全' if occluded else '无'}"
                )
            else:
                overlay = f"时间: {t:.2f}s  M1位于球内  遮蔽: {'完全' if occluded else '无'}"
            try:
                text_color = self.colors['morandi_dusty_rose'] if occluded else self.colors['morandi_sage']
                ax.text2D(0.02, 0.98, overlay, transform=ax.transAxes, va='top', ha='left',
                          fontsize=11, family='Microsoft YaHei', color=text_color, weight='bold',
                          bbox=dict(facecolor=self.colors['background_dark'], alpha=0.9, 
                                   edgecolor=text_color, linewidth=1.5, pad=8))
            except Exception:
                pass

        # 图例：改进样式，避免重叠
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
                # 调整图例位置，放在左下角避免与其他元素重叠
                legend = ax.legend(new_h, new_l, loc='lower left', fontsize=8, framealpha=0.95, 
                                 facecolor=self.colors['background_dark'], 
                                 edgecolor=self.colors['text_light'], linewidth=1,
                                 bbox_to_anchor=(0.02, 0.02), ncol=2)
                legend.get_frame().set_linewidth(1.5)
                for text in legend.get_texts():
                    text.set_color(self.colors['text_light'])
        except Exception:
            pass

        # 恢复用户视角
        if self._preserve_view:
            try:
                ax.view_init(elev=elev, azim=azim)
            except Exception:
                pass
        
        # 设置显示范围
        ax.set_xlim(-1000, 21000)
        ax.set_ylim(-100, 300)
        ax.set_zlim(-50, 2500)

    def _build_layout(self):
        """构建莫兰迪色系的界面布局，优化间距避免重叠"""
        # 增大窗口尺寸以提供更好的视觉体验和避免重叠
        self.fig = plt.figure(figsize=(20, 16))
        
        # 调整网格布局比例，给控制区域更多空间
        gs = self.fig.add_gridspec(3, 2, 
                                   height_ratios=[3.0, 1.8, 1.0], 
                                   width_ratios=[2.5, 1.2],
                                   hspace=0.35, wspace=0.25)
        
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
        
        # 调整边距，优化按钮区域布局
        self.fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)

        # 分析曲线 + 遮蔽预计算
        ts, angles_deg, dists = self.analyze_projection_area()
        self._times, self._areas, self._dists = ts, angles_deg, dists
        self._occluded_ts, self._occluded_flags, self._occluded_total = self.analyze_full_occlusion(ts)

        # 图1：视线半角变化 - 莫兰迪配色
        self.ax_area.plot(ts, angles_deg, color=self.colors['morandi_dusty_rose'], linewidth=3, alpha=0.9)
        self.ax_area.set_title('图1 - 视线半角变化', fontsize=14, family='Microsoft YaHei', 
                              color=self.colors['text_light'], weight='bold', pad=15)
        self.ax_area.set_xlabel('时间 (s)', fontsize=12, color=self.colors['text_light'])
        self.ax_area.set_ylabel('半角 (°)', fontsize=12, color=self.colors['text_light'])
        self.ax_area.grid(True, alpha=0.4, color='#5A5A5A')
        self.ax_area.tick_params(colors=self.colors['text_light'], labelsize=10)
        self._shade_occlusion(self.ax_area, ts, self._occluded_flags)

        # 图2：距离变化 - 莫兰迪配色
        self.ax_dist.plot(ts, dists, color=self.colors['morandi_soft_blue'], linewidth=3, alpha=0.9)
        self.ax_dist.set_title('图2 - M1到RO距离', fontsize=14, family='Microsoft YaHei', 
                              color=self.colors['text_light'], weight='bold', pad=15)
        self.ax_dist.set_xlabel('时间 (s)', fontsize=12, color=self.colors['text_light'])
        self.ax_dist.set_ylabel('距离 (m)', fontsize=12, color=self.colors['text_light'])
        self.ax_dist.grid(True, alpha=0.4, color='#5A5A5A')
        self.ax_dist.tick_params(colors=self.colors['text_light'], labelsize=10)
        self._shade_occlusion(self.ax_dist, ts, self._occluded_flags)

        # 初始化右上角参数面板 - 莫兰迪样式，调整文字大小避免重叠
        self.info_text = self.ax_info.text(0.05, 0.95, self._compose_info_text(0.0), 
                                          va='top', ha='left', fontsize=8, 
                                          family='Microsoft YaHei', color=self.colors['text_light'],
                                          transform=self.ax_info.transAxes,
                                          bbox=dict(boxstyle="round,pad=0.8", 
                                                   facecolor=self.colors['background_dark'], 
                                                   edgecolor=self.colors['morandi_sage'], 
                                                   linewidth=2, alpha=0.95))

        # 初始 3D 视角
        try:
            if hasattr(self.ax3d, 'view_init'):
                self.ax3d.view_init(elev=self._default_view[0], azim=self._default_view[1])
        except Exception:
            pass

        # 初始化一帧
        try:
            self._draw_scene(self.ax3d, 0.0)
            if self.info_text is not None:
                self.info_text.set_text(self._compose_info_text(0.0))
        except Exception as e:
            print(f"初始化绘制失败: {e}")

        # 控制按钮：莫兰迪配色，优化尺寸和布局
        try:
            # 优化按钮尺寸参数
            btn_height = 0.04  # 适中的按钮高度
            btn_width = 0.10   # 稍宽的按钮便于点击
            btn_spacing = 0.13  # 合适的按钮间距
            btn_y = 0.02       # 底部合适位置
            
            # 播放按钮
            ax_btn_play = plt.axes((0.12, btn_y, btn_width, btn_height), facecolor=self.colors['morandi_sage'])
            self.btn_play = Button(ax_btn_play, '播放', color=self.colors['morandi_sage'], 
                                  hovercolor=self.colors['morandi_dusty_rose'])
            ax_btn_play.tick_params(labelsize=10)
            
            # 暂停按钮
            ax_btn_pause = plt.axes((0.12 + btn_spacing, btn_y, btn_width, btn_height), 
                                   facecolor=self.colors['morandi_mauve'])
            self.btn_pause = Button(ax_btn_pause, '暂停', color=self.colors['morandi_mauve'], 
                                   hovercolor=self.colors['morandi_dusty_rose'])
            ax_btn_pause.tick_params(labelsize=10)
            
            # 重置按钮
            ax_btn_reset = plt.axes((0.12 + 2*btn_spacing, btn_y, btn_width, btn_height), 
                                   facecolor=self.colors['morandi_beige'])
            self.btn_reset = Button(ax_btn_reset, '重置', color=self.colors['morandi_beige'], 
                                   hovercolor=self.colors['morandi_dusty_rose'])
            ax_btn_reset.tick_params(labelsize=10)

            # 时间滑块：优化位置和尺寸
            slider_y = btn_y + 0.01  # 紧贴按钮上方
            slider_width = 0.35      # 更宽的滑块便于操作
            slider_height = 0.025    # 合适的滑块高度
            ax_slider = plt.axes((0.55, slider_y, slider_width, slider_height), facecolor=self.colors['background_dark'])
            self.slider = Slider(ax_slider, '时间进度', 0.0, self.total_time, valinit=0.0, 
                               color=self.colors['morandi_soft_blue'], 
                               facecolor=self.colors['background_dark'])
            
            # 状态文本：移到顶部显示
            status_y = 0.95
            self.status_text = self.fig.text(0.5, status_y, "时间: 0.0s | 遮蔽状态: NO", 
                                           ha='center', va='center', fontsize=12, 
                                           color=self.colors['text_light'], weight='bold',
                                           bbox=dict(boxstyle="round,pad=0.5", 
                                                    facecolor=self.colors['background_dark'], 
                                                    edgecolor=self.colors['morandi_sage'], 
                                                    linewidth=1.5, alpha=0.9))

            # 绑定事件
            self.btn_play.on_clicked(self._on_play)
            self.btn_pause.on_clicked(self._on_pause)
            self.btn_reset.on_clicked(self._on_reset)
            self.slider.on_changed(self._on_slider)
        except Exception as e:
            print(f"控件创建失败: {e}")

    def _compose_info_text(self, t: float) -> str:
        """生成右上角参数面板的详细信息显示"""
        M1 = self.get_M1_position(t)
        to_center = self.RO_center - M1
        d = float(np.linalg.norm(to_center))
        R = self.RO_radius
        occluded = self.is_fully_occluded(t)
        total_val = self._occluded_total if self._occluded_total is not None else 0.0
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
                f"{'─' * 14}\n"
                f"距离: {d:.1f}m\n"
                f"半角α: {alpha_deg:.2f}°\n"
                f"顶角: {apex_deg:.2f}°\n"
                f"切面高: {h_max:.1f}m\n"
                f"切面半径: {rim_radius:.1f}m\n"
            )
        else:
            geom_info = (
                f"切锥几何参数\n"
                f"{'─' * 14}\n"
                f"⚠️ M1位于球体内部\n"
                f"距离: {d:.1f}m\n"
            )
        
        # 运动状态参数
        motion_info = (
            f"\n运动状态参数\n"
            f"{'─' * 14}\n"
            f"时间: {t:.2f}s\n"
            f"M1位置: ({M1[0]:.0f},{M1[1]:.0f},{M1[2]:.0f})\n"
            f"速度: {self.v_missile:.0f}m/s\n"
        )
        
        # 烟团状态
        if S is not None:
            smoke_info = (
                f"\n烟团状态\n"
                f"{'─' * 14}\n"
                f"中心: ({S[0]:.0f},{S[1]:.0f},{S[2]:.0f})\n"
                f"半径: {self.R_smoke:.0f}m\n"
                f"起爆: {self.t_det:.1f}s\n"
                f"下沉: {self.smoke_v_down:.1f}m/s\n"
            )
        else:
            smoke_info = (
                f"\n烟团状态\n"
                f"{'─' * 14}\n"
                f"状态: 未起爆\n"
                f"起爆: {self.t_det:.1f}s\n"
            )
        
        # 遮蔽分析
        occlusion_info = (
            f"\n遮蔽分析\n"
            f"{'─' * 14}\n"
            f"当前: {'完全遮蔽' if occluded else '无遮蔽'}\n"
            f"总时长: {total_val:.2f}s\n"
        )
        
        return geom_info + motion_info + smoke_info + occlusion_info

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

    def _on_play(self, event):
        self.running = True

    def _on_pause(self, event):
        self.running = False

    def _on_reset(self, event):
        self.running = False
        self.current_frame = 0
        self._frame_accum = 0.0

    def _on_slider(self, val):
        if not self._updating_slider:
            frame = int(val / self.dt)
            self.current_frame = min(frame, int(self.total_time / self.dt))

    def _update_frame(self, frame_idx: int):
        t = frame_idx * self.dt
        try:
            self._draw_scene(self.ax3d, t)
            if self.info_text is not None:
                self.info_text.set_text(self._compose_info_text(t))
            if self.status_text is not None:
                occluded = self.is_fully_occluded(t)
                status_color = self.colors['morandi_dusty_rose'] if occluded else self.colors['morandi_sage']
                self.status_text.set_text(f"时间: {t:.1f}s | 遮蔽状态: {'YES' if occluded else 'NO'}")
                self.status_text.set_color(status_color)
        except Exception:
            pass
        
        if self.slider is not None and not self._updating_slider:
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

    def run_interactive(self):
        self._build_layout()
        assert self.fig is not None, "Figure not initialized"
        total_frames = int(self.total_time / self.dt) + 1

        def _animate(_i):
            if self.running:
                self._frame_accum += float(self.play_speed)
                step = int(self._frame_accum)
                if step >= 1:
                    self.current_frame = min(self.current_frame + step, total_frames - 1)
                    self._frame_accum -= step
            self._update_frame(self.current_frame)
            return []

        self.ani = animation.FuncAnimation(self.fig, _animate, frames=total_frames, interval=int(self.dt * 1000), blit=False)
        plt.show()


if __name__ == "__main__":
    proj = RealTimeProjection()
    proj.run_interactive()
