import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from typing import Optional, Tuple
import matplotlib

# 设置色系主题和中文字体
plt.style.use('dark_background')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['font.monospace'] = ['Microsoft YaHei', 'SimHei', 'Fira Code', 'DejaVu Sans Mono']
matplotlib.rcParams['axes.unicode_minus'] = False

# 色系配色方案
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
        # 增强莫兰迪色系配色定义
        self.colors = {
            'morandi_sage': '#9CAF88',      # 鼠尾草绿 - 用于RO目标
            'morandi_dusty_rose': '#D4A5A5', # 玫瑰粉 - 用于FO假目标
            'morandi_warm_gray': '#A8A5A0',  # 暖灰 - 用于烟团
            'morandi_soft_blue': '#8FA5C7',  # 柔和蓝 - 用于FY1
            'morandi_mauve': '#B09FAC',      # 淡紫 - 用于M1导弹
            'morandi_beige': '#C7B299',      # 米色 - 用于轨迹
            'morandi_lavender': '#A5A2C7',   # 薰衣草 - 用于切锥
            'morandi_cream': '#E5D5C8',      # 奶油色 - 用于文字背景
            'morandi_terracotta': '#C49A7C', # 赤陶色 - 用于警告
            'morandi_olive': '#A0A882',      # 橄榄绿 - 用于成功状态
            'morandi_pearl': '#F0EDE8',      # 珍珠白 - 用于高亮
            'morandi_slate': '#6B7B7F',      # 石板灰 - 用于次要元素
            'background_dark': '#3A3A3A',    # 深色背景
            'background_panel': '#2F2F2F',   # 面板背景
            'text_light': '#E5E5E5',         # 浅色文字
            'accent_gold': '#D4AF37',        # 金色强调
            'accent_crimson': '#DC143C',     # 深红强调
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
        
        # 界面控制和视觉增强
        self.show_sphere = True
        self.show_fy1 = True
        self.show_smoke = True
        self.show_cone = True
        self.show_rim = True
        self.show_axis = True
        self.show_overlay = True
        self.show_grid = True
        self.show_trajectory = True
        self.show_effects = True  # 新增：特效开关
        self.cone_alpha = 0.3
        self._preserve_view = True
        self._default_view = (30.0, -60.0)
        
        # 动画和特效控制
        self.running = False
        self.current_frame = 0
        self.play_speed = 1.0
        self._frame_accum = 0.0
        self._updating_slider = False
        self.pulse_factor = 0.0  # 新增：脉冲效果因子
        self.glow_intensity = 1.0  # 新增：发光强度
        
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
        """为遮蔽时间段添加风格的阴影显示"""
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
                # 使用鼠尾草绿阴影表示遮蔽区域
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
        
        # 设置美化的3D场景外观
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False  
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(self.colors['morandi_slate'])
        ax.yaxis.pane.set_edgecolor(self.colors['morandi_slate'])
        ax.zaxis.pane.set_edgecolor(self.colors['morandi_slate'])
        
        # 美化网格
        if self.show_grid:
            ax.grid(True, alpha=0.3, color=self.colors['morandi_slate'], linewidth=0.8)
        
        # 计算动态效果参数
        self.pulse_factor = (np.sin(t * 2) + 1) * 0.5  # 0到1的脉冲
        
        M1 = self.get_M1_position(t)
        occluded = self.is_fully_occluded(t)

        # 增强的固定对象显示
        # FO假目标 - 添加脉冲效果
        fo_size = 200 + (40 * self.pulse_factor if self.show_effects else 0)
        fo_alpha = 0.9 + (0.1 * self.pulse_factor if self.show_effects else 0)
        ax.scatter(*self.FO, color=self.colors['morandi_dusty_rose'], s=fo_size, 
                  label='FO (假目标)', marker='*', edgecolors=self.colors['morandi_pearl'], 
                  linewidth=2.5, alpha=fo_alpha, zorder=10)
        
        # RO真目标 - 根据遮蔽状态动态变化
        ro_color = self.colors['accent_crimson'] if occluded else self.colors['morandi_sage']
        ro_size = 220 if occluded else 200
        ro_glow = self.colors['morandi_pearl'] if occluded else self.colors['morandi_sage']
        ax.scatter(*self.RO_center, color=ro_color, s=ro_size, 
                  label='RO (真目标)', marker='o', edgecolors=ro_glow, 
                  linewidth=3, alpha=0.95, zorder=10)

        # 美化球体线框
        if self.show_sphere:
            u = np.linspace(0, 2 * np.pi, 32)  # 增加密度
            v = np.linspace(0, np.pi, 20)
            x = self.RO_center[0] + self.RO_radius * np.outer(np.cos(u), np.sin(v))
            y = self.RO_center[1] + self.RO_radius * np.outer(np.sin(u), np.sin(v))
            z = self.RO_center[2] + self.RO_radius * np.outer(np.ones_like(u), np.cos(v))
            sphere_color = self.colors['accent_crimson'] if occluded else self.colors['morandi_sage']
            sphere_alpha = 0.6 if occluded else 0.4
            ax.plot_wireframe(x, y, z, color=sphere_color, alpha=sphere_alpha, linewidth=1.2)

        # 增强M1导弹显示
        m1_base_color = self.colors['morandi_mauve'] if not occluded else self.colors['accent_crimson']
        m1_marker = 'D' if not occluded else '^'
        m1_size = 250 + (30 * self.pulse_factor if occluded and self.show_effects else 0)
        m1_glow = self.colors['morandi_pearl'] if not occluded else self.colors['accent_gold']
        ax.scatter(*M1, color=m1_base_color, s=m1_size, label=f'M1 t={t:.1f}s', marker=m1_marker,
                  edgecolors=m1_glow, linewidth=3, alpha=0.95, zorder=10)

        # 增强FY1无人机轨迹显示
        if self.show_fy1:
            fy_t = float(max(0.0, t))
            fy_pos = self.FY1_start + self.fy_dir * (self.fy_speed * fy_t)
            
            # 美化航迹线
            if self.show_trajectory:
                tt_fy = np.linspace(0.0, fy_t, 100)  # 增加轨迹密度
                traj_fy = self.FY1_start + self.fy_dir[None, :] * (self.fy_speed * tt_fy[:, None])
                
                # 渐变轨迹效果
                for i in range(len(traj_fy) - 1):
                    alpha = 0.3 + 0.7 * (i / len(traj_fy))  # 渐变透明度
                    ax.plot([traj_fy[i, 0], traj_fy[i+1, 0]], 
                           [traj_fy[i, 1], traj_fy[i+1, 1]], 
                           [traj_fy[i, 2], traj_fy[i+1, 2]], 
                           color=self.colors['morandi_soft_blue'], alpha=alpha, linewidth=2.5)
            
            # FY1位置显示
            fy_size = 150 + (20 * self.pulse_factor if self.show_effects else 0)
            ax.scatter(*fy_pos, color=self.colors['morandi_soft_blue'], s=fy_size, label='FY1', 
                      marker='s', edgecolors=self.colors['morandi_pearl'], linewidth=2.5, alpha=0.9)
            
            # 投弹点 - 增强显示
            drop_size = 120 + (15 * self.pulse_factor if self.show_effects else 0)
            ax.scatter(*self.pos_drop, color=self.colors['morandi_lavender'], s=drop_size, 
                      label='投弹点 t=1.5s', marker='v', edgecolors=self.colors['accent_gold'], 
                      linewidth=2.5, alpha=0.95)
            
            # 弹体轨迹 - 抛物线轨迹
            if t >= self.t_drop:
                t0 = self.t_drop
                t1 = min(t, self.t_det)
                ts_seg = np.linspace(t0, t1, 80)  # 增加密度
                dt_seg = ts_seg - t0
                pos_seg = self.pos_drop[None, :] + (self.fy_speed * self.fy_dir)[None, :] * dt_seg[:, None] \
                          + np.array([0.0, 0.0, -0.5 * self.g])[None, :] * (dt_seg[:, None] ** 2)
                
                # 渐变弹体轨迹
                for i in range(len(pos_seg) - 1):
                    alpha = 0.4 + 0.6 * (i / len(pos_seg))
                    ax.plot([pos_seg[i, 0], pos_seg[i+1, 0]], 
                           [pos_seg[i, 1], pos_seg[i+1, 1]], 
                           [pos_seg[i, 2], pos_seg[i+1, 2]], 
                           color=self.colors['morandi_beige'], linestyle='--', 
                           linewidth=2.5, alpha=alpha)

        # 增强烟团显示
        S = self._smoke_center(t)
        if self.show_smoke and S is not None:
            # 多层烟团效果
            for layer in range(3):
                radius_scale = 1.0 + layer * 0.2
                alpha_scale = 0.8 - layer * 0.2
                
                u_s = np.linspace(0, 2 * np.pi, 24)
                v_s = np.linspace(0, np.pi, 16)
                xs = S[0] + self.smoke_radius * radius_scale * np.outer(np.cos(u_s), np.sin(v_s))
                ys = S[1] + self.smoke_radius * radius_scale * np.outer(np.sin(u_s), np.sin(v_s))
                zs = S[2] + self.smoke_radius * radius_scale * np.outer(np.ones_like(u_s), np.cos(v_s))
                
                smoke_color = self.colors['accent_crimson'] if occluded else self.colors['morandi_warm_gray']
                smoke_alpha = (0.6 if not occluded else 0.8) * alpha_scale
                
                if layer == 0:  # 内核
                    ax.plot_wireframe(xs, ys, zs, color=smoke_color, alpha=smoke_alpha, linewidth=1.5)
                else:  # 外层
                    ax.plot_wireframe(xs, ys, zs, color=smoke_color, alpha=smoke_alpha * 0.5, linewidth=0.8)
            
            # 起爆点增强显示
            det_size = 100 + (25 * self.pulse_factor if self.show_effects else 0)
            det_color = self.colors['accent_gold'] if occluded else self.colors['morandi_cream']
            ax.scatter(*self.pos_det, color=det_color, s=det_size, label='烟团起爆点', 
                      marker='*', edgecolors=self.colors['background_dark'], linewidth=2, alpha=0.95)

        # 增强切锥显示
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
            
            # 更高密度的切锥
            h = np.linspace(0.0, h_max, 30)
            uu = np.linspace(0.0, 2.0 * np.pi, 80)
            H, U = np.meshgrid(h, uu, indexing='ij')
            R_h = H * float(np.tan(half_angle))
            X = M1[0] + view_dir[0] * H + R_h * (np.cos(U) * v1[0] + np.sin(U) * v2[0])
            Y = M1[1] + view_dir[1] * H + R_h * (np.cos(U) * v1[1] + np.sin(U) * v2[1])
            Z = M1[2] + view_dir[2] * H + R_h * (np.cos(U) * v1[2] + np.sin(U) * v2[2])
            
            # 动态切锥颜色和透明度
            cone_base_color = self.colors['morandi_lavender'] if not occluded else self.colors['accent_crimson']
            cone_alpha = (0.25 if not occluded else 0.45) + (0.1 * self.pulse_factor if self.show_effects else 0)
            
            if self.show_cone:
                ax.plot_surface(X, Y, Z, color=cone_base_color, alpha=cone_alpha, 
                               shade=True, linewidth=0, antialiased=True)
            
            # 美化切面圆
            theta = np.linspace(0.0, 2.0 * np.pi, 150)  # 增加密度
            rim = center_rim + rim_radius * (np.cos(theta)[:, None] * v1 + np.sin(theta)[:, None] * v2)
            if self.show_rim:
                rim_color = cone_base_color
                rim_width = 3.5 + (0.5 * self.pulse_factor if self.show_effects else 0)
                ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=rim_color, linewidth=rim_width, 
                       label='切面圆', alpha=0.9)
                
                # 切面圆心点
                center_size = 100 + (20 * self.pulse_factor if self.show_effects else 0)
                ax.scatter(*center_rim, color=rim_color, s=center_size, zorder=8, marker='o', 
                          edgecolors=self.colors['morandi_pearl'], linewidth=2, alpha=0.9)

        # 美化视线轴线
        if self.show_axis:
            axis_color = self.colors['morandi_lavender'] if not occluded else self.colors['accent_crimson']
            axis_width = 2.8 + (0.4 * self.pulse_factor if self.show_effects else 0)
            ax.plot([M1[0], self.RO_center[0]], [M1[1], self.RO_center[1]], [M1[2], self.RO_center[2]],
                    color=axis_color, linestyle='-.', linewidth=axis_width, alpha=0.95, label='视线轴线')

        # 增强M1轨迹显示
        if self.show_trajectory:
            t2 = min(float(t) + 2.0, self.total_time)  # 扩展轨迹长度
            traj_t = np.linspace(max(0.0, t2 - 2.0), t2, 100)  # 增加密度
            traj = np.array([self.get_M1_position(tt) for tt in traj_t])
            
            # 渐变轨迹效果
            for i in range(len(traj) - 1):
                alpha = 0.2 + 0.8 * (i / len(traj))
                width = 1.5 + 1.5 * (i / len(traj))
                ax.plot([traj[i, 0], traj[i+1, 0]], 
                       [traj[i, 1], traj[i+1, 1]], 
                       [traj[i, 2], traj[i+1, 2]], 
                       '--', color=self.colors['morandi_beige'], 
                       alpha=alpha, linewidth=width)

        # 美化坐标轴标签
        ax.set_xlabel('X 坐标 (m)', fontsize=13, color=self.colors['text_light'], 
                     weight='bold', labelpad=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=13, color=self.colors['text_light'], 
                     weight='bold', labelpad=10)
        ax.set_zlabel('Z 坐标 (m)', fontsize=13, color=self.colors['text_light'], 
                     weight='bold', labelpad=10)
        
        # 美化坐标轴刻度
        ax.tick_params(axis='x', colors=self.colors['morandi_slate'], labelsize=10)
        ax.tick_params(axis='y', colors=self.colors['morandi_slate'], labelsize=10)
        ax.tick_params(axis='z', colors=self.colors['morandi_slate'], labelsize=10)
        
        # 动态增强标题
        title_base_color = self.colors['morandi_dusty_rose'] if occluded else self.colors['morandi_sage']
        title_glow_color = self.colors['accent_gold'] if occluded else self.colors['morandi_olive']
        occlusion_status = "🚨 完全遮蔽" if occluded else "✅ 视野清晰"
        title_size = 17 + (2 if occluded and self.show_effects else 0)
        
        ax.set_title(f'🎯 烟幕干扰三维仿真场景 - {occlusion_status} (t={t:.1f}s)', 
                    fontsize=title_size, family='Microsoft YaHei', color=title_base_color, 
                    weight='bold', pad=25)
        
        # 美化3D叠加信息面板
        if self.show_overlay:
            to_center = self.RO_center - M1
            d = float(np.linalg.norm(to_center))
            if d > self.RO_radius:
                alpha = float(np.arcsin(self.RO_radius / d))
                alpha_deg = float(np.degrees(alpha))
                apex_deg = 2.0 * alpha_deg
                overlay = (
                    f"⏱️ 时间: {t:.2f}s  📏 距离: {d:.1f}m\n"
                    f"📐 半角: {alpha_deg:.2f}°  🔺 顶角: {apex_deg:.2f}°\n"
                    f"{'🚫 遮蔽状态: 完全' if occluded else '👁️ 遮蔽状态: 无'}"
                )
            else:
                overlay = (
                    f"⏱️ 时间: {t:.2f}s\n"
                    f"⚠️ M1位于目标球体内部\n"
                    f"{'🚫 遮蔽状态: 完全' if occluded else '👁️ 遮蔽状态: 无'}"
                )
            try:
                text_color = self.colors['morandi_dusty_rose'] if occluded else self.colors['morandi_sage']
                bg_color = self.colors['background_panel']
                border_color = self.colors['accent_crimson'] if occluded else self.colors['morandi_olive']
                
                ax.text2D(0.02, 0.98, overlay, transform=ax.transAxes, va='top', ha='left',
                          fontsize=11, family='Microsoft YaHei', color=text_color, weight='bold',
                          bbox=dict(boxstyle="round,pad=1.0", facecolor=bg_color, alpha=0.95, 
                                   edgecolor=border_color, linewidth=2))
            except Exception:
                pass

        # 优化图例显示
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
                legend = ax.legend(new_h, new_l, loc='lower left', fontsize=9, framealpha=0.95, 
                                 facecolor=self.colors['background_panel'], 
                                 edgecolor=self.colors['morandi_slate'], linewidth=2,
                                 bbox_to_anchor=(0.02, 0.02), ncol=2, columnspacing=1.2)
                legend.get_frame().set_linewidth(2)
                # 美化图例文字
                for text in legend.get_texts():
                    text.set_color(self.colors['text_light'])
                    text.set_weight('bold')
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
        """构建清晰整洁的界面布局"""
        # 合理的窗口尺寸
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.patch.set_facecolor(self.colors['background_dark'])
        
        # 简洁的标题
        self.fig.suptitle('烟幕干扰三维仿真系统', fontsize=16, family='Microsoft YaHei', 
                         color=self.colors['text_light'], weight='bold', y=0.95)
        
        # 清晰的网格布局：左侧3D图，右侧信息面板，底部图表和控制
        gs = self.fig.add_gridspec(3, 3, 
                                   height_ratios=[2.5, 1.2, 0.6], 
                                   width_ratios=[2.0, 0.8, 0.8],
                                   hspace=0.25, wspace=0.15)
        
        # 主3D可视化区域 - 占据左侧大部分空间
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax3d.set_facecolor(self.colors['background_panel'])
        
        # 右上：参数信息面板
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_info.axis('off')
        self.ax_info.set_facecolor(self.colors['background_panel'])
        
        # 右上角：遮蔽统计面板
        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor(self.colors['background_panel'])
        
        # 底部左：视线半角图
        self.ax_area = self.fig.add_subplot(gs[1, 0])
        self.ax_area.set_facecolor(self.colors['background_panel'])
        
        # 底部中：距离变化图
        self.ax_dist = self.fig.add_subplot(gs[1, 1])
        self.ax_dist.set_facecolor(self.colors['background_panel'])
        
        # 底部右：空白区域用于未来扩展
        self.ax_extra = self.fig.add_subplot(gs[1, 2])
        self.ax_extra.axis('off')
        self.ax_extra.set_facecolor(self.colors['background_panel'])
        
        # 最底部：控制面板（跨所有列）
        self.ax_control = self.fig.add_subplot(gs[2, :])
        self.ax_control.axis('off')
        self.ax_control.set_facecolor(self.colors['background_dark'])
        
        # 调整边距
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.08, hspace=0.25, wspace=0.15)

        # 分析数据预计算
        ts, angles_deg, dists = self.analyze_projection_area()
        self._times, self._areas, self._dists = ts, angles_deg, dists
        self._occluded_ts, self._occluded_flags, self._occluded_total = self.analyze_full_occlusion(ts)

        # 美化图1：视线半角变化
        self.ax_area.plot(ts, angles_deg, color=self.colors['morandi_dusty_rose'], 
                         linewidth=4, alpha=0.9, label='视线半角')
        self.ax_area.fill_between(ts, 0, angles_deg, color=self.colors['morandi_dusty_rose'], 
                                 alpha=0.2, label='半角区域')
        self.ax_area.set_title('📐 视线半角变化分析', fontsize=14, family='Microsoft YaHei', 
                              color=self.colors['text_light'], weight='bold', pad=15)
        self.ax_area.set_xlabel('时间 (s)', fontsize=11, color=self.colors['text_light'], weight='bold')
        self.ax_area.set_ylabel('半角 (°)', fontsize=11, color=self.colors['text_light'], weight='bold')
        self.ax_area.grid(True, alpha=0.3, color=self.colors['morandi_slate'], linewidth=0.8)
        self.ax_area.tick_params(colors=self.colors['text_light'], labelsize=9)
        self._shade_occlusion(self.ax_area, ts, self._occluded_flags)
        self.ax_area.legend(loc='upper right', fontsize=9, framealpha=0.9, 
                           facecolor=self.colors['background_panel'])

        # 美化图2：距离变化
        self.ax_dist.plot(ts, dists, color=self.colors['morandi_soft_blue'], 
                         linewidth=4, alpha=0.9, label='M1-RO距离')
        self.ax_dist.fill_between(ts, min(dists)*0.8, dists, color=self.colors['morandi_soft_blue'], 
                                 alpha=0.2, label='距离变化')
        self.ax_dist.axhline(y=self.RO_radius, color=self.colors['accent_crimson'], 
                            linestyle='--', linewidth=2, alpha=0.8, label='RO半径')
        self.ax_dist.set_title('📏 M1到RO距离分析', fontsize=14, family='Microsoft YaHei', 
                              color=self.colors['text_light'], weight='bold', pad=15)
        self.ax_dist.set_xlabel('时间 (s)', fontsize=11, color=self.colors['text_light'], weight='bold')
        self.ax_dist.set_ylabel('距离 (m)', fontsize=11, color=self.colors['text_light'], weight='bold')
        self.ax_dist.grid(True, alpha=0.3, color=self.colors['morandi_slate'], linewidth=0.8)
        self.ax_dist.tick_params(colors=self.colors['text_light'], labelsize=9)
        self._shade_occlusion(self.ax_dist, ts, self._occluded_flags)
        self.ax_dist.legend(loc='upper right', fontsize=9, framealpha=0.9, 
                           facecolor=self.colors['background_panel'])

        # 新增：遮蔽效果统计面板
        occlusion_stats = self._create_occlusion_stats()
        self.occlusion_text = self.ax_stats.text(0.05, 0.95, occlusion_stats, 
                                                    va='top', ha='left', fontsize=10, 
                                                    family='Microsoft YaHei', color=self.colors['text_light'],
                                                    transform=self.ax_stats.transAxes,
                                                    bbox=dict(boxstyle="round,pad=0.8", 
                                                             facecolor=self.colors['background_panel'], 
                                                             edgecolor=self.colors['morandi_olive'], 
                                                             linewidth=2, alpha=0.95))

        # 美化参数信息面板
        self.info_text = self.ax_info.text(0.05, 0.95, self._compose_info_text(0.0), 
                                          va='top', ha='left', fontsize=9, 
                                          family='Microsoft YaHei', color=self.colors['text_light'],
                                          transform=self.ax_info.transAxes,
                                          bbox=dict(boxstyle="round,pad=1.0", 
                                                   facecolor=self.colors['background_panel'], 
                                                   edgecolor=self.colors['morandi_sage'], 
                                                   linewidth=2, alpha=0.95))

        # 初始化3D场景
        try:
            if hasattr(self.ax3d, 'view_init'):
                self.ax3d.view_init(elev=self._default_view[0], azim=self._default_view[1])
        except Exception:
            pass

        # 绘制初始帧
        try:
            self._draw_scene(self.ax3d, 0.0)
            if self.info_text is not None:
                self.info_text.set_text(self._compose_info_text(0.0))
        except Exception as e:
            print(f"初始化绘制失败: {e}")

        # 简洁的控制按钮区域
        try:
            # 优化按钮布局参数
            btn_height = 0.04
            btn_width = 0.07
            btn_spacing = 0.09
            btn_y = 0.02
            
            # 播放按钮
            ax_btn_play = plt.axes((0.12, btn_y, btn_width, btn_height))
            self.btn_play = Button(ax_btn_play, '▶️ 播放', color=self.colors['morandi_sage'], 
                                  hovercolor=self.colors['morandi_olive'])
            
            # 暂停按钮
            ax_btn_pause = plt.axes((0.12 + btn_spacing, btn_y, btn_width, btn_height))
            self.btn_pause = Button(ax_btn_pause, '⏸️ 暂停', color=self.colors['morandi_mauve'], 
                                   hovercolor=self.colors['morandi_dusty_rose'])
            
            # 重置按钮
            ax_btn_reset = plt.axes((0.12 + 2*btn_spacing, btn_y, btn_width, btn_height))
            self.btn_reset = Button(ax_btn_reset, '🔄 重置', color=self.colors['morandi_beige'], 
                                   hovercolor=self.colors['morandi_terracotta'])

            # 时间进度滑块
            slider_y = btn_y + 0.01
            slider_width = 0.35
            slider_height = 0.03  # 增加滑块高度
            ax_slider = plt.axes((0.45, slider_y, slider_width, slider_height), 
                                facecolor=self.colors['background_panel'])
            self.slider = Slider(ax_slider, '⏰ 时间进度', 0.0, self.total_time, valinit=0.0, 
                               color=self.colors['morandi_soft_blue'], 
                               facecolor=self.colors['background_panel'])
            ax_slider.spines['bottom'].set_color(self.colors['morandi_soft_blue'])
            ax_slider.spines['top'].set_color(self.colors['morandi_soft_blue'])
            ax_slider.spines['right'].set_color(self.colors['morandi_soft_blue'])
            ax_slider.spines['left'].set_color(self.colors['morandi_soft_blue'])

            # 🔢 播放速度控制滑块
            speed_slider_x = 0.87
            ax_speed = plt.axes((speed_slider_x, slider_y, 0.10, slider_height), 
                               facecolor=self.colors['background_panel'])
            ax_speed = plt.axes((0.50, slider_y, 0.15, 0.025))
            self.speed_slider = Slider(ax_speed, '速度', 0.1, 3.0, valinit=1.0, 
                                      color=self.colors['accent_gold'])
            
            # 状态显示 - 简洁版本
            status_y = 0.98
            self.status_text = self.fig.text(0.5, status_y, "⏱️ 时间: 0.0s | 遮蔽状态: 无", 
                                           ha='center', va='top', fontsize=12, 
                                           color=self.colors['text_light'],
                                           bbox=dict(boxstyle="round,pad=0.5", 
                                                    facecolor=self.colors['background_panel'], 
                                                    alpha=0.9))

            # 绑定事件处理器
            self.btn_play.on_clicked(self._on_play)
            self.btn_pause.on_clicked(self._on_pause)
            self.btn_reset.on_clicked(self._on_reset)
            self.slider.on_changed(self._on_slider)
            self.speed_slider.on_changed(self._on_speed_change)
            
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
                f"警告: M1位于球体内部\n"
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

    def _create_occlusion_stats(self) -> str:
        """创建遮蔽效果统计信息"""
        total_val = self._occluded_total if self._occluded_total is not None else 0.0
        total_time = self.total_time
        occlusion_ratio = (total_val / total_time * 100) if total_time > 0 else 0.0
        
        stats = (
            f"遮蔽效果分析\n"
            f"================\n"
            f"目标类型: RO真目标\n"
            f"烟团半径: {self.R_smoke:.1f}m\n"
            f"起爆时间: {self.t_det:.1f}s\n"
            f"下沉速度: {self.smoke_v_down:.1f}m/s\n"
            f"持续时间: {self.smoke_duration:.1f}s\n"
            f"总遮蔽时长: {total_val:.2f}s\n"
            f"遮蔽效率: {occlusion_ratio:.1f}%\n"
            f"================\n"
            f"{'轻度遮蔽' if occlusion_ratio < 30 else '中度遮蔽' if occlusion_ratio < 60 else '高度遮蔽'}"
        )
        return stats

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

    def _on_speed_change(self, val):
        """处理播放速度变化"""
        self.play_speed = float(val)

    def _update_frame(self, frame_idx: int):
        t = frame_idx * self.dt
        try:
            self._draw_scene(self.ax3d, t)
            if self.info_text is not None:
                self.info_text.set_text(self._compose_info_text(t))
            if self.status_text is not None:
                occluded = self.is_fully_occluded(t)
                status_color = self.colors['accent_crimson'] if occluded else self.colors['morandi_sage']
                status_icon = "🚨" if occluded else "👁️"
                status_text = "完全遮蔽" if occluded else "视野清晰"
                self.status_text.set_text(f"🕐 时间: {t:.1f}s | {status_icon} 遮蔽状态: {status_text}")
                self.status_text.set_color(status_color)
                # 更新边框颜色
                try:
                    bbox = self.status_text.get_bbox_patch()
                    if bbox is not None:
                        bbox.set_edgecolor(status_color)
                except Exception:
                    pass
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
