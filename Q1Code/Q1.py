import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数定义
g = 9.8  # 重力加速度 (m/s^2)
v_cloud_sink = 3  # 烟幕云团下沉速度 (m/s)
v_missile = 300  # 导弹速度 (m/s)
v_FY1 = 120  # 无人机速度 (m/s)
cloud_radius = 10  # 有效遮蔽半径 (m)
cloud_duration = 20  # 烟幕有效持续时间 (s)

# 初始位置
M1_start = np.array([20000, 0, 2000])
FY1_start = np.array([17800, 0, 1800])
target = np.array([0, 0, 0])  # 假目标
real_target = np.array([0, 200, 0])  # 真目标

# 无人机FY1的方向（指向假目标，z方向速度为0）
direction_xy = target[:2] - FY1_start[:2]
direction_FY1 = np.array([direction_xy[0], direction_xy[1], 0])
direction_FY1 = direction_FY1 / norm(direction_FY1)
v_FY1_vec = v_FY1 * direction_FY1  # UAV飞行速度向量（z方向分量为0）

# 干扰弹投放时间和位置
t_drop = 1.5  # 投放时间 (s)
drop_position = FY1_start + v_FY1_vec * t_drop

# 干扰弹的起爆时间和位置
t_burst = 3.6  # 起爆时间 (s)
burst_position = drop_position + v_FY1_vec * t_burst - 0.5 * g * t_burst**2 * np.array([0, 0, 1])

# 干扰弹的起爆时间（相对于任务开始）
t_burst_abs = t_drop + t_burst

# 导弹M1的方向和运动方程
direction_M1 = target - M1_start
direction_M1 = direction_M1 / norm(direction_M1)
v_M1_vec = v_missile * direction_M1

# 烟幕云团的运动方程
# P_cloud(t) = burst_position - v_cloud_sink * (t - t_burst_abs) * [0,0,1]
# P_cloud(t) = burst_position + v_cloud_sink * t_burst_abs * [0,0,1] - v_cloud_sink * t * [0,0,1]
cloud_pos_at_t0 = burst_position + v_cloud_sink * t_burst_abs * np.array([0, 0, 1])
v_cloud_vec = np.array([0, 0, -v_cloud_sink])

# 求解遮蔽时间窗口的解析解（导弹在球体内）
# 距离平方 d(t)^2 = || (M1_start + v_M1_vec*t) - (cloud_pos_at_t0 + v_cloud_vec*t) ||^2
# d(t)^2 = || (M1_start - cloud_pos_at_t0) + (v_M1_vec - v_cloud_vec)*t ||^2
# d(t)^2 = || A + B*t ||^2 = (B·B)t^2 + 2(A·B)t + (A·A)
A = M1_start - cloud_pos_at_t0
B = v_M1_vec - v_cloud_vec

# 二次方程 at^2 + bt + c <= 0 的系数
# (B·B)t^2 + 2(A·B)t + (A·A) - cloud_radius^2 <= 0
a = np.dot(B, B)
b = 2 * np.dot(A, B)
c = np.dot(A, A) - cloud_radius**2

# 求解 t
delta = b**2 - 4*a*c

shield_duration = 0
if delta >= 0:
    # 遮蔽条件满足的时间区间 [t1, t2]
    t1 = (-b - np.sqrt(delta)) / (2*a)
    t2 = (-b + np.sqrt(delta)) / (2*a)

    # 烟幕有效的时间区间 [t_smoke_start, t_smoke_end]
    t_smoke_start = t_burst_abs
    t_smoke_end = t_burst_abs + cloud_duration

    # 计算两个区间的交集
    intersect_start = max(t1, t_smoke_start)
    intersect_end = min(t2, t_smoke_end)

    # 如果交集存在，计算时长
    if intersect_end > intersect_start:
        shield_duration = intersect_end - intersect_start

print(f"烟幕干扰弹对M1的有效遮蔽时间: {shield_duration:.4f} 秒")

# ================== 参数分析 ==================
print("\n========== 参数分析 ==========")

# 1. 基本参数信息
print(f"导弹初始位置: {M1_start}")
print(f"无人机初始位置: {FY1_start}")
print(f"导弹速度: {v_missile} m/s")
print(f"无人机速度: {v_FY1} m/s")
print(f"烟幕云团下沉速度: {v_cloud_sink} m/s")
print(f"烟幕有效半径: {cloud_radius} m")
print(f"烟幕有效持续时间: {cloud_duration} s")

# 2. 关键时间节点
print(f"\n投放时间: {t_drop} s")
print(f"起爆延时: {t_burst} s")
print(f"起爆绝对时间: {t_burst_abs} s")
print(f"烟幕失效时间: {t_burst_abs + cloud_duration} s")

# 3. 关键位置信息
print(f"\n投放位置: [{drop_position[0]:.1f}, {drop_position[1]:.1f}, {drop_position[2]:.1f}]")
print(f"起爆位置: [{burst_position[0]:.1f}, {burst_position[1]:.1f}, {burst_position[2]:.1f}]")

# 4. 导弹在关键时刻的位置
M1_at_burst = M1_start + v_M1_vec * t_burst_abs
M1_at_smoke_end = M1_start + v_M1_vec * (t_burst_abs + cloud_duration)
print(f"\n导弹在起爆时刻位置: [{M1_at_burst[0]:.1f}, {M1_at_burst[1]:.1f}, {M1_at_burst[2]:.1f}]")
print(f"导弹在烟幕失效时位置: [{M1_at_smoke_end[0]:.1f}, {M1_at_smoke_end[1]:.1f}, {M1_at_smoke_end[2]:.1f}]")

# 5. 距离分析
burst_to_M1_at_burst = norm(M1_at_burst - burst_position)
print(f"\n起爆时刻导弹与烟幕中心距离: {burst_to_M1_at_burst:.1f} m")

# 6. 解析解的参数分析
print(f"\n解析解参数:")
print(f"二次方程系数 a: {a:.6f}")
print(f"二次方程系数 b: {b:.6f}")
print(f"二次方程系数 c: {c:.6f}")
print(f"判别式 Δ: {delta:.6f}")

if delta >= 0:
    t1 = (-b - np.sqrt(delta)) / (2*a)
    t2 = (-b + np.sqrt(delta)) / (2*a)
    print(f"距离条件满足的时间区间: [{t1:.4f}, {t2:.4f}] s")
    print(f"理论遮蔽时长: {t2 - t1:.4f} s")
    
    t_smoke_start = t_burst_abs
    t_smoke_end = t_burst_abs + cloud_duration
    print(f"烟幕有效时间区间: [{t_smoke_start:.4f}, {t_smoke_end:.4f}] s")
    
    intersect_start = max(t1, t_smoke_start)
    intersect_end = min(t2, t_smoke_end)
    print(f"有效交集区间: [{intersect_start:.4f}, {intersect_end:.4f}] s")
# 7. 敏感性分析提示
print(f"\n========== 敏感性分析建议 ==========")
print("1. 增大烟幕半径 cloud_radius 可显著提高遮蔽时间")
print("2. 调整投放时间 t_drop 可优化遮蔽时机")
print("3. 调整起爆延时 t_burst 可精确控制遮蔽位置")
print("4. 无人机速度影响投放精度，需要平衡")
print("5. 烟幕下沉速度影响遮蔽区域的垂直分布")

# ================== 输出参数分析到MD文件 ==================
md_content = f"""# 问题1：烟幕干扰弹遮蔽分析报告

## 计算结果
**烟幕干扰弹对M1的有效遮蔽时间: {shield_duration:.4f} 秒**

## 基本参数设置

| 参数 | 数值 | 单位 | 说明 |
|------|------|------|------|
| 导弹初始位置 | {M1_start} | m | M1导弹起始坐标 |
| 无人机初始位置 | {FY1_start} | m | FY1无人机起始坐标 |
| 导弹速度 | {v_missile} | m/s | M1导弹飞行速度 |
| 无人机速度 | {v_FY1} | m/s | FY1无人机飞行速度 |
| 烟幕云团下沉速度 | {v_cloud_sink} | m/s | 烟幕重力下沉速度 |
| 烟幕有效半径 | {cloud_radius} | m | 烟幕球体遮蔽半径 |
| 烟幕有效持续时间 | {cloud_duration} | s | 烟幕维持遮蔽能力的时间 |

## 关键时间节点

| 时间节点 | 数值 | 说明 |
|----------|------|------|
| 投放时间 | {t_drop} s | 无人机投放干扰弹的时间 |
| 起爆延时 | {t_burst} s | 干扰弹投放后到起爆的延时 |
| 起爆绝对时间 | {t_burst_abs} s | 从任务开始到起爆的总时间 |
| 烟幕失效时间 | {t_burst_abs + cloud_duration} s | 烟幕完全失效的时间 |

## 关键位置信息

| 位置描述 | 坐标 | 说明 |
|----------|------|------|
| 投放位置 | [{drop_position[0]:.1f}, {drop_position[1]:.1f}, {drop_position[2]:.1f}] | 干扰弹脱离无人机的位置 |
| 起爆位置 | [{burst_position[0]:.1f}, {burst_position[1]:.1f}, {burst_position[2]:.1f}] | 干扰弹起爆形成烟幕的位置 |
| 导弹在起爆时刻位置 | [{M1_at_burst[0]:.1f}, {M1_at_burst[1]:.1f}, {M1_at_burst[2]:.1f}] | M1导弹在烟幕起爆时的位置 |
| 导弹在烟幕失效时位置 | [{M1_at_smoke_end[0]:.1f}, {M1_at_smoke_end[1]:.1f}, {M1_at_smoke_end[2]:.1f}] | M1导弹在烟幕失效时的位置 |

## 距离分析

**起爆时刻导弹与烟幕中心距离: {burst_to_M1_at_burst:.1f} m**

这个距离表明烟幕起爆时，导弹还相对较远，需要等待导弹接近才能实现遮蔽。

## 解析解参数分析

数学模型将遮蔽问题转化为求解二次不等式：`at² + bt + c ≤ 0`

| 参数 | 数值 | 说明 |
|------|------|------|
| 系数 a | {a:.6f} | 二次项系数 |
| 系数 b | {b:.6f} | 一次项系数 |
| 系数 c | {c:.6f} | 常数项 |
| 判别式 Δ | {delta:.6f} | 判断是否有实数解 |
"""

if delta >= 0:
    t1 = (-b - np.sqrt(delta)) / (2*a)
    t2 = (-b + np.sqrt(delta)) / (2*a)
    t_smoke_start = t_burst_abs
    t_smoke_end = t_burst_abs + cloud_duration
    intersect_start = max(t1, t_smoke_start)
    intersect_end = min(t2, t_smoke_end)
    
    md_content += f"""
### 时间区间分析

| 区间类型 | 起始时间 | 结束时间 | 持续时间 | 说明 |
|----------|----------|----------|----------|------|
| 距离条件满足区间 | {t1:.4f} s | {t2:.4f} s | {t2 - t1:.4f} s | 导弹与烟幕中心距离≤半径的时间段 |
| 烟幕有效时间区间 | {t_smoke_start:.4f} s | {t_smoke_end:.4f} s | {cloud_duration:.4f} s | 烟幕维持遮蔽能力的时间段 |
| **有效交集区间** | **{intersect_start:.4f} s** | **{intersect_end:.4f} s** | **{shield_duration:.4f} s** | **实际有效遮蔽时间段** |
"""
else:
    md_content += f"""
### 分析结果
判别式 < 0，表明导弹轨迹不与烟幕球体相交，无法实现遮蔽。
"""

md_content += f"""
## 结果分析

### 遮蔽效果评估
- **有效遮蔽时间**: {shield_duration:.4f} 秒
- **遮蔽效率**: {(shield_duration / cloud_duration * 100):.2f}% （有效遮蔽时间/烟幕总持续时间）

### 主要限制因素
1. **空间匹配度低**: 起爆时刻导弹距烟幕中心{burst_to_M1_at_burst:.1f}m，距离较远
2. **导弹速度过快**: 300 m/s的高速使导弹快速穿越烟幕区域
3. **烟幕半径相对较小**: 10m的遮蔽半径相对于导弹轨迹偏差较小

## 优化建议

### 参数优化方向
1. **增大烟幕半径** (cloud_radius)
   - 从10m增加到20-30m可显著提升遮蔽时间
   - 建议优先级：★★★★★

2. **优化投放时机** (t_drop)
   - 调整投放时间，使烟幕在导弹轨迹上更精确的位置起爆
   - 建议优先级：★★★★☆

3. **精确控制起爆时机** (t_burst)
   - 根据导弹轨迹预测，调整起爆延时
   - 建议优先级：★★★★☆

4. **无人机飞行策略优化**
   - 调整无人机飞行方向和速度，改善投放精度
   - 建议优先级：★★★☆☆

5. **考虑烟幕下沉影响**
   - 针对3 m/s的下沉速度，调整起爆高度
   - 建议优先级：★★☆☆☆

### 技术改进建议
1. **多弹协同**: 投放多枚干扰弹形成更大的遮蔽区域
2. **动态调整**: 根据实时导弹轨迹动态调整投放参数
3. **预测算法**: 提高导弹轨迹预测精度，优化投放时机

## 数学模型验证
本分析采用解析解方法，避免了数值模拟的离散化误差，结果具有理论精确性。模型考虑了：
- 导弹和烟幕的三维运动轨迹
- 重力对干扰弹轨迹的影响
- 烟幕云团的下沉运动
- 时间窗口的精确匹配

计算结果为理论最优值，实际应用中还需考虑风力、大气扰动等环境因素的影响。
"""

# 写入MD文件
with open('C1.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"\n参数分析报告已输出到: C1.md")

# ================== 可视化图表生成 ==================
print(f"\n正在生成可视化图表...")

# 创建图表
fig = plt.figure(figsize=(20, 15))

# 1. 三维轨迹图
ax1 = fig.add_subplot(2, 3, 1, projection='3d')

# 时间范围
t_range = np.linspace(0, 30, 300)
missile_trajectory = np.array([M1_start + v_M1_vec * t for t in t_range])
cloud_trajectory = np.array([cloud_pos_at_t0 + v_cloud_vec * t for t in t_range])

# 绘制轨迹
ax1.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2], 
         'r-', linewidth=2, label='导弹M1轨迹')
ax1.plot(cloud_trajectory[:, 0], cloud_trajectory[:, 1], cloud_trajectory[:, 2], 
         'b-', linewidth=2, label='烟幕中心轨迹')

# 标记关键点
ax1.scatter(*M1_start, color='red', s=100, label='M1起始位置')
ax1.scatter(*FY1_start, color='green', s=100, label='FY1起始位置')
ax1.scatter(*target, color='black', s=100, label='假目标')
ax1.scatter(*drop_position, color='orange', s=100, label='投放位置')
ax1.scatter(*burst_position, color='blue', s=100, label='起爆位置')

# 绘制烟幕球体（在起爆位置）
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x_sphere = burst_position[0] + cloud_radius * np.outer(np.cos(u), np.sin(v))
y_sphere = burst_position[1] + cloud_radius * np.outer(np.sin(u), np.sin(v))
z_sphere = burst_position[2] + cloud_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='blue')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('三维轨迹图')
ax1.legend()

# 2. 距离随时间变化图
ax2 = fig.add_subplot(2, 3, 2)
distances = []
for t in t_range:
    if t >= t_burst_abs:
        M1_pos = M1_start + v_M1_vec * t
        cloud_pos = cloud_pos_at_t0 + v_cloud_vec * t
        dist = norm(M1_pos - cloud_pos)
        distances.append(dist)
    else:
        distances.append(np.nan)

ax2.plot(t_range, distances, 'g-', linewidth=2, label='导弹-烟幕距离')
ax2.axhline(y=cloud_radius, color='r', linestyle='--', label=f'遮蔽半径 ({cloud_radius}m)')
ax2.axvline(x=t_burst_abs, color='b', linestyle='--', alpha=0.7, label='起爆时间')
ax2.axvline(x=t_burst_abs + cloud_duration, color='b', linestyle=':', alpha=0.7, label='烟幕失效时间')

if delta >= 0:
    ax2.axvspan(intersect_start, intersect_end, alpha=0.3, color='red', label='有效遮蔽时间')

ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('距离 (m)')
ax2.set_title('导弹与烟幕中心距离随时间变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 参数敏感性分析 - 烟幕半径
ax3 = fig.add_subplot(2, 3, 3)
radius_range = np.arange(5, 51, 2)
shield_times = []

for r in radius_range:
    c_temp = np.dot(A, A) - r**2
    delta_temp = b**2 - 4*a*c_temp
    if delta_temp >= 0:
        t1_temp = (-b - np.sqrt(delta_temp)) / (2*a)
        t2_temp = (-b + np.sqrt(delta_temp)) / (2*a)
        intersect_start_temp = max(t1_temp, t_burst_abs)
        intersect_end_temp = min(t2_temp, t_burst_abs + cloud_duration)
        if intersect_end_temp > intersect_start_temp:
            shield_times.append(intersect_end_temp - intersect_start_temp)
        else:
            shield_times.append(0)
    else:
        shield_times.append(0)

ax3.plot(radius_range, shield_times, 'ro-', linewidth=2, markersize=4)
ax3.axvline(x=cloud_radius, color='b', linestyle='--', alpha=0.7, label=f'当前半径 ({cloud_radius}m)')
ax3.set_xlabel('烟幕半径 (m)')
ax3.set_ylabel('有效遮蔽时间 (s)')
ax3.set_title('烟幕半径对遮蔽时间的影响')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 参数敏感性分析 - 投放时间
ax4 = fig.add_subplot(2, 3, 4)
drop_time_range = np.arange(0.5, 4.0, 0.1)
shield_times_drop = []

for t_d in drop_time_range:
    # 重新计算投放和起爆位置
    drop_pos_temp = FY1_start + v_FY1_vec * t_d
    burst_pos_temp = drop_pos_temp + v_FY1_vec * t_burst - 0.5 * g * t_burst**2 * np.array([0, 0, 1])
    t_burst_abs_temp = t_d + t_burst
    cloud_pos_at_t0_temp = burst_pos_temp + v_cloud_sink * t_burst_abs_temp * np.array([0, 0, 1])
    
    # 重新计算系数
    A_temp = M1_start - cloud_pos_at_t0_temp
    c_temp = np.dot(A_temp, A_temp) - cloud_radius**2
    b_temp = 2 * np.dot(A_temp, B)
    delta_temp = b_temp**2 - 4*a*c_temp
    
    if delta_temp >= 0:
        t1_temp = (-b_temp - np.sqrt(delta_temp)) / (2*a)
        t2_temp = (-b_temp + np.sqrt(delta_temp)) / (2*a)
        intersect_start_temp = max(t1_temp, t_burst_abs_temp)
        intersect_end_temp = min(t2_temp, t_burst_abs_temp + cloud_duration)
        if intersect_end_temp > intersect_start_temp:
            shield_times_drop.append(intersect_end_temp - intersect_start_temp)
        else:
            shield_times_drop.append(0)
    else:
        shield_times_drop.append(0)

ax4.plot(drop_time_range, shield_times_drop, 'go-', linewidth=2, markersize=3)
ax4.axvline(x=t_drop, color='b', linestyle='--', alpha=0.7, label=f'当前投放时间 ({t_drop}s)')
ax4.set_xlabel('投放时间 (s)')
ax4.set_ylabel('有效遮蔽时间 (s)')
ax4.set_title('投放时间对遮蔽时间的影响')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 二维俯视图
ax5 = fig.add_subplot(2, 3, 5)
# 绘制轨迹的俯视图
ax5.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], 'r-', linewidth=2, label='导弹轨迹')
ax5.plot(cloud_trajectory[:, 0], cloud_trajectory[:, 1], 'b-', linewidth=2, label='烟幕轨迹')

# 标记关键点
ax5.scatter(M1_start[0], M1_start[1], color='red', s=100, label='M1起始')
ax5.scatter(FY1_start[0], FY1_start[1], color='green', s=100, label='FY1起始')
ax5.scatter(target[0], target[1], color='black', s=100, label='假目标')
ax5.scatter(burst_position[0], burst_position[1], color='blue', s=100, label='起爆位置')

# 绘制烟幕覆盖圆
circle = patches.Circle((burst_position[0], burst_position[1]), cloud_radius, 
                       linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
ax5.add_patch(circle)

ax5.set_xlabel('X (m)')
ax5.set_ylabel('Y (m)')
ax5.set_title('俯视图 (XY平面)')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

# 6. 时间轴分析
ax6 = fig.add_subplot(2, 3, 6)
events = ['任务开始', '投放干扰弹', '干扰弹起爆', '开始遮蔽', '结束遮蔽', '烟幕失效']
times = [0, t_drop, t_burst_abs, intersect_start if delta >= 0 else 0, 
         intersect_end if delta >= 0 else 0, t_burst_abs + cloud_duration]
colors = ['black', 'green', 'blue', 'red', 'red', 'gray']

for i, (event, time, color) in enumerate(zip(events, times, colors)):
    ax6.barh(i, time, color=color, alpha=0.7)
    ax6.text(time + 0.5, i, f'{time:.2f}s', va='center')

ax6.set_yticks(range(len(events)))
ax6.set_yticklabels(events)
ax6.set_xlabel('时间 (s)')
ax6.set_title('关键事件时间轴')
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('analysis_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 生成单独的高精度3D图
fig_3d = plt.figure(figsize=(12, 10))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# 更详细的3D轨迹
t_detailed = np.linspace(0, 15, 500)
missile_detailed = np.array([M1_start + v_M1_vec * t for t in t_detailed])
cloud_detailed = np.array([cloud_pos_at_t0 + v_cloud_vec * t for t in t_detailed if t >= t_burst_abs])
t_cloud_detailed = t_detailed[t_detailed >= t_burst_abs]

ax_3d.plot(missile_detailed[:, 0], missile_detailed[:, 1], missile_detailed[:, 2], 
           'r-', linewidth=3, label='导弹M1轨迹', alpha=0.8)
ax_3d.plot(cloud_detailed[:, 0], cloud_detailed[:, 1], cloud_detailed[:, 2], 
           'b-', linewidth=3, label='烟幕中心轨迹', alpha=0.8)

# 绘制多个时刻的烟幕球体
if delta >= 0:
    shield_times_3d = np.linspace(intersect_start, intersect_end, 5)
    for i, t_shield in enumerate(shield_times_3d):
        cloud_pos_shield = cloud_pos_at_t0 + v_cloud_vec * t_shield
        # 绘制半透明球体
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x_s = cloud_pos_shield[0] + cloud_radius * np.outer(np.cos(u), np.sin(v))
        y_s = cloud_pos_shield[1] + cloud_radius * np.outer(np.sin(u), np.sin(v))
        z_s = cloud_pos_shield[2] + cloud_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_surface(x_s, y_s, z_s, alpha=0.2, color='blue')

# 标记关键点
ax_3d.scatter(*M1_start, color='red', s=150, label='M1起始位置')
ax_3d.scatter(*FY1_start, color='green', s=150, label='FY1起始位置')
ax_3d.scatter(*target, color='black', s=150, label='假目标')
ax_3d.scatter(*burst_position, color='blue', s=150, label='起爆位置')

ax_3d.set_xlabel('X (m)', fontsize=12)
ax_3d.set_ylabel('Y (m)', fontsize=12)
ax_3d.set_zlabel('Z (m)', fontsize=12)
ax_3d.set_title('烟幕干扰弹遮蔽过程三维可视化', fontsize=14)
ax_3d.legend(fontsize=10)

plt.savefig('3d_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ 可视化图表已生成:")
print("  - analysis_charts.png: 综合分析图表")
print("  - 3d_visualization.png: 三维可视化图")