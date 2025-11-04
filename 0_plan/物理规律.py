"""
训练集物理规律分析脚本
目标：提取速度、加速度、转向等物理约束参数
"""

import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from official.CoordinateConvert import lonlat_to_xy

def calculate_distance(x1, y1, x2, y2):
    """计算两点间距离（米）"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_bearing(x1, y1, x2, y2):
    """计算方位角（度）"""
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dx, dy))
    return (angle + 360) % 360

def normalize_angle_diff(angle_diff):
    """标准化角度差到[-180, 180]"""
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    return angle_diff

def analyze_trajectory_file(filepath):
    """分析单个轨迹文件"""
    df = pd.read_csv(filepath)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time')
    
    # 转换为平面坐标
    xy_coords = [lonlat_to_xy(lon, lat) for lon, lat in zip(df['LON'], df['LAT'])]
    df['X'] = [xy[0] for xy in xy_coords]
    df['Y'] = [xy[1] for xy in xy_coords]
    
    results = {
        'velocities': [],      # 速度 (m/s)
        'accelerations': [],   # 加速度 (m/s²)
        'turn_rates': [],      # 转向角速度 (度/分钟)
        'bearings': [],        # 航向角 (度)
        'positions': []        # 位置 (x, y) 用于区域分析
    }
    
    prev_velocity = None
    prev_bearing = None
    
    for i in range(len(df) - 1):
        # 时间差（秒）
        time_diff = (df.iloc[i+1]['Time'] - df.iloc[i]['Time']).total_seconds()
        if time_diff == 0:
            continue
        
        # 距离（米）
        dist = calculate_distance(
            df.iloc[i]['X'], df.iloc[i]['Y'],
            df.iloc[i+1]['X'], df.iloc[i+1]['Y']
        )
        
        # 速度 (m/s)
        velocity = dist / time_diff
        results['velocities'].append(velocity)
        
        # 航向角
        bearing = calculate_bearing(
            df.iloc[i]['X'], df.iloc[i]['Y'],
            df.iloc[i+1]['X'], df.iloc[i+1]['Y']
        )
        results['bearings'].append(bearing)
        
        # 加速度 (m/s²)
        if prev_velocity is not None:
            acceleration = (velocity - prev_velocity) / time_diff
            results['accelerations'].append(acceleration)
        
        # 转向角速度 (度/分钟)
        if prev_bearing is not None:
            angle_diff = normalize_angle_diff(bearing - prev_bearing)
            turn_rate = angle_diff / (time_diff / 60)  # 转为度/分钟
            results['turn_rates'].append(turn_rate)
        
        # 保存位置和航向（用于区域分析）
        results['positions'].append({
            'x': df.iloc[i]['X'],
            'y': df.iloc[i]['Y'],
            'lon': df.iloc[i]['LON'],
            'lat': df.iloc[i]['LAT'],
            'bearing': bearing
        })
        
        prev_velocity = velocity
        prev_bearing = bearing
    
    return results

def main():
    print("=" * 60)
    print("开始分析训练集物理规律...")
    print("=" * 60)
    
    # 读取所有ground_truth文件
    gt_files = glob.glob('official/train/ground_truth/*.csv')
    print(f"\n找到 {len(gt_files)} 个训练轨迹文件")
    
    # 汇总所有数据
    all_velocities = []
    all_accelerations = []
    all_turn_rates = []
    all_bearings = []
    all_positions = []
    
    print("\n正在处理文件...")
    for idx, filepath in enumerate(gt_files):
        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx+1}/{len(gt_files)}")
        
        try:
            results = analyze_trajectory_file(filepath)
            all_velocities.extend(results['velocities'])
            all_accelerations.extend(results['accelerations'])
            all_turn_rates.extend(results['turn_rates'])
            all_bearings.extend(results['bearings'])
            all_positions.extend(results['positions'])
        except Exception as e:
            print(f"  警告: 处理 {filepath} 时出错: {e}")
            continue
    
    # 转换为numpy数组
    all_velocities = np.array(all_velocities)
    all_accelerations = np.array(all_accelerations)
    all_turn_rates = np.array(all_turn_rates)
    all_bearings = np.array(all_bearings)
    
    # 计算统计数据
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    
    # 速度统计
    print("\n【速度分布】(m/s)")
    print(f"  最小值:     {np.min(all_velocities):.3f} m/s")
    print(f"  5%分位:     {np.percentile(all_velocities, 5):.3f} m/s")
    print(f"  中位数:     {np.median(all_velocities):.3f} m/s")
    print(f"  平均值:     {np.mean(all_velocities):.3f} m/s")
    print(f"  95%分位:    {np.percentile(all_velocities, 95):.3f} m/s")
    print(f"  99%分位:    {np.percentile(all_velocities, 99):.3f} m/s")
    print(f"  最大值:     {np.max(all_velocities):.3f} m/s")
    print(f"  标准差:     {np.std(all_velocities):.3f} m/s")
    
    # 加速度统计
    print("\n【加速度分布】(m/s²)")
    print(f"  最小值:     {np.min(all_accelerations):.4f} m/s²")
    print(f"  5%分位:     {np.percentile(all_accelerations, 5):.4f} m/s²")
    print(f"  中位数:     {np.median(all_accelerations):.4f} m/s²")
    print(f"  平均值:     {np.mean(all_accelerations):.4f} m/s²")
    print(f"  95%分位:    {np.percentile(all_accelerations, 95):.4f} m/s²")
    print(f"  99%分位:    {np.percentile(all_accelerations, 99):.4f} m/s²")
    print(f"  最大值:     {np.max(all_accelerations):.4f} m/s²")
    print(f"  绝对值95%:  {np.percentile(np.abs(all_accelerations), 95):.4f} m/s²")
    print(f"  绝对值99%:  {np.percentile(np.abs(all_accelerations), 99):.4f} m/s²")
    
    # 转向角速度统计
    print("\n【转向角速度分布】(度/分钟)")
    print(f"  最小值:     {np.min(all_turn_rates):.3f} °/min")
    print(f"  5%分位:     {np.percentile(all_turn_rates, 5):.3f} °/min")
    print(f"  中位数:     {np.median(all_turn_rates):.3f} °/min")
    print(f"  平均值:     {np.mean(all_turn_rates):.3f} °/min")
    print(f"  95%分位:    {np.percentile(all_turn_rates, 95):.3f} °/min")
    print(f"  99%分位:    {np.percentile(all_turn_rates, 99):.3f} °/min")
    print(f"  最大值:     {np.max(all_turn_rates):.3f} °/min")
    print(f"  绝对值95%:  {np.percentile(np.abs(all_turn_rates), 95):.3f} °/min")
    print(f"  绝对值99%:  {np.percentile(np.abs(all_turn_rates), 99):.3f} °/min")
    
    # 航向角统计（按区域）
    print("\n【区域航向分析】")
    print("  将海域划分为网格，统计主航向...")
    
    # 网格化分析
    grid_size = 0.1  # 经纬度网格大小（约10km）
    grid_bearings = {}
    
    for pos in all_positions:
        grid_x = int(pos['lon'] / grid_size)
        grid_y = int(pos['lat'] / grid_size)
        grid_key = (grid_x, grid_y)
        
        if grid_key not in grid_bearings:
            grid_bearings[grid_key] = []
        grid_bearings[grid_key].append(pos['bearing'])
    
    # 找出主要航向的网格
    print(f"  活跃网格数: {len(grid_bearings)}")
    
    major_grids = []
    for grid_key, bearings in grid_bearings.items():
        if len(bearings) >= 50:  # 至少50个观测点
            avg_bearing = np.mean(bearings)
            std_bearing = np.std(bearings)
            major_grids.append({
                'grid': grid_key,
                'lon_center': grid_key[0] * grid_size,
                'lat_center': grid_key[1] * grid_size,
                'count': len(bearings),
                'main_bearing': avg_bearing,
                'bearing_std': std_bearing
            })
    
    print(f"  主要航线网格: {len(major_grids)} 个")
    if len(major_grids) > 0:
        print(f"  (显示前10个)")
        for i, grid in enumerate(sorted(major_grids, key=lambda x: x['count'], reverse=True)[:10]):
            print(f"    网格{i+1}: 中心({grid['lon_center']:.2f}, {grid['lat_center']:.2f}), "
                  f"观测{grid['count']}次, 主航向{grid['main_bearing']:.1f}°")
    
    # 保存约束参数到JSON
    constraints = {
        "velocity": {
            "min": float(np.min(all_velocities)),
            "max": float(np.max(all_velocities)),
            "median": float(np.median(all_velocities)),
            "mean": float(np.mean(all_velocities)),
            "p05": float(np.percentile(all_velocities, 5)),
            "p95": float(np.percentile(all_velocities, 95)),
            "p99": float(np.percentile(all_velocities, 99)),
            "std": float(np.std(all_velocities)),
            "recommended_min": float(np.percentile(all_velocities, 1)),  # 1%分位作为软下限
            "recommended_max": float(np.percentile(all_velocities, 99))  # 99%分位作为软上限
        },
        "acceleration": {
            "min": float(np.min(all_accelerations)),
            "max": float(np.max(all_accelerations)),
            "median": float(np.median(all_accelerations)),
            "mean": float(np.mean(all_accelerations)),
            "p95_abs": float(np.percentile(np.abs(all_accelerations), 95)),
            "p99_abs": float(np.percentile(np.abs(all_accelerations), 99)),
            "recommended_max_abs": float(np.percentile(np.abs(all_accelerations), 99))  # 99%分位作为上限
        },
        "turn_rate": {
            "min": float(np.min(all_turn_rates)),
            "max": float(np.max(all_turn_rates)),
            "median": float(np.median(all_turn_rates)),
            "mean": float(np.mean(all_turn_rates)),
            "p95_abs": float(np.percentile(np.abs(all_turn_rates), 95)),
            "p99_abs": float(np.percentile(np.abs(all_turn_rates), 99)),
            "recommended_max_abs": float(np.percentile(np.abs(all_turn_rates), 99))  # 99%分位作为上限
        },
        "regional_bearings": major_grids[:50]  # 保存前50个主要航线网格
    }
    
    output_file = 'physical_constraints.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(constraints, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 约束参数已保存到: {output_file}")
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    # 输出建议
    print("\n【建议的约束参数】")
    print(f"  速度范围:    {constraints['velocity']['recommended_min']:.2f} ~ {constraints['velocity']['recommended_max']:.2f} m/s")
    print(f"  加速度上限:  ±{constraints['acceleration']['recommended_max_abs']:.4f} m/s²")
    print(f"  转向速度上限: ±{constraints['turn_rate']['recommended_max_abs']:.2f} °/min")

if __name__ == '__main__':
    main()

