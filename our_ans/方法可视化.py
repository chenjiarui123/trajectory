"""
在validation_set上验证参数效果
包括：
1. 测试不同参数组合
2. 可视化单交叉点船和多交叉点船的预测效果
3. 计算得分对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d
from pathlib import Path
from official.CoordinateConvert import lonlat_to_xy, xy_to_lonlat, space_intersection, adjust_angle
from official.sensor_config import SENSOR_INFO, ESM_SENSORS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_score(distance):
    """官方评分函数"""
    return 100 * np.exp(-0.0046 * distance)


def generate_trajectory(boat_id, radar_df, esm_data, sensor_xy_dict, params):
    """
    生成单艘船的轨迹
    
    params: {
        'extrap_minutes': 外推时长,
        'default_speed': 默认速度,
        'default_bearing': 默认航向
    }
    """
    boat_esm = esm_data[esm_data['BoatID'] == boat_id].copy()
    
    if len(boat_esm) == 0:
        return None, None
    
    # 找交叉点，同时收集单ESM信息
    grouped = boat_esm.groupby('Time')
    intersection_points = []
    single_esm_list = []
    
    for time, group in grouped:
        sensors = group['SensorID'].values
        azimuths = group['Azimuth'].values
        valid_sensors = [s for s in sensors if s in ESM_SENSORS]
        
        if len(valid_sensors) >= 2:
            s1_id, s2_id = valid_sensors[0], valid_sensors[1]
            idx1, idx2 = list(sensors).index(s1_id), list(sensors).index(s2_id)
            
            s1_xy = sensor_xy_dict[s1_id]
            s2_xy = sensor_xy_dict[s2_id]
            
            theta1 = adjust_angle(azimuths[idx1])
            theta2 = adjust_angle(azimuths[idx2])
            
            target_xy = space_intersection(s1_xy, s2_xy, theta1, theta2)
            
            intersection_points.append({
                'time': time,
                'x': target_xy[0],
                'y': target_xy[1]
            })
        
        elif len(valid_sensors) == 1:
            # 单ESM射线
            s_id = valid_sensors[0]
            idx = list(sensors).index(s_id)
            theta = adjust_angle(azimuths[idx])
            s_xy = sensor_xy_dict[s_id]
            
            single_esm_list.append({
                'sensor_xy': s_xy,
                'azimuth': theta,
                'time': time
            })
    
    if len(intersection_points) == 0:
        return None, 'no_crossing'
    
    results = []
    boat_type = 'single' if len(intersection_points) == 1 else 'multi'
    
    # ===== 单交叉点船 =====
    if len(intersection_points) == 1:
        point = intersection_points[0]
        anchor_time = point['time']
        
        # 使用固定航向和速度直线外推
        direction = np.array([
            np.sin(np.radians(params['default_bearing'])),
            np.cos(np.radians(params['default_bearing']))
        ])
        
        extrap_start = anchor_time - pd.Timedelta(minutes=params['extrap_minutes'])
        extrap_end = anchor_time + pd.Timedelta(minutes=params['extrap_minutes'])
        time_range = pd.date_range(start=extrap_start, end=extrap_end, freq='1min')
        
        for time in time_range:
            time_diff = (time - anchor_time).total_seconds()
            extrap_x = point['x'] + direction[0] * params['default_speed'] * time_diff
            extrap_y = point['y'] + direction[1] * params['default_speed'] * time_diff
            lon, lat = xy_to_lonlat(extrap_x, extrap_y)
            
            results.append({
                'Time': time,
                'LON': lon,
                'LAT': lat,
                'X': extrap_x,
                'Y': extrap_y
            })
    
    # ===== 多交叉点船 =====
    else:
        intersection_points = sorted(intersection_points, key=lambda p: p['time'])
        
        # 交叉点之间线性插值
        times = [p['time'] for p in intersection_points]
        xs = [p['x'] for p in intersection_points]
        ys = [p['y'] for p in intersection_points]
        
        start_time = times[0]
        end_time = times[-1]
        time_numeric = [(t - start_time).total_seconds() for t in times]
        
        x_interp = interp1d(time_numeric, xs, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
        y_interp = interp1d(time_numeric, ys, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        
        time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        for time in time_range:
            time_sec = (time - start_time).total_seconds()
            x = float(x_interp(time_sec))
            y = float(y_interp(time_sec))
            lon, lat = xy_to_lonlat(x, y)
            
            results.append({
                'Time': time,
                'LON': lon,
                'LAT': lat,
                'X': x,
                'Y': y
            })
        
        # 外推
        first_point = intersection_points[0]
        last_point = intersection_points[-1]
        second_point = intersection_points[1]
        second_last = intersection_points[-2]
        
        # 计算速度
        dx_end = last_point['x'] - second_last['x']
        dy_end = last_point['y'] - second_last['y']
        dt_end = (last_point['time'] - second_last['time']).total_seconds()
        velocity_end = np.array([dx_end / dt_end, dy_end / dt_end]) if dt_end > 0 else np.array([7.0, 0])
        
        dx_start = second_point['x'] - first_point['x']
        dy_start = second_point['y'] - first_point['y']
        dt_start = (second_point['time'] - first_point['time']).total_seconds()
        velocity_start = np.array([dx_start / dt_start, dy_start / dt_start]) if dt_start > 0 else velocity_end
        
        # 前向外推
        extrap_start = first_point['time'] - pd.Timedelta(minutes=params['extrap_minutes'])
        time_range_before = pd.date_range(start=extrap_start, 
                                          end=first_point['time'] - pd.Timedelta(minutes=1),
                                          freq='1min')
        
        for time in time_range_before:
            time_diff = (time - first_point['time']).total_seconds()
            extrap_x = first_point['x'] + velocity_start[0] * time_diff
            extrap_y = first_point['y'] + velocity_start[1] * time_diff
            lon, lat = xy_to_lonlat(extrap_x, extrap_y)
            
            results.append({
                'Time': time,
                'LON': lon,
                'LAT': lat,
                'X': extrap_x,
                'Y': extrap_y
            })
        
        # 后向外推
        extrap_end = last_point['time'] + pd.Timedelta(minutes=params['extrap_minutes'])
        time_range_after = pd.date_range(start=last_point['time'] + pd.Timedelta(minutes=1),
                                         end=extrap_end,
                                         freq='1min')
        
        for time in time_range_after:
            time_diff = (time - last_point['time']).total_seconds()
            extrap_x = last_point['x'] + velocity_end[0] * time_diff
            extrap_y = last_point['y'] + velocity_end[1] * time_diff
            lon, lat = xy_to_lonlat(extrap_x, extrap_y)
            
            results.append({
                'Time': time,
                'LON': lon,
                'LAT': lat,
                'X': extrap_x,
                'Y': extrap_y
            })
    
    return pd.DataFrame(results), boat_type


def evaluate_and_visualize(boat_id, params, show_plot=True):
    """
    评估并可视化单艘船的效果
    """
    # 读取数据
    radar_df = pd.read_csv('validation_set/radar_detection.csv')
    radar_df['Time'] = pd.to_datetime(radar_df['Time'])
    
    esm_data = radar_df[radar_df['Distance'] == -100].copy()
    
    sensor_xy_dict = {}
    for sid in ESM_SENSORS:
        info = SENSOR_INFO[sid]
        sensor_xy_dict[sid] = lonlat_to_xy(info['lon'], info['lat'])
    
    # 提取交叉点和单ESM（用于可视化）
    boat_esm = esm_data[esm_data['BoatID'] == boat_id]
    crossing_points = []
    single_esm_list = []
    
    for time, group in boat_esm.groupby('Time'):
        sensors = group['SensorID'].values
        azimuths = group['Azimuth'].values
        valid_sensors = [s for s in sensors if s in ESM_SENSORS]
        
        if len(valid_sensors) >= 2:
            s1_id, s2_id = valid_sensors[0], valid_sensors[1]
            idx1, idx2 = list(sensors).index(s1_id), list(sensors).index(s2_id)
            
            s1_xy = sensor_xy_dict[s1_id]
            s2_xy = sensor_xy_dict[s2_id]
            
            theta1 = adjust_angle(azimuths[idx1])
            theta2 = adjust_angle(azimuths[idx2])
            
            target_xy = space_intersection(s1_xy, s2_xy, theta1, theta2)
            
            crossing_points.append({
                'x': target_xy[0],
                'y': target_xy[1],
                'time': time
            })
        
        elif len(valid_sensors) == 1:
            # 单ESM
            s_id = valid_sensors[0]
            idx = list(sensors).index(s_id)
            theta = adjust_angle(azimuths[idx])
            s_xy = sensor_xy_dict[s_id]
            
            single_esm_list.append({
                'sensor_xy': s_xy,
                'azimuth': theta,
                'time': time
            })
    
    # 提取BoatID=-1的雷达点（用于密度热力图）
    radar_unknown = radar_df[radar_df['BoatID'] == -1].copy()
    radar_xy_list = []
    
    for _, row in radar_unknown.iterrows():
        sensor_info = SENSOR_INFO.get(row['SensorID'])
        if sensor_info is None:
            continue
        sensor_xy = lonlat_to_xy(sensor_info['lon'], sensor_info['lat'])
        theta = adjust_angle(row['Azimuth'])
        distance = row['Distance']
        
        target_x = sensor_xy[0] + distance * np.sin(np.radians(theta))
        target_y = sensor_xy[1] + distance * np.cos(np.radians(theta))
        
        radar_xy_list.append({'x': target_x, 'y': target_y})
    
    # 生成预测轨迹
    pred_df, boat_type = generate_trajectory(boat_id, radar_df, esm_data, sensor_xy_dict, params)
    
    if pred_df is None:
        print(f"船 {boat_id} 无交叉点")
        return None
    
    # 检查并排序
    pred_df = pred_df.sort_values('Time').reset_index(drop=True)
    
    # 检查是否有重复时间点
    duplicates = pred_df[pred_df.duplicated(subset=['Time'], keep=False)]
    if len(duplicates) > 0:
        print(f"  警告：船{boat_id}有{len(duplicates)}个重复时间点，去重中...")
        pred_df = pred_df.drop_duplicates(subset=['Time'], keep='first')
    
    # 打印轨迹信息
    print(f"  船{boat_id}({boat_type}): 预测{len(pred_df)}个点, 时间{pred_df['Time'].min()} ~ {pred_df['Time'].max()}")
    
    # 读取真值
    gt_file = f'validation_set/ground_truth/{boat_id}.csv'
    if not Path(gt_file).exists():
        print(f"船 {boat_id} 无真值数据")
        return None
    
    gt_df = pd.read_csv(gt_file)
    gt_df['Time'] = pd.to_datetime(gt_df['Time'])
    
    # 转换真值到XY
    gt_xy = []
    for _, row in gt_df.iterrows():
        x, y = lonlat_to_xy(row['LON'], row['LAT'])
        gt_xy.append({'Time': row['Time'], 'X': x, 'Y': y})
    gt_xy_df = pd.DataFrame(gt_xy)
    
    # 计算得分
    pred_df_indexed = pred_df.set_index('Time')
    gt_xy_indexed = gt_xy_df.set_index('Time')
    
    common_times = pred_df_indexed.index.intersection(gt_xy_indexed.index)
    
    if len(common_times) == 0:
        print(f"船 {boat_id} 无共同时间点")
        return None
    
    scores = []
    distances = []
    
    for time in common_times:
        pred_x = pred_df_indexed.loc[time, 'X']
        pred_y = pred_df_indexed.loc[time, 'Y']
        gt_x = gt_xy_indexed.loc[time, 'X']
        gt_y = gt_xy_indexed.loc[time, 'Y']
        
        dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        distances.append(dist)
        scores.append(calculate_score(dist))
    
    avg_score = np.mean(scores)
    avg_dist = np.mean(distances)
    
    # 可视化
    if show_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制预测轨迹（统一样式）
        pred_sorted = pred_df.sort_values('Time')
        ax.plot(pred_sorted['X']/1000, pred_sorted['Y']/1000,
                color='red', linewidth=2.5, linestyle='--', 
                label='预测轨迹', zorder=7, alpha=0.9, dashes=(8, 4))
        
        # 真值轨迹（确保排序，放在预测之后画）
        gt_xy_sorted = gt_xy_df.sort_values('Time')
        ax.plot(gt_xy_sorted['X']/1000, gt_xy_sorted['Y']/1000, 
                'b-', linewidth=3.5, label='真值轨迹', zorder=8, alpha=1.0)
        
        # 2D雷达点密度热力图（在真值轨迹附近）
        if len(radar_xy_list) > 0:
            # 筛选真值轨迹附近的雷达点（100km范围内）
            gt_x_min, gt_x_max = gt_xy_df['X'].min(), gt_xy_df['X'].max()
            gt_y_min, gt_y_max = gt_xy_df['Y'].min(), gt_xy_df['Y'].max()
            
            nearby_radar = [
                p for p in radar_xy_list
                if (gt_x_min - 100000 < p['x'] < gt_x_max + 100000) and
                   (gt_y_min - 100000 < p['y'] < gt_y_max + 100000)
            ]
            
            if len(nearby_radar) > 0:
                radar_xs = [p['x'] for p in nearby_radar[:500]]  # 最多500个点
                radar_ys = [p['y'] for p in nearby_radar[:500]]
                ax.scatter(np.array(radar_xs)/1000, np.array(radar_ys)/1000,
                          c='gray', s=5, alpha=0.3, label=f'2D雷达点(n={len(nearby_radar)})',
                          zorder=1)
        
        # 单ESM方向射线
        if len(single_esm_list) > 0:
            for i, esm in enumerate(single_esm_list[:5]):  # 只画前5条
                line_length = 200000  # 200km
                s_xy = esm['sensor_xy']
                theta = esm['azimuth']
                
                end_x = s_xy[0] + line_length * np.sin(np.radians(theta))
                end_y = s_xy[1] + line_length * np.cos(np.radians(theta))
                start_x = s_xy[0] - line_length * np.sin(np.radians(theta))
                start_y = s_xy[1] - line_length * np.cos(np.radians(theta))
                
                if i == 0:
                    ax.plot([start_x/1000, end_x/1000], 
                           [start_y/1000, end_y/1000],
                           color='cyan', alpha=0.4, linewidth=1.5, 
                           label=f'单ESM射线(n={len(single_esm_list)})', 
                           zorder=2, linestyle=':')
                else:
                    ax.plot([start_x/1000, end_x/1000], 
                           [start_y/1000, end_y/1000],
                           color='cyan', alpha=0.4, linewidth=1.5, 
                           zorder=2, linestyle=':')
        
        # 交叉点（改用黄色，避免和红色预测混淆）
        if len(crossing_points) > 0:
            cross_xs = [p['x'] for p in crossing_points]
            cross_ys = [p['y'] for p in crossing_points]
            ax.scatter(np.array(cross_xs)/1000, np.array(cross_ys)/1000,
                       c='gold', s=250, marker='*', label=f'交叉点(n={len(crossing_points)})', 
                       zorder=12, edgecolors='orange', linewidths=2.5)
        
        # 真值起点和终点
        ax.scatter(gt_xy_df.iloc[0]['X']/1000, gt_xy_df.iloc[0]['Y']/1000,
                   c='blue', s=180, marker='o', label='真值起点', zorder=10)
        ax.scatter(gt_xy_df.iloc[-1]['X']/1000, gt_xy_df.iloc[-1]['Y']/1000,
                   c='purple', s=180, marker='s', label='真值终点', zorder=10)
        
        # 设置显示范围
        margin = 50
        x_min = min(gt_xy_df['X'].min(), pred_df['X'].min())/1000 - margin
        x_max = max(gt_xy_df['X'].max(), pred_df['X'].max())/1000 + margin
        y_min = min(gt_xy_df['Y'].min(), pred_df['Y'].min())/1000 - margin
        y_max = max(gt_xy_df['Y'].max(), pred_df['Y'].max())/1000 + margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.legend(loc='best', fontsize=11)
        ax.set_xlabel('X坐标 (km)', fontsize=12)
        ax.set_ylabel('Y坐标 (km)', fontsize=12)
        ax.set_title(f'船舶 {boat_id} ({boat_type}类型)\n'
                    f'参数: 时长={params["extrap_minutes"]}min, '
                    f'速度={params["default_speed"]}m/s, 航向={params["default_bearing"]}°\n'
                    f'得分: {avg_score:.2f} | 平均误差: {avg_dist:.1f}m',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        output_file = f'validation_viz_{boat_id}_{boat_type}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        print(f"  图片已保存: {output_file}")
        plt.close()
    
    return {
        'boat_id': boat_id,
        'boat_type': boat_type,
        'score': avg_score,
        'avg_distance': avg_dist,
        'num_points': len(common_times)
    }


def evaluate_all_boats():
    """评估所有船只并统计得分"""
    print("=" * 70)
    print("评估所有船只")
    print("=" * 70)
    
    # 读取validation数据
    radar_df = pd.read_csv('validation_set/radar_detection.csv')
    radar_df['Time'] = pd.to_datetime(radar_df['Time'])
    
    esm_data = radar_df[radar_df['Distance'] == -100].copy()
    
    sensor_xy_dict = {}
    for sid in ESM_SENSORS:
        info = SENSOR_INFO[sid]
        sensor_xy_dict[sid] = lonlat_to_xy(info['lon'], info['lat'])
    
    # 测试参数
    params = {
        'extrap_minutes': 450,
        'default_speed': 6.3,
        'default_bearing': 175
    }
    
    all_results = []
    single_results = []
    multi_results = []
    
    boat_ids = esm_data['BoatID'].unique()
    print(f"\n总船舶数: {len(boat_ids)} 艘")
    print("\n开始评估...")
    
    for idx, boat_id in enumerate(boat_ids):
        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx+1}/{len(boat_ids)}")
        
        # 生成预测轨迹
        pred_df, boat_type = generate_trajectory(boat_id, radar_df, esm_data, sensor_xy_dict, params)
        
        if pred_df is None:
            continue
        
        # 检查ground truth是否存在
        gt_file = f'validation_set/ground_truth/{boat_id}.csv'
        if not Path(gt_file).exists():
            continue
        
        gt_df = pd.read_csv(gt_file)
        gt_df['Time'] = pd.to_datetime(gt_df['Time'])
        
        # 转换真值到XY
        gt_xy = []
        for _, row in gt_df.iterrows():
            x, y = lonlat_to_xy(row['LON'], row['LAT'])
            gt_xy.append({'Time': row['Time'], 'X': x, 'Y': y})
        gt_xy_df = pd.DataFrame(gt_xy)
        
        # 计算得分
        pred_df = pred_df.sort_values('Time').reset_index(drop=True)
        pred_df = pred_df.drop_duplicates(subset=['Time'], keep='first')
        
        pred_df_indexed = pred_df.set_index('Time')
        gt_xy_indexed = gt_xy_df.set_index('Time')
        
        common_times = pred_df_indexed.index.intersection(gt_xy_indexed.index)
        
        if len(common_times) == 0:
            continue
        
        distances = []
        for time in common_times:
            pred_x = pred_df_indexed.loc[time, 'X']
            pred_y = pred_df_indexed.loc[time, 'Y']
            gt_x = gt_xy_indexed.loc[time, 'X']
            gt_y = gt_xy_indexed.loc[time, 'Y']
            
            dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            distances.append(dist)
        
        avg_dist = np.mean(distances)
        avg_score = calculate_score(avg_dist)
        
        result = {
            'boat_id': boat_id,
            'boat_type': boat_type,
            'score': avg_score,
            'avg_distance': avg_dist,
            'num_points': len(common_times)
        }
        
        all_results.append(result)
        
        if boat_type == 'single':
            single_results.append(result)
        else:
            multi_results.append(result)
    
    # 统计结果
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    
    print(f"\n总评估船只: {len(all_results)} 艘")
    print(f"  单交叉点船: {len(single_results)} 艘")
    print(f"  多交叉点船: {len(multi_results)} 艘")
    
    if len(single_results) > 0:
        single_scores = [r['score'] for r in single_results]
        single_dists = [r['avg_distance'] for r in single_results]
        print(f"\n【单交叉点船统计】")
        print(f"  得分均值: {np.mean(single_scores):.2f}")
        print(f"  得分中位数: {np.median(single_scores):.2f}")
        print(f"  得分范围: {np.min(single_scores):.2f} ~ {np.max(single_scores):.2f}")
        print(f"  平均误差: {np.mean(single_dists):.1f} m")
        print(f"  误差范围: {np.min(single_dists):.1f} ~ {np.max(single_dists):.1f} m")
    
    if len(multi_results) > 0:
        multi_scores = [r['score'] for r in multi_results]
        multi_dists = [r['avg_distance'] for r in multi_results]
        print(f"\n【多交叉点船统计】")
        print(f"  得分均值: {np.mean(multi_scores):.2f}")
        print(f"  得分中位数: {np.median(multi_scores):.2f}")
        print(f"  得分范围: {np.min(multi_scores):.2f} ~ {np.max(multi_scores):.2f}")
        print(f"  平均误差: {np.mean(multi_dists):.1f} m")
        print(f"  误差范围: {np.min(multi_dists):.1f} ~ {np.max(multi_dists):.1f} m")
    
    if len(all_results) > 0:
        all_scores = [r['score'] for r in all_results]
        print(f"\n【总体统计】")
        print(f"  总体均分: {np.mean(all_scores):.2f}")
        print(f"  总体中位数: {np.median(all_scores):.2f}")
    
    return all_results, single_results, multi_results


def visualize_different_types():
    """可视化不同类型船的效果"""
    print("=" * 70)
    print("可视化不同类型船的预测效果")
    print("=" * 70)
    
    # 读取validation数据
    radar_df = pd.read_csv('validation_set/radar_detection.csv')
    radar_df['Time'] = pd.to_datetime(radar_df['Time'])
    
    esm_data = radar_df[radar_df['Distance'] == -100].copy()
    
    sensor_xy_dict = {}
    for sid in ESM_SENSORS:
        info = SENSOR_INFO[sid]
        sensor_xy_dict[sid] = lonlat_to_xy(info['lon'], info['lat'])
    
    # 找单交叉点和多交叉点船
    single_boats = []
    multi_boats = []
    
    for boat_id in esm_data['BoatID'].unique():
        boat_esm = esm_data[esm_data['BoatID'] == boat_id]
        crossing_count = 0
        
        for time, group in boat_esm.groupby('Time'):
            sensors = group['SensorID'].values
            valid_sensors = [s for s in sensors if s in ESM_SENSORS]
            if len(valid_sensors) >= 2:
                crossing_count += 1
        
        if crossing_count == 1:
            single_boats.append(boat_id)
        elif crossing_count >= 2:
            multi_boats.append(boat_id)
    
    print(f"\n找到:")
    print(f"  单交叉点船: {len(single_boats)} 艘")
    print(f"  多交叉点船: {len(multi_boats)} 艘")
    
    # 测试参数
    params = {
        'extrap_minutes': 450,
        'default_speed': 6.3,
        'default_bearing': 175
    }
    
    # 可视化单交叉点船（随机选1个）
    if len(single_boats) > 0:
        boat_id = random.choice(single_boats)
        print(f"\n随机选择单交叉点船: {boat_id}")
        evaluate_and_visualize(boat_id, params, show_plot=True)
    
    # 可视化多交叉点船（随机选1个）
    if len(multi_boats) > 0:
        boat_id = random.choice(multi_boats)
        print(f"\n随机选择多交叉点船: {boat_id}")
        evaluate_and_visualize(boat_id, params, show_plot=True)


if __name__ == '__main__':
    # 评估所有船只
    all_results, single_results, multi_results = evaluate_all_boats()
    
    # 可视化示例船只
    print("\n" + "=" * 70)
    print("可视化示例船只")
    print("=" * 70)
    
    # 读取validation数据
    radar_df = pd.read_csv('validation_set/radar_detection.csv')
    radar_df['Time'] = pd.to_datetime(radar_df['Time'])
    
    params = {
        'extrap_minutes': 450,
        'default_speed': 6.3,
        'default_bearing': 175
    }
    
    # 随机选择并可视化单交叉点船
    if len(single_results) > 0:
        boat_id = random.choice([r['boat_id'] for r in single_results])
        print(f"\n随机选择单交叉点船: {boat_id}")
        evaluate_and_visualize(boat_id, params, show_plot=True)
    
    # 随机选择并可视化多交叉点船
    if len(multi_results) > 0:
        boat_id = random.choice([r['boat_id'] for r in multi_results])
        print(f"\n随机选择多交叉点船: {boat_id}")
        evaluate_and_visualize(boat_id, params, show_plot=True)

