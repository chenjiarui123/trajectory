"""
17.py删除踩石头版本（对照实验）
目的：验证踩石头是否真的起作用

删除内容：
- 所有踩石头相关函数
- 踩石头调用

保留内容：
- 雷达点坐标转换
- 单交叉点船处理
- 外推逻辑
- 其他所有和14.py不同的地方
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from official.CoordinateConvert import lonlat_to_xy, xy_to_lonlat, space_intersection, adjust_angle
from official.sensor_config import SENSOR_INFO, ESM_SENSORS

print("17.py删除踩石头版本（对照实验）")

# ===== 主程序 =====
print("\n正在读取数据...")
radar_df = pd.read_csv('official/test/radar_detection.csv')
radar_df['Time'] = pd.to_datetime(radar_df['Time'])

# 分离ESM数据
esm_data = radar_df[radar_df['Distance'] == -100].copy()
boat_ids = esm_data['BoatID'].unique()
print(f"总船舶数: {len(boat_ids)} 艘")

# 准备传感器位置的平面坐标
sensor_xy_dict = {}
for sid in ESM_SENSORS:
    info = SENSOR_INFO[sid]
    sensor_xy_dict[sid] = lonlat_to_xy(info['lon'], info['lat'])

# 结果列表
results = []

# 统计
stats = {'多交叉点': 0, '单交叉点': 0, '无交叉点': 0}

# 处理每艘船
print("\n开始生成轨迹...")
for idx, boat_id in enumerate(boat_ids):
    if (idx + 1) % 50 == 0:
        print(f"  进度: {idx+1}/{len(boat_ids)}")
    
    # 筛选该船的ESM数据
    boat_esm = esm_data[esm_data['BoatID'] == boat_id].copy()
    
    if len(boat_esm) == 0:
        continue
    
    # 按时间分组，找双ESM交叉点
    grouped = boat_esm.groupby('Time')
    
    intersection_points = []
    
    for time, group in grouped:
        sensors = group['SensorID'].values
        azimuths = group['Azimuth'].values
        
        valid_sensors = [s for s in sensors if s in ESM_SENSORS]
        
        if len(valid_sensors) >= 2:
            sensor1_id = valid_sensors[0]
            sensor2_id = valid_sensors[1]
            idx1 = list(sensors).index(sensor1_id)
            idx2 = list(sensors).index(sensor2_id)
            
            sensor1_xy = sensor_xy_dict[sensor1_id]
            sensor2_xy = sensor_xy_dict[sensor2_id]
            
            theta1 = adjust_angle(azimuths[idx1])
            theta2 = adjust_angle(azimuths[idx2])
            
            target_xy = space_intersection(sensor1_xy, sensor2_xy, theta1, theta2)
            
            intersection_points.append({
                'time': time,
                'x': target_xy[0],
                'y': target_xy[1]
            })
    
    if len(intersection_points) == 0:
        stats['无交叉点'] += 1
        continue
    
    if len(intersection_points) == 1:
        stats['单交叉点'] += 1
        # 单交叉点船：简单外推（和14.py/17.py一样）
        point = intersection_points[0]
        anchor_time = point['time']
        
        # 外推200分钟
        extrap_start = anchor_time - pd.Timedelta(minutes=200)
        extrap_end = anchor_time + pd.Timedelta(minutes=200)
        time_range = pd.date_range(start=extrap_start, end=extrap_end, freq='1min')
        
        # 使用默认速度和方向
        direction = np.array([1.0, 1.0]) / np.sqrt(2)
        default_speed = 7.0
        
        for time in time_range:
            time_diff = (time - anchor_time).total_seconds()
            extrap_x = point['x'] + direction[0] * default_speed * time_diff
            extrap_y = point['y'] + direction[1] * default_speed * time_diff
            lon, lat = xy_to_lonlat(extrap_x, extrap_y)
            
            results.append({
                'ID': boat_id,
                'Time': time,
                'LON': round(float(lon), 6),
                'LAT': round(float(lat), 6)
            })
        
        continue
    
    # ===== 多交叉点船 =====
    stats['多交叉点'] += 1
    
    # 按时间排序
    intersection_points = sorted(intersection_points, key=lambda p: p['time'])
    
    # ===== 删除踩石头，直接线性插值 =====
    # 提取时间和坐标
    times = [p['time'] for p in intersection_points]
    xs = [p['x'] for p in intersection_points]
    ys = [p['y'] for p in intersection_points]
    
    start_time = times[0]
    end_time = times[-1]
    
    # 转换为相对秒数
    time_numeric = [(t - start_time).total_seconds() for t in times]
    
    # 线性插值
    x_interp = interp1d(time_numeric, xs, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
    y_interp = interp1d(time_numeric, ys, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    
    # 生成交叉点之间的轨迹
    time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    for time in time_range:
        time_sec = (time - start_time).total_seconds()
        x = float(x_interp(time_sec))
        y = float(y_interp(time_sec))
        lon, lat = xy_to_lonlat(x, y)
        
        results.append({
            'ID': boat_id,
            'Time': time,
            'LON': round(float(lon), 6),
            'LAT': round(float(lat), 6)
        })
    
    # ===== 外推部分（保持17.py的逻辑）=====
    if len(intersection_points) >= 2:
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
        extrap_start = first_point['time'] - pd.Timedelta(minutes=200)
        time_range_before = pd.date_range(start=extrap_start, 
                                          end=first_point['time'] - pd.Timedelta(minutes=1),
                                          freq='1min')
        
        for time in time_range_before:
            time_diff = (time - first_point['time']).total_seconds()
            extrap_x = first_point['x'] + velocity_start[0] * time_diff
            extrap_y = first_point['y'] + velocity_start[1] * time_diff
            lon, lat = xy_to_lonlat(extrap_x, extrap_y)
            
            results.append({
                'ID': boat_id,
                'Time': time,
                'LON': round(float(lon), 6),
                'LAT': round(float(lat), 6)
            })
        
        # 后向外推
        extrap_end = last_point['time'] + pd.Timedelta(minutes=200)
        time_range_after = pd.date_range(start=last_point['time'] + pd.Timedelta(minutes=1),
                                         end=extrap_end,
                                         freq='1min')
        
        for time in time_range_after:
            time_diff = (time - last_point['time']).total_seconds()
            extrap_x = last_point['x'] + velocity_end[0] * time_diff
            extrap_y = last_point['y'] + velocity_end[1] * time_diff
            lon, lat = xy_to_lonlat(extrap_x, extrap_y)
            
            results.append({
                'ID': boat_id,
                'Time': time,
                'LON': round(float(lon), 6),
                'LAT': round(float(lat), 6)
            })

print(f"\n总共生成 {len(results)} 条轨迹点")

# 转换为DataFrame并保存
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['ID', 'Time'])

# 统计信息
unique_boats = results_df['ID'].nunique()
print(f"\n成功处理的船舶数量: {unique_boats} 艘")
print(f"  多交叉点(≥2): {stats['多交叉点']} 艘")
print(f"  单交叉点(=1): {stats['单交叉点']} 艘")
print(f"  无交叉点(=0): {stats['无交叉点']} 艘")
avg_points = len(results_df) / unique_boats if unique_boats > 0 else 0
print(f"平均每艘船的轨迹点数: {avg_points:.1f} 个")

# 格式化时间列
results_df['Time'] = pd.to_datetime(results_df['Time']).dt.strftime("%Y-%m-%d %H:%M:%S")

# 保存结果
import csv
output_file = 'submission_17_no_stepping.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Time', 'LON', 'LAT'])
    for _, row in results_df.iterrows():
        writer.writerow([row['ID'], row['Time'], row['LON'], row['LAT']])

print(f"\n结果已保存到: {output_file}")
print("\n前10行结果示例:")
print(results_df.head(10).to_string(index=False))

