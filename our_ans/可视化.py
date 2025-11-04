"""
训练集可视化脚本
包含：
1. 真值轨迹（ground truth）
2. 双ESM交叉定位点
3. 单ESM方向线
4. 雷达观测点
5. 传感器位置
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from official.CoordinateConvert import lonlat_to_xy, xy_to_lonlat, space_intersection, adjust_angle
from official.sensor_config import SENSOR_INFO, ESM_SENSORS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_boat(boat_id):
    """可视化单艘船的完整信息"""
    
    # 读取数据
    radar_file = f'official/train/radar_detection/{boat_id}.csv'
    gt_file = f'official/train/ground_truth/{boat_id}.csv'
    
    if not Path(radar_file).exists() or not Path(gt_file).exists():
        print(f"船 {boat_id} 数据不存在")
        return
    
    radar_df = pd.read_csv(radar_file)
    radar_df['Time'] = pd.to_datetime(radar_df['Time'])
    
    gt_df = pd.read_csv(gt_file)
    gt_df['Time'] = pd.to_datetime(gt_df['Time'])
    
    # 转换真值轨迹到XY坐标
    gt_xy = []
    for _, row in gt_df.iterrows():
        x, y = lonlat_to_xy(row['LON'], row['LAT'])
        gt_xy.append({'x': x, 'y': y, 'time': row['Time']})
    gt_xy_df = pd.DataFrame(gt_xy)
    
    # 准备传感器位置
    sensor_xy_dict = {}
    for sid in ESM_SENSORS:
        info = SENSOR_INFO[sid]
        sensor_xy_dict[sid] = lonlat_to_xy(info['lon'], info['lat'])
    
    # 分离ESM和雷达数据
    esm_data = radar_df[radar_df['Distance'] == -100].copy()
    radar_data = radar_df[radar_df['Distance'] != -100].copy()
    
    # 计算双ESM交叉点
    crossing_points = []
    single_esm_data = []
    
    for time, group in esm_data.groupby('Time'):
        sensors = group['SensorID'].values
        azimuths = group['Azimuth'].values
        valid_sensors = [s for s in sensors if s in ESM_SENSORS]
        
        if len(valid_sensors) >= 2:
            # 双ESM交叉定位
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
                'time': time,
                's1_id': s1_id,
                's2_id': s2_id,
                's1_xy': s1_xy,
                's2_xy': s2_xy,
                'theta1': theta1,
                'theta2': theta2
            })
        
        elif len(valid_sensors) == 1:
            # 单ESM
            s_id = valid_sensors[0]
            idx = list(sensors).index(s_id)
            s_xy = sensor_xy_dict[s_id]
            theta = adjust_angle(azimuths[idx])
            
            single_esm_data.append({
                's_id': s_id,
                's_xy': s_xy,
                'theta': theta,
                'time': time
            })
    
    # 计算雷达观测点坐标
    radar_xy = []
    for _, row in radar_data.iterrows():
        sensor_info = SENSOR_INFO.get(row['SensorID'])
        if sensor_info is None:
            continue
        
        sensor_xy = lonlat_to_xy(sensor_info['lon'], sensor_info['lat'])
        theta = adjust_angle(row['Azimuth'])
        distance = row['Distance']
        
        target_x = sensor_xy[0] + distance * np.sin(np.radians(theta))
        target_y = sensor_xy[1] + distance * np.cos(np.radians(theta))
        
        radar_xy.append({'x': target_x, 'y': target_y})
    
    # 开始绘图
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 1. 画真值轨迹（最重要）
    ax.plot(gt_xy_df['x']/1000, gt_xy_df['y']/1000, 
            'b-', linewidth=2.5, label='真值轨迹', zorder=5, alpha=0.8)
    
    # 2. 画双ESM交叉点
    if len(crossing_points) > 0:
        cross_xs = [p['x'] for p in crossing_points]
        cross_ys = [p['y'] for p in crossing_points]
        ax.scatter(np.array(cross_xs)/1000, np.array(cross_ys)/1000, 
                   c='red', s=150, marker='*', label='双ESM交叉点', 
                   zorder=10, edgecolors='darkred', linewidths=1.5)
        
        # 画交叉定位的方位线（前3个点，无限延伸）
        for i, cp in enumerate(crossing_points[:3]):
            # 传感器1的方位线（无限延伸）
            line_length = 1000000  # 1000km（足够长）
            end_x1 = cp['s1_xy'][0] + line_length * np.sin(np.radians(cp['theta1']))
            end_y1 = cp['s1_xy'][1] + line_length * np.cos(np.radians(cp['theta1']))
            start_x1 = cp['s1_xy'][0] - line_length * np.sin(np.radians(cp['theta1']))
            start_y1 = cp['s1_xy'][1] - line_length * np.cos(np.radians(cp['theta1']))
            ax.plot([start_x1/1000, end_x1/1000], 
                   [start_y1/1000, end_y1/1000],
                   'r--', alpha=0.4, linewidth=1, zorder=3)
            
            # 传感器2的方位线（无限延伸）
            end_x2 = cp['s2_xy'][0] + line_length * np.sin(np.radians(cp['theta2']))
            end_y2 = cp['s2_xy'][1] + line_length * np.cos(np.radians(cp['theta2']))
            start_x2 = cp['s2_xy'][0] - line_length * np.sin(np.radians(cp['theta2']))
            start_y2 = cp['s2_xy'][1] - line_length * np.cos(np.radians(cp['theta2']))
            ax.plot([start_x2/1000, end_x2/1000], 
                   [start_y2/1000, end_y2/1000],
                   'r--', alpha=0.4, linewidth=1, zorder=3)
    
    # 3. 画单ESM方向线（前5个，无限延伸）
    if len(single_esm_data) > 0:
        for i, esm in enumerate(single_esm_data[:5]):
            line_length = 1000000  # 1000km（无限延伸）
            end_x = esm['s_xy'][0] + line_length * np.sin(np.radians(esm['theta']))
            end_y = esm['s_xy'][1] + line_length * np.cos(np.radians(esm['theta']))
            start_x = esm['s_xy'][0] - line_length * np.sin(np.radians(esm['theta']))
            start_y = esm['s_xy'][1] - line_length * np.cos(np.radians(esm['theta']))
            
            if i == 0:
                ax.plot([start_x/1000, end_x/1000], 
                       [start_y/1000, end_y/1000],
                       'orange', alpha=0.5, linewidth=1.5, 
                       label='单ESM方向线', zorder=2, linestyle=':')
            else:
                ax.plot([start_x/1000, end_x/1000], 
                       [start_y/1000, end_y/1000],
                       'orange', alpha=0.5, linewidth=1.5, zorder=2, linestyle=':')
    
    # 4. 画雷达观测点
    if len(radar_xy) > 0:
        radar_xs = [p['x'] for p in radar_xy]
        radar_ys = [p['y'] for p in radar_xy]
        ax.scatter(np.array(radar_xs)/1000, np.array(radar_ys)/1000,
                   c='yellow', s=30, marker='o', label='雷达观测点',
                   zorder=4, alpha=0.6, edgecolors='orange', linewidths=0.5)
    
    # 5. 画传感器位置
    sensor_names = []
    sensor_xs = []
    sensor_ys = []
    
    for sid in ESM_SENSORS:
        info = SENSOR_INFO[sid]
        xy = lonlat_to_xy(info['lon'], info['lat'])
        sensor_names.append(f'传感器{sid}')
        sensor_xs.append(xy[0])
        sensor_ys.append(xy[1])
    
    ax.scatter(np.array(sensor_xs)/1000, np.array(sensor_ys)/1000,
               c='green', s=200, marker='^', label='ESM传感器',
               zorder=15, edgecolors='darkgreen', linewidths=2)
    
    # 标注传感器ID
    for i, (x, y, name) in enumerate(zip(sensor_xs, sensor_ys, sensor_names)):
        ax.annotate(name, (x/1000, y/1000), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='darkgreen')
    
    # 6. 画真值轨迹的起点和终点
    ax.scatter(gt_xy_df.iloc[0]['x']/1000, gt_xy_df.iloc[0]['y']/1000,
               c='blue', s=200, marker='o', label='起点', zorder=12,
               edgecolors='darkblue', linewidths=2)
    ax.scatter(gt_xy_df.iloc[-1]['x']/1000, gt_xy_df.iloc[-1]['y']/1000,
               c='purple', s=200, marker='s', label='终点', zorder=12,
               edgecolors='darkviolet', linewidths=2)
    
    # 设置显示范围：聚焦在真值轨迹附近
    margin = 50  # 边距50km
    x_min = gt_xy_df['x'].min()/1000 - margin
    x_max = gt_xy_df['x'].max()/1000 + margin
    y_min = gt_xy_df['y'].min()/1000 - margin
    y_max = gt_xy_df['y'].max()/1000 + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 设置图例和标题
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_xlabel('X坐标 (km)', fontsize=12)
    ax.set_ylabel('Y坐标 (km)', fontsize=12)
    ax.set_title(f'船舶 {boat_id} 训练数据可视化\n'
                f'真值点数: {len(gt_df)} | 交叉点: {len(crossing_points)} | '
                f'单ESM: {len(single_esm_data)} | 雷达点: {len(radar_xy)}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # 保存图片
    output_file = f'train_visualization_{boat_id}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图片已保存: {output_file}")
    plt.show()
    
    # 打印统计信息
    print(f"\n船舶 {boat_id} 数据统计:")
    print(f"  真值轨迹点数: {len(gt_df)}")
    print(f"  双ESM交叉点: {len(crossing_points)}")
    print(f"  单ESM观测: {len(single_esm_data)}")
    print(f"  雷达观测点: {len(radar_xy)}")
    print(f"  时间跨度: {gt_df['Time'].min()} ~ {gt_df['Time'].max()}")


def main(boat_id=None):
    """
    主函数：可视化指定船舶
    
    参数:
        boat_id: 船舶ID，如果为None则随机选择
    """
    print("=" * 70)
    print("训练集可视化")
    print("=" * 70)
    
    if boat_id is None:
        # 扫描所有训练船舶
        gt_files = list(Path('official/train/ground_truth').glob('*.csv'))
        
        if len(gt_files) == 0:
            print("未找到训练数据")
            return
        
        # 随机选择一艘船
        random_file = random.choice(gt_files)
        boat_id = int(random_file.stem)
        print(f"\n随机选择船舶: {boat_id}")
    else:
        print(f"\n指定船舶: {boat_id}")
    
    print("正在生成可视化...\n")
    
    visualize_boat(boat_id)


if __name__ == '__main__':
    main(2553)

