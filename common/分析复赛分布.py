"""
分析复赛数据的ESM分布情况
用于指导验证集构建
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_final_data():
    """分析复赛数据"""
    print("=" * 80)
    print("复赛数据深度分析")
    print("=" * 80)

    # 读取复赛数据
    final_file = Path('official/radar_detection.csv')
    df = pd.read_csv(final_file)
    df['Time'] = pd.to_datetime(df['Time'])

    # 基本信息
    print(f"\n【基本信息】")
    print(f"  总记录数: {len(df):,}")
    time_min = df['Time'].min()
    time_max = df['Time'].max()
    time_duration = (time_max - time_min).total_seconds() / 3600  # 小时
    print(f"  时间范围: {time_min} ~ {time_max}")
    print(f"  时间跨度: {time_duration:.1f} 小时")

    # 数据类型
    esm_data = df[df['Distance'] == -100]
    radar_data = df[df['Distance'] != -100]

    print(f"\n【数据类型分布】")
    print(f"  ESM记录: {len(esm_data):,} ({len(esm_data)/len(df)*100:.1f}%)")
    print(f"  雷达记录: {len(radar_data):,} ({len(radar_data)/len(df)*100:.1f}%)")
    
    print(f"\n【传感器分布】")
    print(f"  ESM传感器 (1, 5, 10):")
    for sensor_id in [1, 5, 10]:
        count = len(esm_data[esm_data['SensorID'] == sensor_id])
        boats = esm_data[esm_data['SensorID'] == sensor_id]['BoatID'].nunique()
        print(f"    传感器{sensor_id:2d}: {count:5d} 条 ({count/len(esm_data)*100:.1f}%), 覆盖 {boats} 艘船")

    print(f"\n  2D雷达传感器 (2,3,4,6,7,8,9):")
    for sensor_id in [2, 3, 4, 6, 7, 8, 9]:
        count = len(radar_data[radar_data['SensorID'] == sensor_id])
        print(f"    传感器{sensor_id:2d}: {count:5d} 条 ({count/len(radar_data)*100:.1f}%)")
    
    # 船舶统计
    boat_ids = esm_data['BoatID'].unique()
    print(f"\n【船舶统计】")
    print(f"  船舶数量: {len(boat_ids)}")

    # 每艘船的ESM点数
    esm_per_boat = esm_data.groupby('BoatID').size()
    print(f"\n  每艘船的ESM观测点数:")
    print(f"    平均: {esm_per_boat.mean():.1f}")
    print(f"    中位数: {esm_per_boat.median():.0f}")
    print(f"    最大: {esm_per_boat.max()}")
    print(f"    最小: {esm_per_boat.min()}")

    # 每艘船的观测时长
    print(f"\n  每艘船的观测时长:")
    durations = []
    for boat_id in boat_ids:
        boat_data = esm_data[esm_data['BoatID'] == boat_id]
        duration = (boat_data['Time'].max() - boat_data['Time'].min()).total_seconds() / 60  # 分钟
        durations.append(duration)

    durations = np.array(durations)
    print(f"    平均: {durations.mean():.1f} 分钟 ({durations.mean()/60:.1f} 小时)")
    print(f"    中位数: {np.median(durations):.0f} 分钟 ({np.median(durations)/60:.1f} 小时)")
    print(f"    最大: {durations.max():.0f} 分钟 ({durations.max()/60:.1f} 小时)")
    print(f"    最小: {durations.min():.0f} 分钟 ({durations.min()/60:.1f} 小时)")

    # 每艘船的雷达点数
    radar_per_boat = []
    for boat_id in boat_ids:
        # 注意：雷达数据中BoatID=-1，无法直接统计
        # 这里只能统计ESM能确定的船
        pass

    print(f"\n  观测频率分析:")
    intervals = []
    for boat_id in boat_ids:
        boat_data = esm_data[esm_data['BoatID'] == boat_id].sort_values('Time')
        if len(boat_data) > 1:
            time_diffs = boat_data['Time'].diff().dt.total_seconds() / 60  # 分钟
            intervals.extend(time_diffs.dropna().tolist())

    intervals = np.array(intervals)
    print(f"    平均观测间隔: {intervals.mean():.1f} 分钟")
    print(f"    中位数观测间隔: {np.median(intervals):.0f} 分钟")
    print(f"    最大观测间隔: {intervals.max():.0f} 分钟")
    print(f"    最小观测间隔: {intervals.min():.0f} 分钟")
    
    # 交叉点分析
    print(f"\n【ESM交叉点分析】")
    crossing_stats = []

    for boat_id in boat_ids:
        boat_esm = esm_data[esm_data['BoatID'] == boat_id]

        # 统计每个时刻有多少个ESM观测
        crossing_count = 0
        triple_count = 0  # 三个ESM同时观测
        for _, group in boat_esm.groupby('Time'):
            if len(group) >= 3:
                triple_count += 1
                crossing_count += 1
            elif len(group) >= 2:
                crossing_count += 1

        crossing_stats.append({
            'boat_id': boat_id,
            'total_esm': len(boat_esm),
            'crossing_points': crossing_count,
            'triple_points': triple_count
        })

    stats_df = pd.DataFrame(crossing_stats)

    print(f"  有交叉点的船舶: {len(stats_df[stats_df['crossing_points'] > 0])} / {len(boat_ids)}")
    print(f"\n  双ESM交叉点数量:")
    print(f"    平均: {stats_df['crossing_points'].mean():.2f}")
    print(f"    中位数: {stats_df['crossing_points'].median():.0f}")
    print(f"    最大: {stats_df['crossing_points'].max()}")
    print(f"    最小: {stats_df['crossing_points'].min()}")

    print(f"\n  三ESM交叉点数量:")
    print(f"    平均: {stats_df['triple_points'].mean():.2f}")
    print(f"    中位数: {stats_df['triple_points'].median():.0f}")
    print(f"    最大: {stats_df['triple_points'].max()}")
    print(f"    有三ESM交叉点的船舶: {len(stats_df[stats_df['triple_points'] > 0])} / {len(boat_ids)}")
    
    # 分层统计
    stats_df['quality'] = stats_df['crossing_points'].apply(
        lambda x: 'low' if x < 2 else ('medium' if x < 5 else 'high')
    )

    print(f"\n  质量分层:")
    for quality in ['low', 'medium', 'high']:
        count = len(stats_df[stats_df['quality'] == quality])
        pct = count / len(stats_df) * 100
        print(f"    {quality:8s} (交叉点 {'<2' if quality=='low' else '2-4' if quality=='medium' else '>=5'}): {count:3d} 艘 ({pct:5.1f}%)")

    # 5号ESM的覆盖情况
    print(f"\n【5号ESM覆盖分析】（关键！）")
    esm5_data = esm_data[esm_data['SensorID'] == 5]
    boats_with_esm5 = esm5_data['BoatID'].unique()
    print(f"  5号ESM观测的船舶: {len(boats_with_esm5)} / {len(boat_ids)} ({len(boats_with_esm5)/len(boat_ids)*100:.1f}%)")
    print(f"  5号ESM总观测点: {len(esm5_data):,}")

    # 5号ESM每艘船的观测点数
    esm5_per_boat = esm5_data.groupby('BoatID').size()
    print(f"\n  5号ESM每艘船的观测点数:")
    print(f"    平均: {esm5_per_boat.mean():.1f}")
    print(f"    中位数: {esm5_per_boat.median():.0f}")
    print(f"    最大: {esm5_per_boat.max()}")
    print(f"    最小: {esm5_per_boat.min()}")

    # 5号ESM与其他传感器的同时观测
    print(f"\n  5号ESM与其他ESM同时观测分析:")
    concurrent_count = 0
    concurrent_with_1 = 0
    concurrent_with_10 = 0

    for boat_id in boats_with_esm5:
        boat_esm = esm_data[esm_data['BoatID'] == boat_id]
        for _, group in boat_esm.groupby('Time'):
            if 5 in group['SensorID'].values and len(group) >= 2:
                concurrent_count += 1
                if 1 in group['SensorID'].values:
                    concurrent_with_1 += 1
                if 10 in group['SensorID'].values:
                    concurrent_with_10 += 1

    print(f"    5号ESM与其他ESM同时观测次数: {concurrent_count:,}")
    print(f"    占5号ESM总观测的比例: {concurrent_count/len(esm5_data)*100:.1f}%")
    print(f"    其中与1号ESM同时: {concurrent_with_1:,} ({concurrent_with_1/concurrent_count*100:.1f}%)")
    print(f"    其中与10号ESM同时: {concurrent_with_10:,} ({concurrent_with_10/concurrent_count*100:.1f}%)")

    return stats_df


if __name__ == '__main__':
    # 分析复赛数据
    final_stats = analyze_final_data()

