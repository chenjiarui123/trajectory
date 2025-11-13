"""
消融实验包装模块
对 20_ans 的关键函数进行参数化包装，用于消融实验
不修改原始代码，保持独立性
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, MultiPoint
import sys

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'our_ans' / '20_ans'))

from CoordinateConvert import lonlat_to_xy, xy_to_lonlat, space_intersection, adjust_angle, radar_conversion
from preprocess import (
    load_sensor_coords, get_rader_points_xy,
    generate_ray, angle_between_rays_numpy
)


def get_ESM_points_xy_custom(ESM_points, sensors, output_dir, check_distance=200000):
    """ESM交叉定位 - 参数化版本"""
    located_ESM_points = []
    unlocated_ESM_points = []

    grouped_ESM_points = ESM_points.groupby(['Time', 'BoatID'])
    for group, data in grouped_ESM_points:
        if data.shape[0] < 2:
            unlocated_ESM_points.append(data)
        else:
            sensor_id1 = int(data.iloc[0]['SensorID'])
            sensor_id2 = int(data.iloc[1]['SensorID'])
            sensor_xy1 = sensors[str(sensor_id1)]['coord']
            sensor_xy2 = sensors[str(sensor_id2)]['coord']
            point_xy = space_intersection(
                sensor_xy1, sensor_xy2,
                adjust_angle(data.iloc[0]['Azimuth']),
                adjust_angle(data.iloc[1]['Azimuth'])
            )

            # 使用参数化的距离检查
            dist1 = np.sqrt((point_xy[0] - sensor_xy1[0])**2 + (point_xy[1] - sensor_xy1[1])**2)
            dist2 = np.sqrt((point_xy[0] - sensor_xy2[0])**2 + (point_xy[1] - sensor_xy2[1])**2)

            if dist1 < check_distance and dist2 < check_distance:
                located_ESM_points.append([group[0], group[1], point_xy[0], point_xy[1]])
            else:
                unlocated_ESM_points.append(data)

    unlocated_ESM_points = pd.concat(unlocated_ESM_points, ignore_index=True)
    output_file = output_dir / 'unlocated_ESM_points.csv'
    unlocated_ESM_points.to_csv(output_file, index=False)

    located_ESM_points = pd.DataFrame(located_ESM_points, columns=['Time', 'BoatID', 'X', 'Y'])
    output_file = output_dir / 'located_ESM_points.csv'
    located_ESM_points.to_csv(output_file, index=False)


def preprocess_with_params(
    radar_file,
    sensor_coords_file,
    output_dir,
    safe_distance=200,
    esm_angle_threshold=0.001,
    esm_speed_threshold=1000,
    esm_check_distance=200000,
    enable_deduplication=True,
    enable_esm_merge=True
):
    """
    参数化的预处理流程

    Args:
        radar_file: 雷达数据文件
        sensor_coords_file: 传感器配置文件
        output_dir: 输出目录
        safe_distance: 去重安全距离(米)
        esm_angle_threshold: ESM角度匹配阈值(度)
        esm_speed_threshold: ESM速度阈值(米/分钟)
        esm_check_distance: ESM交叉定位距离检查(米)
        enable_deduplication: 是否启用去重
        enable_esm_merge: 是否启用未定位ESM合并
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载传感器
    sensors = load_sensor_coords(sensor_coords_file)

    # 步骤1: 提取点坐标
    points = pd.read_csv(radar_file, encoding='utf-8')
    start_time = pd.to_datetime('2025-06-25 05:00:00')
    points['Time'] = ((pd.to_datetime(points['Time']) - start_time).dt.total_seconds() / 60).astype(int)
    points['SensorID'] = points['SensorID'].astype('Int64')

    ESM_ids = [int(id) for id in sensors if sensors[id]['type'] == 'ESM']
    ESM_points = points[points['SensorID'].isin(ESM_ids)]
    radar_points = points[~points['SensorID'].isin(ESM_ids)]

    get_rader_points_xy(radar_points, sensors, output_dir)
    get_ESM_points_xy_custom(ESM_points, sensors, output_dir, check_distance=esm_check_distance)

    # 步骤2: 去重
    radar_points_file = output_dir / 'radar_points.csv'
    located_ESM_file = output_dir / 'located_ESM_points.csv'
    deduplacate_radar_file = output_dir / 'deduplacate_radar_points.csv'

    if enable_deduplication:
        dedumplacate_custom(
            str(radar_points_file),
            str(located_ESM_file),
            str(deduplacate_radar_file),
            safe_distance=safe_distance
        )
    else:
        # 不去重，直接复制
        import shutil
        shutil.copy(radar_points_file, deduplacate_radar_file)

    # 步骤3: 合并未定位ESM
    unlocated_ESM_file = output_dir / 'unlocated_ESM_points.csv'
    radar_file_updated = output_dir / 'deduplacate_radar_points_updated.csv'
    located_ESM_updated = output_dir / 'located_ESM_points_updated.csv'

    if enable_esm_merge:
        merge_unlocated_ESM_custom(
            str(unlocated_ESM_file),
            str(deduplacate_radar_file),
            str(located_ESM_file),
            sensor_coords_file,
            str(radar_file_updated),
            str(located_ESM_updated),
            angle_threshold=esm_angle_threshold,
            speed_threshold=esm_speed_threshold
        )
    else:
        # 不合并，直接复制
        import shutil
        shutil.copy(deduplacate_radar_file, radar_file_updated)
        shutil.copy(located_ESM_file, located_ESM_updated)


def dedumplacate_custom(radar_points_file, located_ESM_points_file, deduplacate_radar_file, safe_distance=200):
    """去重函数 - 参数化版本"""
    radar_points = pd.read_csv(radar_points_file, encoding='utf-8')
    radar_points['BoatID'] = -1

    located_ESM_points = pd.read_csv(located_ESM_points_file, encoding='utf-8')

    located_points = pd.concat([radar_points, located_ESM_points], ignore_index=True)
    located_points_group = located_points.groupby('Time')

    retain_points = []
    for time, data in located_points_group:
        data = data.sort_values(by='BoatID', ascending=False)

        points = []
        for i, row in data.iterrows():
            if row['BoatID'] != -1:
                points.append(Point(row['X'], row['Y']))
            else:
                cur_point = Point(row['X'], row['Y'])
                if len(points) > 0:
                    buffer = MultiPoint(points).buffer(safe_distance)
                    if not buffer.contains(cur_point):
                        points.append(cur_point)
                        retain_points.append(row)
                else:
                    points.append(cur_point)
                    retain_points.append(row)

    retain_radar_points = pd.concat(retain_points, axis=1).transpose()
    retain_radar_points.to_csv(deduplacate_radar_file, index=False)


def merge_unlocated_ESM_custom(
    unlocated_ESM_file, radar_file, located_ESM_file, sensor_coords_file,
    radar_file_updated, located_ESM_updated,
    angle_threshold=0.001, speed_threshold=1000
):
    """合并未定位ESM - 参数化版本"""
    unlocated_ESM = pd.read_csv(unlocated_ESM_file)
    unlocated_ESM['SensorID'] = unlocated_ESM['SensorID'].astype('Int64')
    
    located_ESM_points = pd.read_csv(located_ESM_file)
    radar = pd.read_csv(radar_file)
    sensors = load_sensor_coords(sensor_coords_file)

    located = []
    for i, row in unlocated_ESM.iterrows():
        sensor_xy = sensors[str(row['SensorID'])]['coord']
        azimuth = row['Azimuth']
        time = row['Time']
        end_xy = generate_ray(sensor_xy, azimuth)
        
        senor_point = Point(sensor_xy)
        condidates = radar[radar['Time'] == time]
        
        min_dis = 200000
        min_point = None
        index = -1

        for j, candidate in condidates.iterrows():
            point = Point(candidate['X'], candidate['Y'])
            if angle_between_rays_numpy(sensor_xy[0], sensor_xy[1], point.x, point.y, end_xy[0], end_xy[1]) < angle_threshold:
                earlier = located_ESM_points[(located_ESM_points['BoatID'] == row['BoatID']) & (located_ESM_points['Time'] <= time)]
                closest_earlier = earlier.iloc[(time - earlier['Time']).abs().argsort()[:1]]
                later = located_ESM_points[(located_ESM_points['BoatID'] == row['BoatID']) & (located_ESM_points['Time'] >= time)]
                closest_later = later.iloc[(later['Time'] - time).abs().argsort()[:1]]

                if earlier.shape[0] == 0 and later.shape[0] == 0:
                    continue

                if earlier.shape[0] > 0:
                    if closest_earlier['Time'].values[0] == time:
                        continue
                    speed_earlier = (np.sqrt(
                        (closest_earlier['X'].values[0] - point.x) ** 2 +
                        (closest_earlier['Y'].values[0] - point.y) ** 2)
                        / (time - closest_earlier['Time'].values[0]))
                    if speed_earlier > speed_threshold:
                        continue

                if later.shape[0] > 0:
                    if closest_later['Time'].values[0] == time:
                        continue
                    speed_later = (np.sqrt(
                        (closest_later['X'].values[0] - point.x) ** 2 +
                        (closest_later['Y'].values[0] - point.y) ** 2)
                        / (closest_later['Time'].values[0] - time))
                    if speed_later > speed_threshold:
                        continue

                if point.distance(senor_point) < min_dis:
                    min_dis = point.distance(senor_point)
                    min_point = point
                    index = j

        if min_point is not None and index != -1:
            located.append([time, row['BoatID'], min_point.x, min_point.y])
            radar = radar.drop(index)

    new_located = pd.DataFrame(located, columns=['Time', 'BoatID', 'X', 'Y'])
    located_ESM_points = pd.concat([located_ESM_points, new_located])
    located_ESM_points.to_csv(located_ESM_updated, index=False)
    radar.to_csv(radar_file_updated, index=False)


def interpolation_with_params(
    located_ESM_file,
    radar_file,
    output_dir,
    update_distance=600,
    update_speed=800,
    update_angle=90,
    ransac_distance=500,
    ransac_min_points=5,
    iteration_rounds=3,
    enable_ransac=True
):
    """
    参数化的插值流程 - 简化版本

    Args:
        located_ESM_file: 定位的ESM点文件
        radar_file: 雷达点文件
        output_dir: 输出目录
        update_distance: 更新点距离阈值(米) [暂未实现]
        update_speed: 更新点速度阈值(米/分钟) [暂未实现]
        update_angle: 更新点角度阈值(度) [暂未实现]
        ransac_distance: RANSAC距离阈值(米) [暂未实现]
        ransac_min_points: RANSAC最小点数 [暂未实现]
        iteration_rounds: 迭代轮数 [已实现]
        enable_ransac: 是否启用RANSAC [已实现]

    注意: 由于原函数中很多参数是硬编码的，目前只实现了 iteration_rounds 和 enable_ransac 的参数化
          其他参数需要完全重写 update_unlocated 和 ransac_fit 函数才能实现
    """
    from simple_interpolation import fit_curve_seperate, fit_curve, update_unlocated, ransac_fit, point2result

    output_dir = Path(output_dir)
    startx = 0
    starty = 0

    # 读取数据
    radar_points_df = pd.read_csv(radar_file)
    points = []
    for i, row in radar_points_df.iterrows():
        points.append([row['Time'], row['X']-startx, row['Y']-starty])

    located_ESM_points = pd.read_csv(located_ESM_file)
    located_ESM_points = located_ESM_points.sort_values(by='Time', ascending=True)
    sorted_BoatIDs = located_ESM_points.groupby('BoatID').size().sort_values(ascending=False).index

    tracks = []
    for i, BoatID in enumerate(sorted_BoatIDs, 1):
        group_data = located_ESM_points[located_ESM_points['BoatID'] == BoatID].copy()
        group_data['BoatID'] = group_data['BoatID'].astype('Int64')

        if group_data.shape[0] > 1:
            # 使用参数化的迭代次数
            for j in range(iteration_rounds):
                try:
                    interp_pointx, interp_pointy, interp_pointz = fit_curve_seperate(group_data)
                except:
                    break

                point_dict = {}
                for index, point in enumerate(zip(interp_pointx, interp_pointy, interp_pointz)):
                    id = int(point[0]+0.1)
                    point_dict[id] = point[1:]

                group_data, points = update_unlocated(group_data, point_dict, points, BoatID)
                group_data = group_data.sort_values(by='Time', ascending=True)
                interp_pointx, interp_pointy, interp_pointz = fit_curve_seperate(group_data)

        elif group_data.shape[0] == 1:
            # 使用参数化的 enable_ransac
            if enable_ransac:
                group_data, points = ransac_fit(group_data, points, BoatID)
                if group_data.shape[0] == 1:
                    continue
                else:
                    interp_pointx, interp_pointy, interp_pointz = fit_curve(group_data)
            else:
                # 不启用RANSAC，跳过单点轨迹
                continue
        else:
            continue

        track = point2result(interp_pointx, interp_pointy, interp_pointz, BoatID)
        tracks.extend(track)

    # 保存结果
    outdf = pd.DataFrame(tracks, columns=['ID', 'Time', 'LON', 'LAT'])
    df_sorted = outdf.sort_values(['ID', 'Time'])
    df_sorted['Time'] = pd.to_datetime(df_sorted['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    output_file = output_dir / 'results.csv'
    df_sorted.to_csv(output_file, index=False, float_format='%.6f')

