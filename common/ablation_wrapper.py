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
# 优先使用 official/20_ans，避免在项目根目录创建 20_ans
sys.path.insert(0, str(project_root / 'official' / '20_ans'))

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


def calculate_point_confidence(point, group_data, point_dict,
                              update_distance, update_speed, update_angle):
    """
    计算雷达点属于某条轨迹的置信度分数

    Args:
        point: 雷达点 [time, x, y]
        group_data: 船只的ESM点数据
        point_dict: 插值点字典 {time: (x, y)}
        update_distance: 距离阈值
        update_speed: 速度阈值
        update_angle: 角度阈值

    Returns:
        score: 置信度分数 (0-1)，0表示不匹配
    """
    from preprocess import angle_between_rays_numpy

    id = int(point[0])

    if id not in point_dict:
        return 0.0

    # 1. 距离检查和分数
    distance = np.sqrt((point_dict[id][0] - point[1])**2 +
                      (point_dict[id][1] - point[2])**2)

    if distance >= update_distance:
        return 0.0

    distance_score = 1.0 - (distance / update_distance)

    # 2. 找前后最近的ESM点
    earlier = group_data[group_data['Time'] <= id]
    closest_earlier = earlier.iloc[(id - earlier['Time']).abs().argsort()[:1]]
    later = group_data[group_data['Time'] >= id]
    closest_later = later.iloc[(later['Time'] - id).abs().argsort()[:1]]

    if earlier.shape[0] == 0 or later.shape[0] == 0:
        return 0.0

    if closest_earlier['Time'].values[0] == id or closest_later['Time'].values[0] == id:
        return 0.0

    # 3. 速度检查和一致性分数
    speed_earlier = (np.sqrt((closest_earlier['X'].values[0] - point[1])**2 +
                            (closest_earlier['Y'].values[0] - point[2])**2) /
                    (id - closest_earlier['Time'].values[0]))

    speed_later = (np.sqrt((closest_later['X'].values[0] - point[1])**2 +
                          (closest_later['Y'].values[0] - point[2])**2) /
                  (closest_later['Time'].values[0] - id))

    if speed_earlier > update_speed or speed_later > update_speed:
        return 0.0

    # 速度一致性：两个速度越接近越好
    avg_speed = (speed_earlier + speed_later) / 2
    if avg_speed > 0:
        speed_diff = abs(speed_earlier - speed_later)
        speed_consistency = 1.0 - min(speed_diff / avg_speed, 1.0)
    else:
        speed_consistency = 0.5

    # 4. 角度检查和分数
    angle = angle_between_rays_numpy(
        point[1], point[2],
        closest_earlier['X'].values[0], closest_earlier['Y'].values[0],
        closest_later['X'].values[0], closest_later['Y'].values[0],
        degrees=True
    )

    if angle < update_angle:
        return 0.0

    # 角度越接近180°越好（说明在直线上）
    angle_score = (angle - update_angle) / (180 - update_angle)

    # 5. 综合分数（加权平均）
    final_score = (
        0.4 * distance_score +      # 距离最重要
        0.3 * speed_consistency +    # 速度一致性次之
        0.3 * angle_score            # 角度也重要
    )

    return final_score


def update_unlocated_custom(group_data, point_dict, points, BoatID,
                           update_distance=600, update_speed=800, update_angle=90,
                           enable_kinematic_constraints=False):
    """
    参数化的 update_unlocated 函数

    Args:
        group_data: 当前船只的ESM点数据
        point_dict: 插值点字典 {time: (x, y)}
        points: 候选雷达点列表
        BoatID: 船只ID
        update_distance: 雷达点与插值点的最大距离(米)
        update_speed: 与前后ESM点的最大速度(米/分钟)
        update_angle: 前后ESM点夹角的最小值(度)

    Returns:
        group_data: 更新后的数据
        points: 剩余的候选点
    """
    from preprocess import angle_between_rays_numpy

    located = []
    for point in points:
        id = int(point[0])
        if id in point_dict:
            distance = np.sqrt((point_dict[id][0] - point[1]) ** 2 + (point_dict[id][1] - point[2]) ** 2)
            if distance < update_distance:
                earlier = group_data[group_data['Time'] <= id]
                closest_earlier = earlier.iloc[(id - earlier['Time']).abs().argsort()[:1]]
                later = group_data[group_data['Time'] >= id]
                closest_later = later.iloc[(later['Time'] - id).abs().argsort()[:1]]

                if earlier.shape[0] == 0 or later.shape[0] == 0:
                    continue

                if closest_earlier['Time'].values[0] == id or closest_later['Time'].values[0] == id:
                   continue

                speed_earlier = (np.sqrt((closest_earlier['X'].values[0] - point[1]) ** 2 + (closest_earlier['Y'].values[0] - point[2]) ** 2)
                                 / (id - closest_earlier['Time'].values[0]))
                speed_later = (np.sqrt((closest_later['X'].values[0] - point[1]) ** 2 + (closest_later['Y'].values[0] - point[2]) ** 2)
                               / (closest_later['Time'].values[0] - id))

                if speed_earlier > update_speed or speed_later > update_speed:
                    continue

                if angle_between_rays_numpy(point[1], point[2], closest_earlier['X'].values[0], closest_earlier['Y'].values[0],
                                            closest_later['X'].values[0], closest_later['Y'].values[0], degrees=True) < update_angle:
                    continue

                n = group_data.shape[0]
                group_data.loc[n] = [int(point[0]), BoatID, point[1], point[2]]
                located.append(point)

    points = [point for point in points if point not in located]
    return group_data, points


def update_unlocated_with_confidence(group_data, point_dict, points, BoatID,
                                     update_distance=600, update_speed=800, update_angle=90,
                                     confidence_threshold=0.3, top_k=None):
    """
    基于置信度的雷达点匹配（改进版）

    Args:
        group_data: 当前船只的ESM点数据
        point_dict: 插值点字典 {time: (x, y)}
        points: 候选雷达点列表
        BoatID: 船只ID
        update_distance: 雷达点与插值点的最大距离(米)
        update_speed: 与前后ESM点的最大速度(米/分钟)
        update_angle: 前后ESM点夹角的最小值(度)
        confidence_threshold: 置信度阈值，低于此值的点不考虑
        top_k: 最多选择前k个高分点，None表示不限制

    Returns:
        group_data: 更新后的数据
        points: 剩余的候选点
    """
    # 计算所有候选点的置信度
    candidates = []

    for point in points:
        score = calculate_point_confidence(
            point, group_data, point_dict,
            update_distance, update_speed, update_angle
        )

        if score > confidence_threshold:
            candidates.append((point, score))

    # 按置信度排序（从高到低）
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 选择前top_k个点
    if top_k is not None:
        candidates = candidates[:top_k]

    # 添加选中的点
    located = []
    for point, score in candidates:
        n = group_data.shape[0]
        group_data.loc[n] = [int(point[0]), BoatID, point[1], point[2]]
        located.append(point)

    # 从候选池中移除已分配的点
    points = [point for point in points if point not in located]

    return group_data, points


def ransac_fit_custom(group_data, points, BoatID, ransac_distance=500, ransac_min_points=5):
    """
    参数化的 RANSAC 拟合函数

    Args:
        group_data: 当前船只的ESM点数据
        points: 候选雷达点列表
        BoatID: 船只ID
        ransac_distance: RANSAC距离阈值(米)
        ransac_min_points: RANSAC最小点数要求

    Returns:
        group_data: 更新后的数据
        points: 剩余的候选点
    """
    startx = 0
    starty = 0

    x = group_data['Time'].values[0]
    y = group_data['X'].values[0] - startx
    z = group_data['Y'].values[0] - starty

    max_point_num = 0
    max_points = None

    points_array = np.array(points)
    line_point = np.array((x, y, z))

    for point in points:
        line_direction = np.array((point[0]-x, point[1]-y, point[2]-z))

        # 单位化方向向量
        line_direction = line_direction / np.linalg.norm(line_direction)

        # 处理直线垂直于x轴的情况
        if abs(line_direction[0]) < 1e-10:
            continue

        # 计算参数t向量
        t_values = (points_array[:, 0] - line_point[0]) / line_direction[0]

        # 计算投影点
        x0, y0, z0 = line_point
        dx, dy, dz = line_direction

        n = len(t_values)
        projections = np.empty((n, 3))
        projections[:, 0] = x0 + t_values * dx
        projections[:, 1] = y0 + t_values * dy
        projections[:, 2] = z0 + t_values * dz

        # 计算距离（yz平面距离）
        yz_distances = np.sqrt(
          (points_array[:, 1] - projections[:, 1]) ** 2 +
          (points_array[:, 2] - projections[:, 2]) ** 2
        )

        mask = yz_distances < ransac_distance
        if mask.sum() > max_point_num:
            max_point_num = mask.sum()
            max_points = points_array[mask]

    if max_point_num >= ransac_min_points:
        for point in max_points:
            if int(point[0]) == x:
                continue
            n = group_data.shape[0]
            group_data.loc[n] = [int(point[0]), BoatID, point[1], point[2]]
            point_list = point.tolist()
            if point_list in points:
                points.remove(point_list)

    group_data = group_data.sort_values(by='Time', ascending=True)
    return group_data, points


def smooth_velocity(interp_pointx, interp_pointy, interp_pointz, sigma=2.0, max_speed=1000):
    """
    对轨迹进行速度平滑 (V4优化3)

    Args:
        interp_pointx: 时间序列
        interp_pointy: X坐标序列
        interp_pointz: Y坐标序列
        sigma: 高斯滤波的sigma参数
        max_speed: 最大速度阈值 (米/分钟)

    Returns:
        平滑后的轨迹
    """
    from scipy.ndimage import gaussian_filter1d

    # 计算速度
    if len(interp_pointx) < 3:
        return interp_pointx, interp_pointy, interp_pointz

    vx = np.diff(interp_pointy) / np.diff(interp_pointx)
    vy = np.diff(interp_pointz) / np.diff(interp_pointx)
    speed = np.sqrt(vx**2 + vy**2)

    # 检测异常速度
    outliers = speed > max_speed

    if np.any(outliers):
        # 使用高斯滤波平滑
        interp_pointy_smooth = gaussian_filter1d(interp_pointy, sigma=sigma)
        interp_pointz_smooth = gaussian_filter1d(interp_pointz, sigma=sigma)
        return interp_pointx, interp_pointy_smooth, interp_pointz_smooth
    else:
        # 没有异常速度，也进行轻微平滑
        interp_pointy_smooth = gaussian_filter1d(interp_pointy, sigma=sigma*0.5)
        interp_pointz_smooth = gaussian_filter1d(interp_pointz, sigma=sigma*0.5)
        return interp_pointx, interp_pointy_smooth, interp_pointz_smooth


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
    enable_ransac=True,
    interpolation_method='spline',
    # V4新增参数
    enable_ransac_for_all=False,
    enable_adaptive_iteration=False,
    enable_velocity_smoothing=False,
    max_iterations=10,
    velocity_smooth_sigma=2.0,
    # 置信度匹配参数
    enable_confidence_matching=False,
    confidence_threshold=0.3,
    confidence_top_k=None,
    # v3.2/v3.3 新增参数
    enable_kinematic_constraints=False,
    enable_outlier_detection=False
):
    """
    完全参数化的插值流程

    Args:
        located_ESM_file: 定位的ESM点文件
        radar_file: 雷达点文件
        output_dir: 输出目录
        update_distance: 更新点距离阈值(米) [已实现✓]
        update_speed: 更新点速度阈值(米/分钟) [已实现✓]
        update_angle: 更新点角度阈值(度) [已实现✓]
        ransac_distance: RANSAC距离阈值(米) [已实现✓]
        ransac_min_points: RANSAC最小点数 [已实现✓]
        iteration_rounds: 迭代轮数 [已实现✓]
        enable_ransac: 是否启用RANSAC [已实现✓]
        interpolation_method: 插值方法 ('spline', 'akima', 'pchip') [新增✓]
        enable_ransac_for_all: 对所有轨迹启用RANSAC扩展 [V4优化1]
        enable_adaptive_iteration: 自适应迭代直到收敛 [V4优化2]
        enable_velocity_smoothing: 速度平滑后处理 [V4优化3]
        max_iterations: 自适应迭代的最大轮数
        velocity_smooth_sigma: 高斯滤波的sigma参数
        enable_confidence_matching: 启用置信度匹配 [置信度优化]
        confidence_threshold: 置信度阈值 (0-1)
        confidence_top_k: 每次迭代最多选择前k个高分点
    """
    from simple_interpolation import fit_curve_seperate, fit_curve, point2result

    # 根据方法选择插值函数
    if interpolation_method == 'akima':
        from common.akima_interpolation import fit_curve_akima
        fit_func = fit_curve_akima
    elif interpolation_method == 'pchip':
        from common.pchip_interpolation import fit_curve_pchip
        fit_func = fit_curve_pchip
    else:
        fit_func = fit_curve_seperate

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
            # ⭐ V4优化2: 自适应迭代
            if enable_adaptive_iteration:
                # 自适应迭代直到收敛
                for j in range(max_iterations):
                    old_count = len(group_data)

                    try:
                        interp_pointx, interp_pointy, interp_pointz = fit_func(group_data)
                    except:
                        break

                    point_dict = {}
                    for index, point in enumerate(zip(interp_pointx, interp_pointy, interp_pointz)):
                        id = int(point[0]+0.1)
                        point_dict[id] = point[1:]

                    # 选择匹配方法
                    if enable_confidence_matching:
                        group_data, points = update_unlocated_with_confidence(
                            group_data, point_dict, points, BoatID,
                            update_distance=update_distance,
                            update_speed=update_speed,
                            update_angle=update_angle,
                            confidence_threshold=confidence_threshold,
                            top_k=confidence_top_k
                        )
                    else:
                        group_data, points = update_unlocated_custom(
                            group_data, point_dict, points, BoatID,
                            update_distance=update_distance,
                            update_speed=update_speed,
                            update_angle=update_angle,
                            enable_kinematic_constraints=enable_kinematic_constraints
                        )
                    group_data = group_data.sort_values(by='Time', ascending=True)

                    new_count = len(group_data)

                    # 收敛判断
                    if new_count == old_count:
                        break

                # 最后一次插值
                try:
                    interp_pointx, interp_pointy, interp_pointz = fit_func(group_data)
                except:
                    continue
            else:
                # 原有的固定迭代次数
                for j in range(iteration_rounds):
                    try:
                        interp_pointx, interp_pointy, interp_pointz = fit_func(group_data)
                    except:
                        break

                    point_dict = {}
                    for index, point in enumerate(zip(interp_pointx, interp_pointy, interp_pointz)):
                        id = int(point[0]+0.1)
                        point_dict[id] = point[1:]

                    # 选择匹配方法
                    if enable_confidence_matching:
                        group_data, points = update_unlocated_with_confidence(
                            group_data, point_dict, points, BoatID,
                            update_distance=update_distance,
                            update_speed=update_speed,
                            update_angle=update_angle,
                            confidence_threshold=confidence_threshold,
                            top_k=confidence_top_k
                        )
                    else:
                        group_data, points = update_unlocated_custom(
                            group_data, point_dict, points, BoatID,
                            update_distance=update_distance,
                            update_speed=update_speed,
                            update_angle=update_angle,
                            enable_kinematic_constraints=enable_kinematic_constraints
                        )
                    group_data = group_data.sort_values(by='Time', ascending=True)
                    interp_pointx, interp_pointy, interp_pointz = fit_func(group_data)

            # ⭐ V4优化1: 对所有轨迹启用RANSAC扩展
            if enable_ransac_for_all:
                original_count = len(group_data)
                group_data, points = ransac_fit_custom(
                    group_data, points, BoatID,
                    ransac_distance=ransac_distance,
                    ransac_min_points=ransac_min_points
                )
                # 如果RANSAC找到新点，再插值一次
                if len(group_data) > original_count:
                    try:
                        interp_pointx, interp_pointy, interp_pointz = fit_func(group_data)
                    except:
                        pass

        elif group_data.shape[0] == 1:
            # 使用参数化的 RANSAC
            if enable_ransac:
                group_data, points = ransac_fit_custom(
                    group_data, points, BoatID,
                    ransac_distance=ransac_distance,
                    ransac_min_points=ransac_min_points
                )
                if group_data.shape[0] == 1:
                    continue
                else:
                    interp_pointx, interp_pointy, interp_pointz = fit_curve(group_data)
            else:
                # 不启用RANSAC，跳过单点轨迹
                continue
        else:
            continue

        # ⭐ V4优化3: 速度平滑
        if enable_velocity_smoothing:
            interp_pointx, interp_pointy, interp_pointz = smooth_velocity(
                interp_pointx, interp_pointy, interp_pointz,
                sigma=velocity_smooth_sigma
            )

        track = point2result(interp_pointx, interp_pointy, interp_pointz, BoatID)
        tracks.extend(track)

    # 保存结果
    outdf = pd.DataFrame(tracks, columns=['ID', 'Time', 'LON', 'LAT'])
    df_sorted = outdf.sort_values(['ID', 'Time'])
    df_sorted['Time'] = pd.to_datetime(df_sorted['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    output_file = output_dir / 'results.csv'
    df_sorted.to_csv(output_file, index=False, float_format='%.6f')

