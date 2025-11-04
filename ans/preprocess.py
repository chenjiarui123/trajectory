import pandas as pd
from math import cos, sin, radians, sqrt
from CoordinateConvert import lonlat_to_xy, xy_to_lonlat, space_intersection, adjust_angle, radar_conversion
import numpy as np
from shapely.geometry import Point, MultiPoint, LineString
from tqdm import tqdm

def load_sensor_coords(sensor_coords_file):
    '''
    载入sensor信息
    :param sensor_coords_file:
    :return:
    '''
    sensors = {}
    with open(sensor_coords_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split()
            if len(parts) != 4:  # 检查格式
                continue
            id, type, coordx, coordy = parts
            x, y = lonlat_to_xy(float(coordx), float(coordy))
            sensors[id] = {'type': type, 'coord':(x, y)}
    return sensors



def get_points(rader_file, sensors, start_time = pd.to_datetime('2025-06-25 05:00:00')):
    '''
    计算radar点和esm点平面坐标
    :param rader_file:
    :param sensors:
    :param start_time:
    :return:
    '''
    print(f"\n[步骤1] 读取雷达数据文件: {rader_file}")
    points = pd.read_csv(rader_file, encoding='utf-8')

    points['Time'] = ((pd.to_datetime(points['Time']) - start_time).dt.total_seconds() / 60).astype(int)
    points['SensorID'] = points['SensorID'].astype('Int64')

    ESM_ids = [int(id) for id in sensors if sensors[id]['type'] == 'ESM']
    ESM_points = points[points['SensorID'].isin(ESM_ids)]
    rader_points = points[~points['SensorID'].isin(ESM_ids)]
    print(f'  ESM记录数: {len(ESM_points)}')
    print(f'  雷达记录数: {len(rader_points)}')

    print(f"\n[步骤1.1] 转换雷达点坐标...")
    get_rader_points_xy(rader_points, sensors)
    
    print(f"\n[步骤1.2] 处理ESM交叉定位...")
    get_ESM_points_xy(ESM_points, sensors)


def get_rader_points_xy(rader_points, sensors):
    points = []

    for index, row in tqdm(rader_points.iterrows(), total=len(rader_points), desc='处理雷达点'):
        sensor_id = row['SensorID']
        assert sensors[str(sensor_id)]['type'] == '2D雷达'

        sensor_xy = sensors[str(sensor_id)]['coord']
        point_xy = radar_conversion(sensor_xy, row['Distance'], adjust_angle(row['Azimuth']))
        points.append([row['Time'], point_xy[0], point_xy[1]])

    radar_points = pd.DataFrame(points, columns=['Time', 'X', 'Y'])
    radar_points.to_csv('radar_points.csv', index=False)
    print(f"  已保存: radar_points.csv ({len(radar_points)} 个点)")
    return radar_points


def check_distance(point1, point2,  distance=200000):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < distance


def get_ESM_points_xy(ESM_points, sensors):
    located_ESM_points = []
    unlocated_ESM_points = []

    grouped_ESM_points = ESM_points.groupby(['Time','BoatID'])
    for group, data in tqdm(grouped_ESM_points, desc='处理ESM点'):
        if data.shape[0] < 2:
            unlocated_ESM_points.append(data)

        else:
            if data.shape[0] > 2:
                print('Warning: ESM points more than 2, only 2 used')

            sensor_id1 = int(data.iloc[0]['SensorID'])
            sensor_id2 = int(data.iloc[1]['SensorID'])
            sensor_xy1 = sensors[str(sensor_id1)]['coord']
            sensor_xy2 = sensors[str(sensor_id2)]['coord']
            point_xy = space_intersection(sensor_xy1, sensor_xy2, adjust_angle(data.iloc[0]['Azimuth']), adjust_angle(data.iloc[1]['Azimuth']))

            if check_distance(point_xy, sensor_xy1) and check_distance(point_xy, sensor_xy2):   # 判断点是否在传感器范围内
                located_ESM_points.append([group[0], group[1], point_xy[0], point_xy[1]])
            else:
                print('Warning: ESM points not in sensor range')
                unlocated_ESM_points.append(data)

    unlocated_ESM_points = pd.concat(unlocated_ESM_points, ignore_index=True)
    unlocated_ESM_points.to_csv('unlocated_ESM_points.csv', index=False)

    located_ESM_points= pd.DataFrame(located_ESM_points, columns=['Time', 'BoatID', 'X', 'Y'])
    located_ESM_points.to_csv('located_ESM_points.csv', index=False)
    
    print(f"  已定位ESM点: {len(located_ESM_points)}")
    print(f"  未定位ESM点: {len(unlocated_ESM_points)}")
    print(f"  已保存: located_ESM_points.csv, unlocated_ESM_points.csv")


def dedumplacate(radar_points_file, located_ESM_points_file, deduplacate_radar_file, safe_distance=200):
    '''
    同一时间，距离小于200的点去重，包括内部去重和与已知ESM去重
    :param radar_points_file:
    :param located_ESM_points_file:
    :param deduplacate_radar_file:
    :param safe_distance:
    :return:
    '''
    print(f"\n[步骤2] 雷达点去重（安全距离={safe_distance}m）")
    radar_points = pd.read_csv(radar_points_file, encoding='utf-8')
    radar_points['BoatID'] = -1

    located_ESM_points = pd.read_csv(located_ESM_points_file, encoding='utf-8')

    located_points = pd.concat([radar_points, located_ESM_points], ignore_index=True)
    print(f"  待去重点数: {len(radar_points)}")
    located_points_group = located_points.groupby('Time')

    retain_points = []
    deduplacate_points = []
    for time, data in tqdm(located_points_group, desc='去重雷达点'):
        data = data.sort_values(by='BoatID', ascending=False)

        points = []
        for i, row in data.iterrows():
            if row['BoatID'] != -1:
                points.append(Point(row['X'], row['Y']))

            else:
                cur_point = Point(row['X'], row['Y'])
                buffer = MultiPoint(points).buffer(safe_distance)
                if not buffer.contains(cur_point):
                    points.append(cur_point)
                    retain_points.append(row)

                else:
                    deduplacate_points.append(row)

    retain_radar_points = pd.concat(retain_points, axis=1).transpose()
    retain_radar_points.to_csv(deduplacate_radar_file, index=False)
    print(f"  保留点数: {len(retain_radar_points)}, 去重点数: {len(deduplacate_points)}")
    print(f"  已保存: {deduplacate_radar_file}")


def generate_ray(start_point, angle_degrees, length=200000):
    """
    生成射线

    Args:
        start_point: 起点坐标 (x, y)
        angle_degrees: 角度（度）
        length: 射线长度

    Returns:
        end_point: 射线终点坐标
    """
    angle_rad = radians(angle_degrees)
    dx = cos(angle_rad) * length
    dy = sin(angle_rad) * length
    end_point = (start_point[0] + dx, start_point[1] + dy)
    return end_point


def angle_between_rays_numpy(x0, y0, x1, y1, x2, y2, degrees=True):
    """
    使用numpy计算两条射线夹角
    """
    # 转换为numpy数组
    vec1 = np.array([x1 - x0, y1 - y0])
    vec2 = np.array([x2 - x0, y2 - y0])

    # 计算点积
    dot_product = np.dot(vec1, vec2)

    # 计算模长
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    if mag1 == 0 or mag2 == 0:
        #raise ValueError("射线长度不能为零")
        return 360

    # 计算夹角
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)

    return np.degrees(angle_rad) if degrees else angle_rad

def merge_unlocated_ESM_with_rader(unlocated_ESM_file, radar_file, located_ESM_file, sensor_coords_file, radar_file_updated, located_ESM_updated):
    """
    对未定位的ESM点与rader点匹配（可能会出现误匹配的情况 即rader点符合角度要求，但是在更近处有另外的船）
    :param unlocated_ESM_file:
    :param radar_file:
    :param output_file:
    :return:
    """
    print(f"\n[步骤3] 合并未定位ESM点与雷达点")
    unlocated_ESM = pd.read_csv(unlocated_ESM_file)
    unlocated_ESM['SensorID'] = unlocated_ESM['SensorID'].astype('Int64')
    print(f"  未定位ESM数: {len(unlocated_ESM)}")

    located_ESM_points = pd.read_csv(located_ESM_file)

    radar = pd.read_csv(radar_file)
    print(f"  雷达点数: {len(radar)}")
    sensors = load_sensor_coords(sensor_coords_file)

    located =  []
    #remove_indexes = []
    for i, row in tqdm(unlocated_ESM.iterrows(), total=len(unlocated_ESM), desc='合并未定位ESM'):
        sensor_xy = sensors[str(row['SensorID'])]['coord']
        azimuth = row['Azimuth']
        time = row['Time']
        end_xy =generate_ray(sensor_xy, azimuth)

        senor_point = Point(sensor_xy)
        condidates = radar[radar['Time']==time]

        min_dis = 200000
        min_point = None
        index = -1
        for j, candidate in condidates.iterrows():
            point = Point(candidate['X'], candidate['Y'])
            if angle_between_rays_numpy(sensor_xy[0], sensor_xy[1], point.x, point.y, end_xy[0], end_xy[1])<0.001:
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
                        (closest_earlier['X'].values[0] - point.x) ** 2 + (
                                    closest_earlier['Y'].values[0] - point.y) ** 2)
                                     / (time - closest_earlier['Time'].values[0]))
                    if speed_earlier > 1000:
                         continue

                if later.shape[0] > 0:
                    if closest_later['Time'].values[0] == time:
                        continue
                    speed_later = (np.sqrt(
                        (closest_later['X'].values[0] - point.x) ** 2 + (closest_later['Y'].values[0] - point.y) ** 2)
                                   / (closest_later['Time'].values[0] - time))

                    if speed_later > 1000:  # 与前后ESM点计算速度异常，不符合轨迹
                        continue

                if point.distance(senor_point) < min_dis:
                    min_dis = point.distance(senor_point)
                    min_point = point
                    index = j

        if min_point is not None and index != -1:
            located.append([time, row['BoatID'], min_point.x, min_point.y])
            #remove_indexes = []
            radar = radar.drop(index)
    
    print(f"\n  新匹配到的ESM点数: {len(located)}")
    new_located = pd.DataFrame(located, columns=['Time', 'BoatID', 'X', 'Y'])
    located_ESM_points = pd.concat([located_ESM_points, new_located])
    located_ESM_points.to_csv(located_ESM_updated, index=False)
    radar.to_csv(radar_file_updated, index=False)
    print(f"  更新后ESM点总数: {len(located_ESM_points)}")
    print(f"  剩余雷达点数: {len(radar)}")
    print(f"  已保存: {located_ESM_updated}, {radar_file_updated}")


if __name__ == '__main__':
    print("=" * 70)
    print("参考答案预处理流程")
    print("=" * 70)
    
    sensor_coords_file = 'ans/sensors.txt'
    rader_file = 'official/test/radar_detection.csv'  # 修改为实际路径
    
    print(f"\n加载传感器配置: {sensor_coords_file}")
    sensors = load_sensor_coords(sensor_coords_file)
    print(f"  加载成功: {len(sensors)} 个传感器")
    
    # 步骤1: 提取点坐标
    get_points(rader_file, sensors)
    
    # 步骤2: 去重
    dedumplacate('radar_points.csv', 'located_ESM_points.csv', 'deduplacate_radar_points.csv')
    
    # 步骤3: 合并未定位ESM
    merge_unlocated_ESM_with_rader('unlocated_ESM_points.csv', 'deduplacate_radar_points.csv', 'located_ESM_points.csv',
                                   'ans/sensors.txt', 'deduplacate_radar_points_updated.csv', 'located_ESM_points_updated.csv')
    
    print("\n" + "=" * 70)
    print("预处理完成！")
    print("=" * 70)
    print("\n输出文件:")
    print("  - radar_points.csv")
    print("  - located_ESM_points.csv")
    print("  - unlocated_ESM_points.csv")
    print("  - deduplacate_radar_points.csv")
    print("  - deduplacate_radar_points_updated.csv")
    print("  - located_ESM_points_updated.csv")
    print("\n下一步运行: python ans/simple_interpolation-bak.py")