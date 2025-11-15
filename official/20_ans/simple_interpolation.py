import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev, interp1d, UnivariateSpline
from matplotlib import pyplot as plt
from pathlib import Path
import sys

# 添加路径以便导入模块
sys.path.insert(0, str(Path(__file__).parent))
from CoordinateConvert import xy_to_lonlat
from preprocess import angle_between_rays_numpy
from tqdm import tqdm

# 输出文件夹（默认值，但不在模块级别创建，避免污染项目根目录）
OUTPUT_DIR = Path('20_ans/processed_data')

startx = 0 #200000
starty = 0 #3800000


def fit_curve(group_data):
    '''
    对排序的坐标数据使用曲线拟合并内插均匀采样点
    :param group_data:  pandas dataframe
    :return:
    '''
    x = group_data['Time'].values
    y = group_data['X'].values - startx  # 220000
    z = group_data['Y'].values - starty  # 3840000

    if len(group_data) == 2:
        k = 1  # 2个点，直线拟合

    elif len(group_data) == 3:
        k = 2  # 3个点，2阶曲线拟合

    elif len(group_data) > 3:
        k = 3  # 多个点，3阶曲线拟合

    else:
        return None

    t = np.linspace(0, 1, group_data.shape[0])

    tck, u = splprep([x, y, z], u=t, k=k)

    xmin = np.min(x)
    xmax = np.max(x)
    x_range = int(xmax - xmin)
    all_pointx, all_pointy, all_pointz = splev(np.linspace(0, 1, x_range), tck)

    # 分段线性映射计算均匀坐标（B样条无法得到均匀x坐标）
    x_to_u = interp1d(all_pointx, np.linspace(0, 1, x_range), kind='linear', bounds_error=True)
    target_x = np.arange(max(xmin, np.min(all_pointx)), min(xmax, np.max(all_pointx)))
    u_mapped = x_to_u(target_x)  # 找到对应的参数值
    interp_pointx, interp_pointy, interp_pointz = splev(u_mapped, tck)  # 使用映射后的参数插值
    return interp_pointx, interp_pointy, interp_pointz



def fit_curve_seperate(group_data, extend_ratio = 0.1):
    '''
    对排序的坐标数据使用曲线拟合并内插均匀采样点
    :param group_data:  pandas dataframe
    :return:
    '''
    group_data = group_data.sort_values(by='Time', ascending=True)

    x = group_data['Time'].values
    y = group_data['X'].values - startx  # 220000
    z = group_data['Y'].values - starty  # 3840000

    if len(group_data) == 2:
        k = 1  # 2个点，直线拟合

    elif len(group_data) == 3:
        k = 2  # 3个点，2阶曲线拟合

    elif len(group_data) > 3:
        k = 3  # 多个点，3阶曲线拟合

    else:
        return None

    spline_y = UnivariateSpline(x, y, s=0, k=k)
    spline_z = UnivariateSpline(x, z, s=0, k=k)

    xmin = np.min(x)
    xmax = np.max(x)

    #extend_range = int((xmax-xmin) * extend_ratio)

    #interp_pointx = np.arange(int(max(xmin-extend_range,0)), int(min(xmax+extend_range,1440)+1), 1)
    interp_pointx = np.arange(int(xmin +0.1), int(xmax + 0.1) + 1, 1)
    interp_pointy = spline_y(interp_pointx)
    interp_pointz = spline_z(interp_pointx)
    return interp_pointx, interp_pointy, interp_pointz


def update_unlocated(group_data, point_dict, points, BoatID):
    located = []
    for point in points:
        id = int(point[0])
        if id in point_dict:
            distance = np.sqrt((point_dict[id][0] - point[1]) ** 2 + (point_dict[id][1] - point[2]) ** 2)
            if distance < 600:
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

                if speed_earlier>800 or speed_later>800:  #与前后ESM点计算速度异常，不符合轨迹
                    continue

                if angle_between_rays_numpy(point[1], point[2], closest_earlier['X'].values[0], closest_earlier['Y'].values[0],
                                            closest_later['X'].values[0], closest_later['Y'].values[0], degrees=True)<90:
                    continue

                n = group_data.shape[0]
                group_data.loc[n] = [int(point[0]), BoatID, point[1], point[2]]
                located.append(point)

    points = [point for point in points if point not in located]
    return group_data, points


def ransac_fit(group_data, points, BoatID):
    x = group_data['Time'].values[0]
    y = group_data['X'].values[0] - startx  # 220000
    z = group_data['Y'].values[0] - starty

    max_point_num = 0
    max_points = None

    points_array = np.array(points)
    line_point = np.array((x, y, z))

    for point in tqdm(points):
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

        n = len(t_values)    #加速
        projections = np.empty((n, 3))
        projections[:, 0] = x0 + t_values * dx
        projections[:, 1] = y0 + t_values * dy
        projections[:, 2] = z0 + t_values * dz
        #projections = line_point + t_values.reshape(-1, 1) * line_direction

        # 计算距离（yz平面距离）
        yz_distances = np.sqrt(
          (points_array[:, 1] - projections[:, 1]) ** 2 +
          (points_array[:, 2] - projections[:, 2]) ** 2
        )

        mask = yz_distances < 500
        if mask.sum() > max_point_num:
            max_point_num = mask.sum()
            max_points = points_array[mask]

    if max_point_num >= 5:
        for point in max_points:
            if int(point[0])== x:
                continue
            n = group_data.shape[0]
            group_data.loc[n] = [int(point[0]), BoatID, point[1], point[2]]
            point_list = point.tolist()
            if point_list in points:
                points.remove(point_list)

    group_data = group_data.sort_values(by='Time', ascending=True)
    return group_data, points



def point2result(x, y, z, BoatID):
    start_time = pd.to_datetime('2025-06-25 05:00:00')
    track = []
    for i in range(x.shape[0]):
        time = start_time + pd.to_timedelta(int(x[i]+0.1), unit='m')
        coord = xy_to_lonlat(y[i], z[i])
        track.append([BoatID, time, coord[0], coord[1]])
    return track


def align_rader2ESM(located_ESM_points, radar_points, output_dir=OUTPUT_DIR):
    radar_points = pd.read_csv(radar_points)

    points = []
    for i, row in radar_points.iterrows():
        points.append([row['Time'], row['X']-startx, row['Y']-starty])

    located_ESM_points = pd.read_csv(located_ESM_points)
    located_ESM_points = located_ESM_points.sort_values(by='Time', ascending=True)
    sorted_BoatIDs = located_ESM_points.groupby('BoatID').size().sort_values(ascending=False).index

    tracks = []
    for i, BoatID in tqdm(enumerate(sorted_BoatIDs, 1), total=len(sorted_BoatIDs), desc='生成轨迹'):
        # 先使用B样条拟合轨迹
        group_data = located_ESM_points[located_ESM_points['BoatID'] == BoatID].copy()
        #group_data['Time'] = group_data['Time'].astype('Int64')
        group_data['BoatID'] = group_data['BoatID'].astype('Int64')

        if group_data.shape[0]>1:
            #每个轨迹进行3轮插值
            nrand = 3
            for j in range(nrand):
                try:
                    interp_pointx, interp_pointy, interp_pointz = fit_curve_seperate(group_data)
                    #interp_pointx, interp_pointy, interp_pointz = fit_curve(group_data)

                except:
                    break

                point_dict = {}
                for index, point in enumerate(zip(interp_pointx, interp_pointy, interp_pointz)):
                    id = int(point[0]+0.1)
                    point_dict[id] = point[1:]

                group_data, points =update_unlocated(group_data, point_dict, points, BoatID)

                group_data = group_data.sort_values(by='Time', ascending=True)
                interp_pointx, interp_pointy, interp_pointz = fit_curve_seperate(group_data)

        elif group_data.shape[0] == 1:
            group_data, points = ransac_fit(group_data, points, BoatID)
            if group_data.shape[0] == 1:
                continue
            else:
                interp_pointx, interp_pointy, interp_pointz = fit_curve(group_data)

        else:
            continue

        track = point2result(interp_pointx, interp_pointy, interp_pointz, BoatID)
        tracks.extend(track)

    outdf = pd.DataFrame(tracks, columns=['ID', 'Time', 'LON', 'LAT'])
    df_sorted = outdf.sort_values(['ID', 'Time'])
    df_sorted['Time'] = pd.to_datetime(df_sorted['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    output_file = output_dir / 'results.csv'
    df_sorted.to_csv(output_file, index=False, float_format='%.6f')
    
    print(f"\n✓ 轨迹生成完成 - 共{len(df_sorted)}个点，{df_sorted['ID'].nunique()}艘船")
    print(f"结果已保存到: {output_file}")



if __name__ == '__main__':
    # 使用 processed_data 文件夹中的文件
    located_ESM_file = OUTPUT_DIR / 'located_ESM_points_updated.csv'
    radar_file = OUTPUT_DIR / 'deduplacate_radar_points_updated.csv'
    
    align_rader2ESM(str(located_ESM_file), str(radar_file))

