"""
坐标转换工具模块

提供经纬度与平面坐标系之间的转换，以及雷达坐标转换和ESM空间交点计算功能。
"""
import numpy as np
from pyproj import Transformer

# 将经纬度转换为平面坐标系
def lonlat_to_xy(lon, lat):
    """
    将经纬度转换为平面坐标系（使用WGS84坐标系）
    
    Args:
        lon: 经度
        lat: 纬度
        
    Returns:
        np.array: 平面坐标 [x, y]
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.array([x, y])

# 将平面坐标系转换为经纬度  
def xy_to_lonlat(x, y):
    """
    将平面坐标系转换为经纬度（使用WGS84坐标系）
    
    Args:
        x: 平面坐标x
        y: 平面坐标y
        
    Returns:
        np.array: 经纬度 [lon, lat]
    """
    transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return np.array([lon, lat])

# 雷达极坐标转换为直角坐标
def radar_conversion(sensor_xy, r, theta):
    """
    雷达极坐标转换为直角坐标
    
    Args:
        sensor_xy: 传感器位置坐标 [x, y]
        r: 距离
        theta: 角度（度）
        
    Returns:
        np.array: 转换后的直角坐标 [x, y]
    """
    x = sensor_xy[0] + r * np.sin(np.radians(theta))
    y = sensor_xy[1] + r * np.cos(np.radians(theta))
    return np.array([x, y])

# 计算两观测点及其观测方向的交点坐标（前方交会）
def space_intersection(sensor_xy1, sensor_xy2, theta1, theta2):
    """
    计算两观测点及其观测方向的交点坐标（前方交会）
    
    Args:
        sensor_xy1: 第一个观测点坐标 [x1, y1]
        sensor_xy2: 第二个观测点坐标 [x2, y2]
        theta1: 第一个观测点的观测方向（角度，正北为0，顺时针为正）
        theta2: 第二个观测点的观测方向（角度，正北为0，顺时针为正）
        
    Returns:
        np.array: 交点坐标 [x, y]
    """
    # 角度转弧度
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)

    # 观测点坐标（注意坐标的不同，测量坐标系xy相反）
    y1, x1 = sensor_xy1
    y2, x2 = sensor_xy2
    
    # 计算交点坐标
    sin_diff = np.sin(theta1_rad - theta2_rad)
    
    y = ((x1 * np.sin(theta1_rad) * np.cos(theta2_rad) - 
          x2 * np.cos(theta1_rad) * np.sin(theta2_rad) + 
          (y2 - y1) * np.cos(theta1_rad) * np.cos(theta2_rad)) / sin_diff)
    
    x = ((y2 * np.sin(theta1_rad) * np.cos(theta2_rad) - 
          y1 * np.cos(theta1_rad) * np.sin(theta2_rad) + 
          (x1 - x2) * np.sin(theta1_rad) * np.sin(theta2_rad)) / sin_diff)
    
    return np.array([x, y])

# 调整角度值, 转换为以正北为0度，顺时针增加的角度值
def adjust_angle(theta):
    """
    调整角度值, 转换为以正北为0度，顺时针增加的角度值
    
    Args:
        theta: 输入角度
        
    Returns:
        float: 调整后的角度值
    """
    if theta < 0:
        theta = abs(theta) + 90
    elif theta >= 0 and theta <= 90:
        theta = 90 - theta
    else:
        theta = 360 - theta + 90
    return theta

if __name__ == '__main__':
    # 示例用法
    sensor1 = [122, 37] # 2d radar
    sensor1_xy = lonlat_to_xy(*sensor1)
    r, theta = 100, 30 # 赛题记录的雷达极坐标
    theta = adjust_angle(theta)
    target_xy = radar_conversion(sensor1_xy, r, theta)
    target_lonlat = xy_to_lonlat(*target_xy)
    
    sensor2 = [123, 36] # ESM
    sensor3 = [124, 35] # ESM
    sensor2_xy = lonlat_to_xy(*sensor2)
    sensor3_xy = lonlat_to_xy(*sensor3)
    theta1, theta2 = 300, 200 # 赛题记录的ESM方位角
    theta1 = adjust_angle(theta1)
    theta2 = adjust_angle(theta2)
    target2_xy = space_intersection(sensor2_xy, sensor3_xy, theta1, theta2)
    target2_lonlat = xy_to_lonlat(*target2_xy)