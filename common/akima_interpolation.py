"""
Akima插值实现
用于轨迹重建，相比B样条更鲁棒，无振荡
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator


def fit_curve_akima(group_data):
    """
    使用Akima插值拟合轨迹
    
    Args:
        group_data: pandas DataFrame，包含 Time, X, Y 列
    
    Returns:
        interp_pointx: 插值后的时间序列
        interp_pointy: 插值后的X坐标
        interp_pointz: 插值后的Y坐标
    """
    group_data = group_data.sort_values(by='Time', ascending=True)
    
    t = group_data['Time'].values
    x = group_data['X'].values
    y = group_data['Y'].values
    
    # 检查点数
    if len(t) < 2:
        return None
    
    # 时间范围
    t_min = np.min(t)
    t_max = np.max(t)
    
    # 生成均匀时间序列
    t_new = np.arange(int(t_min + 0.1), int(t_max + 0.1) + 1, 1)
    
    if len(t) == 2:
        # 2个点，线性插值
        x_new = np.interp(t_new, t, x)
        y_new = np.interp(t_new, t, y)
    else:
        # 3个点以上，使用Akima插值
        try:
            akima_x = Akima1DInterpolator(t, x)
            akima_y = Akima1DInterpolator(t, y)
            
            x_new = akima_x(t_new)
            y_new = akima_y(t_new)
        except:
            # 如果Akima失败，回退到线性插值
            x_new = np.interp(t_new, t, x)
            y_new = np.interp(t_new, t, y)
    
    return t_new, x_new, y_new

