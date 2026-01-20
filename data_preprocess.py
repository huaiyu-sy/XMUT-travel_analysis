import pandas as pd
import numpy as np
from datetime import datetime
import os

# 读取原始数据（请替换为你的CSV文件实际路径）
raw_data_path = r"C:\Users\Alice\PycharmProjects\travel_analysis\data\Travel details dataset.csv"
df = pd.read_csv(raw_data_path)

# 1. 处理缺失值（删除关键字段缺失的行）
df = df.dropna(subset=['Duration (days)', 'Traveler age', 'Traveler gender', 'Accommodation cost', 'Transportation cost'])

# 2. 转换费用字段为数值类型（处理字符串格式的费用）
def clean_cost(cost):
    if isinstance(cost, str):
        cost = ''.join(filter(str.isdigit, cost))  # 提取数字部分
        return float(cost) if cost else 0.0
    return float(cost) if not pd.isna(cost) else 0.0

df['Accommodation cost'] = df['Accommodation cost'].apply(clean_cost)
df['Transportation cost'] = df['Transportation cost'].apply(clean_cost)

# 3. 提取季节和月份（从Start date）
def get_season(month):
    if month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    elif month in [9,10,11]:
        return 'Autumn'
    else:
        return 'Winter'

df['Start date'] = pd.to_datetime(df['Start date'], format='%m/%d/%Y', errors='coerce')
df['Month'] = df['Start date'].dt.month
df['Season'] = df['Month'].apply(get_season)

# 4. 划分年龄分段
def get_age_segment(age):
    if age <= 25:
        return '18-25'
    elif age <= 40:
        return '26-40'
    else:
        return '40+'

df['Age segment'] = df['Traveler age'].apply(get_age_segment)

# 5. 划分费用区间（住宿+交通总费用）
df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']
def get_cost_range(total_cost):
    if total_cost <= 1000:
        return 'Low'
    elif total_cost <= 3000:
        return 'Medium'
    else:
        return 'High'

df['Cost range'] = df['Total cost'].apply(get_cost_range)

# 6. 划分地域（从Destination提取）
def get_region(destination):
    if pd.isna(destination):
        return 'Unknown'
    destination = destination.lower()
    if any(reg in destination for reg in ['uk', 'france', 'germany', 'italy', 'spain']):
        return 'Europe'
    elif any(reg in destination for reg in ['usa', 'canada', 'mexico']):
        return 'North America'
    elif any(reg in destination for reg in ['thailand', 'indonesia', 'japan', 'korea', 'china']):
        return 'Asia'
    elif any(reg in destination for reg in ['australia', 'new zealand']):
        return 'Oceania'
    else:
        return 'Other'

df['Region'] = df['Destination'].apply(get_region)

# 保存预处理后的数据到static/data目录
save_path = os.path.join(os.path.dirname(__file__), 'data/cleaned_travel_data.csv')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)

print(f"数据预处理完成！清洁数据已保存到：{save_path}")
print(f"预处理后数据条数：{len(df)}")