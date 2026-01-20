import os
import re
import sys
import django
import pandas as pd
from django.core.wsgi import get_wsgi_application

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'travel_analysis.settings')
django.setup()

# 导入模型
from travel_app.models import TravelRecord


def clean_currency_value(value):
    """清理费用字段中的货币符号和空格，转换为浮点数"""
    if pd.isna(value):
        return None
    # 移除所有非数字和小数点的字符
    cleaned = re.sub(r'[^\d.]', '', str(value).strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(date_str):
    """解析CSV中的日期字符串（MM/DD/YYYY）为Django的Date对象"""
    if pd.isna(date_str):
        return None
    try:
        # 将MM/DD/YYYY格式转换为pandas日期对象，再转为字符串
        return pd.to_datetime(date_str, format='%m/%d/%Y').date()
    except ValueError:
        return None


def import_travel_data():
    # CSV文件路径（替换为你的CSV文件实际路径）
    csv_path = r"C:\Users\Alice\PycharmProjects\travel_analysis\data\Travel details dataset.csv"

    # 读取CSV文件（保留所有字段）
    df = pd.read_csv(csv_path)

    # ========== 1. 数据预处理：清理和转换 ==========
    # 1.1 清理费用字段（移除货币符号、空格）
    df['Accommodation cost'] = df['Accommodation cost'].apply(clean_currency_value)
    df['Transportation cost'] = df['Transportation cost'].apply(clean_currency_value)

    # 1.2 解析日期字段（MM/DD/YYYY → YYYY-MM-DD）
    df['Start date'] = df['Start date'].apply(parse_date)
    df['End date'] = df['End date'].apply(parse_date)

    # 1.3 替换空值为None（避免Django报错）
    df = df.where(pd.notna(df), None)

    # ========== 2. 批量创建数据记录 ==========
    travel_records = []
    for _, row in df.iterrows():
        # 跳过Trip ID为空的记录（主键不能为空）
        if row['Trip ID'] is None:
            continue

        record = TravelRecord(
            # 旅行基本信息
            trip_id=row['Trip ID'],
            destination=row['Destination'],
            start_date=row['Start date'],
            end_date=row['End date'],
            duration_days=row['Duration (days)'],

            # 旅行者信息
            traveler_name=row['Traveler name'],
            traveler_age=row['Traveler age'],
            traveler_gender=row['Traveler gender'],
            traveler_nationality=row['Traveler nationality'],

            # 费用和交通信息
            accommodation_type=row['Accommodation type'],
            accommodation_cost=row['Accommodation cost'],
            transportation_type=row['Transportation type'],
            transportation_cost=row['Transportation cost']
        )
        travel_records.append(record)

    # ========== 3. 批量导入数据库 ==========
    # 先删除已存在的记录（避免重复导入，可选）
    # TravelRecord.objects.all().delete()

    # 批量创建（效率更高）
    TravelRecord.objects.bulk_create(travel_records)

    # 输出导入结果
    print(f"数据导入成功！共导入 {len(travel_records)} 条旅行记录")
    print(f"CSV文件原始行数：{len(df)}")
    print(f"跳过的无效行数：{len(df) - len(travel_records)}")


if __name__ == "__main__":
    import_travel_data()