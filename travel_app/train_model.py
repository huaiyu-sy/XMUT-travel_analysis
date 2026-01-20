import os
import sys
import pandas as pd
import joblib
import django
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer  # 导入缺失值填充工具
from django.core.wsgi import get_wsgi_application

# 修复：添加项目根目录到Python路径（解决模块导入问题）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)

# 配置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'travel_analysis.settings')
django.setup()

from travel_app.models import TravelRecord


def train_travel_duration_model():
    # 1. 从数据库获取训练数据（修正字段名：travel_duration → duration_days）
    records = TravelRecord.objects.all().values(
        'traveler_age', 'accommodation_cost', 'transportation_cost', 'duration_days'
    )
    df = pd.DataFrame(list(records))

    # 2. 数据校验与清洗（核心修复：处理缺失值）
    print(f"原始数据条数：{len(df)}")
    if df.empty:
        print("警告：数据库中无TravelRecord数据，无法训练模型！")
        return

    # 2.1 查看缺失值分布（调试用，可保留）
    print("\n缺失值统计：")
    print(df.isnull().sum())

    # 2.2 处理缺失值：
    # 方案1：删除包含NaN的行（简单直接，适合缺失量少的情况）
    df = df.dropna()
    print(f"\n删除缺失值后数据条数：{len(df)}")

    # 方案2（可选）：用均值填充缺失值（适合缺失量多的情况，注释掉方案1后启用）
    # imputer = SimpleImputer(strategy='mean')  # 均值填充
    # df[['traveler_age', 'accommodation_cost', 'transportation_cost']] = imputer.fit_transform(
    #     df[['traveler_age', 'accommodation_cost', 'transportation_cost']]
    # )

    # 3. 划分特征（X）和目标（y）（修正字段名）
    X = df[['traveler_age', 'accommodation_cost', 'transportation_cost']]
    y = df['duration_days']

    # 4. 分割训练集和测试集（添加空数据校验）
    if len(df) < 5:  # 数据量太少无法分割（测试集占20%）
        print("警告：有效数据量过少（<5条），无法分割训练/测试集！")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. 评估模型
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n模型训练完成，R²分数：{round(r2, 3)}")

    # 7. 保存模型
    model_path = os.path.join(project_root, 'static/model/travel_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"模型已保存到：{model_path}")


if __name__ == "__main__":
    train_travel_duration_model()