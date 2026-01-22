import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 修复：添加项目根目录到Python路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)


def train_travel_duration_model():
    # 1. 读取data_preprocess.py生成的清洁数据CSV文件
    cleaned_data_path = os.path.join(
        project_root,
        'data/cleaned_travel_data.csv'  # 与data_preprocess.py的输出路径一致
    )

    # 校验文件是否存在
    if not os.path.exists(cleaned_data_path):
        print(f"错误：未找到清洁数据文件，请先运行data_preprocess.py！路径：{cleaned_data_path}")
        return

    df = pd.read_csv(cleaned_data_path)
    print(f"原始预处理数据条数：{len(df)}")

    # 2. 轻量缺失值校验（仅检查模型训练所需核心字段，不重复大规模清洗）
    train_required_fields = ['Traveler age', 'Accommodation cost', 'Transportation cost', 'Duration (days)']
    missing_stats = df[train_required_fields].isnull().sum()

    print("\n模型训练核心字段缺失值统计（轻量校验）：")
    print(missing_stats)

    # 仅删除核心字段仍存在的缺失行（极端情况防护，理论上预处理已处理）
    df = df.dropna(subset=train_required_fields)
    print(f"\n轻量校验后有效数据条数：{len(df)}")

    # 空数据校验
    if df.empty:
        print("警告：无有效训练数据（核心字段全缺失），无法训练模型！")
        return

    # 3. 划分特征（X）和目标（y）（适配CSV文件的字段名）
    x = df[['Traveler age', 'Accommodation cost', 'Transportation cost']]
    y = df['Duration (days)']

    # 4. 分割训练集和测试集（数据量校验）
    if len(df) < 5:  # 测试集占20%，至少需要5条数据
        print("警告：有效数据量过少（<5条），无法分割训练/测试集！")
        return

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 5. 训练线性回归模型
    model = LinearRegression()
    model.fit(x_train, y_train)

    # 6. 评估模型
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n模型训练完成，R²分数：{round(r2, 3)}")

    # 7. 保存模型
    model_path = os.path.join(project_root, 'static/model/travel_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"模型已保存到：{model_path}")


if __name__ == "__main__":
    train_travel_duration_model()