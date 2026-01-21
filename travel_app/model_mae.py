# 导入必要库
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error  # 库函数计算MAE
import os

# ---------------------- 步骤1：配置路径与参数 ----------------------
# 请根据你的文件实际路径修改（默认与项目根目录下的data文件夹对应）
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/cleaned_travel_data.csv")
# 特征列（与models.py、views.py一致）
FEATURE_COLS = ["Traveler age", "Accommodation cost", "Transportation cost"]
# 目标列（旅行周期，与数据集中列名一致）
TARGET_COL = "Duration (days)"
# 数据划分参数（与views.py中的模型训练逻辑一致）
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------- 步骤2：加载并预处理数据 ----------------------
# 加载CSV数据
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 成功加载数据，共{len(df)}条记录")
except FileNotFoundError:
    print(f"❌ 数据文件未找到，请检查路径：{DATA_PATH}")
    exit()

# 筛选特征和目标列，删除可能的缺失值（避免模型报错）
df_model = df[FEATURE_COLS + [TARGET_COL]].dropna()
print(f"✅ 预处理后数据量：{len(df_model)}条（已删除缺失值）")

# 分离特征矩阵X和目标变量y
X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

# ---------------------- 步骤3：划分训练集与测试集 ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)
print(f"\n📊 数据划分结果：")
print(f"训练集：{len(X_train)}条样本 | 测试集：{len(X_test)}条样本")

# ---------------------- 步骤4：训练线性回归模型（与views.py一致） ----------------------
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\n✅ 模型训练完成，线性回归系数：")
for col, coef in zip(FEATURE_COLS, model.coef_):
    print(f"  {col}系数：{coef:.6f}（系数正负表示对旅行周期的正负影响）")
print(f"  模型截距：{model.intercept_:.6f}")

# ---------------------- 步骤5：在测试集上预测 ----------------------
y_pred = model.predict(X_test)
# 确保预测值合理（旅行周期至少1天）
y_pred = np.maximum(y_pred, 1.0)

# 打印前5条测试集预测结果（直观验证）
print(f"\n🔍 测试集前5条预测示例：")
sample_result = pd.DataFrame({
    "旅行者年龄": X_test["Traveler age"].values[:5],
    "住宿费用": X_test["Accommodation cost"].values[:5],
    "交通费用": X_test["Transportation cost"].values[:5],
    "实际旅行周期（天）": y_test.values[:5],
    "预测旅行周期（天）": np.round(y_pred[:5], 1)
})
print(sample_result)

# ---------------------- 步骤6：计算测试集MAE ----------------------
# 方法1：使用scikit-learn库函数（推荐，简洁高效）
mae_sklearn = mean_absolute_error(y_test, y_pred)

# 方法2：手动计算MAE（验证原理，与库函数结果一致）
absolute_errors = np.abs(y_test.values - y_pred)  # 每个样本的绝对误差
mae_manual = np.mean(absolute_errors)  # 绝对误差的平均值

# 输出MAE结果
print(f"\n📈 测试集MAE计算结果：")
print(f"  库函数计算MAE：{mae_sklearn:.2f} 天")
print(f"  手动计算MAE：{mae_manual:.2f} 天")
print(f"  解释：模型预测的旅行周期与实际值的平均绝对误差为 {mae_sklearn:.2f} 天，误差越小模型越精准")

# ---------------------- （可选）保存MAE结果到文件 ----------------------
mae_save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/model/model_mae.pkl")
os.makedirs(os.path.dirname(mae_save_path), exist_ok=True)
pd.DataFrame({
    "测试集样本数": [len(X_test)],
    "MAE（天）": [mae_sklearn],
    "计算时间": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]
}).to_csv(mae_save_path, index=False)
print(f"\n💾 MAE结果已保存到：{mae_save_path}")