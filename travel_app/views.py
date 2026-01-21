from django.db import models
from travel_app.models import TravelRecord
import os
import pandas as pd
import joblib
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from .models import TravelRecord
from datetime import datetime
import logging
from django.views.decorators.csrf import csrf_exempt

# 配置日志记录器
logger = logging.getLogger('travel_app')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('travel_app_error.log'),
        logging.StreamHandler()
    ]
)


# ---------------------- 1. 首页视图 ----------------------
def index(request):
    """首页视图：展示4个功能入口"""
    try:
        return render(request, 'index.html')
    except Exception as e:
        logger.error(f"首页视图加载失败: {str(e)}", exc_info=True)
        return render(request, 'error.html', {"error_msg": "首页加载失败，请稍后重试"}, status=500)


# ---------------------- 2. 多维度可视化视图 ----------------------
def multi_visualization(request):
    """多维度可视化视图：支持季节/地域筛选，生成多维度旅行周期分析图表"""
    try:
        # 加载清洁数据
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        data_path = os.path.join(data_dir, 'cleaned_travel_data.csv')

        # 异常1：数据文件不存在/路径错误
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"清洁数据文件不存在：{data_path}")

        # 异常2：CSV文件读取失败（格式错误、权限不足等）
        try:
            df = pd.read_csv(data_path)
        except PermissionError:
            raise PermissionError(f"无读取权限：{data_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV文件为空：{data_path}")
        except pd.errors.ParserError:
            raise ValueError(f"CSV文件格式错误，无法解析：{data_path}")

        # 1. 获取前端筛选参数（季节、地域）
        selected_season = request.GET.get('season', '')
        selected_region = request.GET.get('region', '')

        # 2. 筛选数据（处理空值/无效值）
        filtered_df = df.copy()
        if selected_season and selected_season in df['Season'].unique():
            filtered_df = filtered_df[filtered_df['Season'] == selected_season]
        if selected_region and selected_region in df['Region'].unique():
            filtered_df = filtered_df[filtered_df['Region'] == selected_region]

        # 3. 计算5类对比数据（处理空数据场景）
        def safe_groupby_mean(df, group_col, value_col):
            """安全分组计算均值，处理空分组/空值"""
            if group_col not in df.columns or value_col not in df.columns:
                raise KeyError(f"缺失必要列：{group_col} 或 {value_col}")
            if df.empty:
                return pd.Series([], dtype=float)
            result = df.groupby(group_col)[value_col].mean().round(2)
            return result.fillna(0)  # 空值填充为0

        # 3.1 性别 vs 旅行周期
        gender_data = safe_groupby_mean(filtered_df, 'Traveler gender', 'Duration (days)')
        gender_labels = gender_data.index.tolist()
        gender_values = gender_data.values.tolist()

        # 3.2 年龄分段 vs 旅行周期
        age_data = safe_groupby_mean(filtered_df, 'Age segment', 'Duration (days)')
        age_labels = age_data.index.tolist()
        age_values = age_data.values.tolist()

        # 3.3 费用区间 vs 旅行周期
        cost_data = safe_groupby_mean(filtered_df, 'Cost range', 'Duration (days)')
        cost_labels = cost_data.index.tolist()
        cost_values = cost_data.values.tolist()

        # 3.4 地域 vs 旅行周期
        region_data = safe_groupby_mean(filtered_df, 'Region', 'Duration (days)')
        region_labels = region_data.index.tolist()
        region_values = region_data.values.tolist()

        # 3.5 季节 vs 旅行周期
        season_data = safe_groupby_mean(filtered_df, 'Season', 'Duration (days)')
        season_labels = season_data.index.tolist()
        season_values = season_data.values.tolist()

        # 4. 准备ECharts基础配置
        base_option = {
            "title": {"text": "多维度旅行周期分析", "left": "center", "fontSize": 18},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"data": ["平均旅行周期（天）"], "bottom": 10},
            "toolbox": {
                "feature": {
                    "saveAsImage": {"type": "png", "name": "旅行数据可视化"},
                    "magicType": {"type": ["bar", "line", "pie"],
                                  "title": {"bar": "柱状图", "line": "折线图", "pie": "饼图"}}
                },
                "right": 20
            },
            "dataZoom": [{"type": "slider", "xAxisIndex": 0, "bottom": 40}]
        }

        # 初始渲染性别对比（柱状图）
        init_option = {
            **base_option,
            "xAxis": {"type": "category", "data": gender_labels, "axisLabel": {"rotate": 0}},
            "yAxis": {"type": "value", "name": "天数", "min": 0},
            "series": [
                {"name": "平均旅行周期（天）", "type": "bar", "data": gender_values, "itemStyle": {"color": "#4895ef"}}]
        }

        # 传递所有数据到前端
        context = {
            "init_option": init_option,
            "gender_data": {"labels": gender_labels, "values": gender_values},
            "age_data": {"labels": age_labels, "values": age_values},
            "cost_data": {"labels": cost_labels, "values": cost_values},
            "region_data": {"labels": region_labels, "values": region_values},
            "season_data": {"labels": season_labels, "values": season_values},
            "all_seasons": df['Season'].unique().tolist(),
            "all_regions": df['Region'].unique().tolist(),
            "selected_season": selected_season,
            "selected_region": selected_region
        }
        return render(request, 'visualization.html', context)

    # 细分异常处理
    except FileNotFoundError as e:
        logger.error(f"可视化视图 - 文件不存在: {str(e)}", exc_info=True)
        return render(request, 'error.html', {"error_msg": f"数据文件缺失：{str(e)}"}, status=404)
    except PermissionError as e:
        logger.error(f"可视化视图 - 权限不足: {str(e)}", exc_info=True)
        return render(request, 'error.html', {"error_msg": f"无权限读取数据文件：{str(e)}"}, status=403)
    except KeyError as e:
        logger.error(f"可视化视图 - 列缺失: {str(e)}", exc_info=True)
        return render(request, 'error.html', {"error_msg": f"数据文件缺少必要字段：{str(e)}"}, status=400)
    except ValueError as e:
        logger.error(f"可视化视图 - 数据格式错误: {str(e)}", exc_info=True)
        return render(request, 'error.html', {"error_msg": f"数据格式错误：{str(e)}"}, status=400)
    except Exception as e:
        logger.error(f"可视化视图 - 未知错误: {str(e)}", exc_info=True)
        return render(request, 'error.html', {"error_msg": "可视化加载失败，请联系管理员"}, status=500)


# ---------------------- 3. 旅行周期预测视图 ----------------------
# 3.1 预测页面（展示表单）
def travel_prediction(request):
    """预测页面：展示预测表单"""
    try:
        return render(request, 'prediction.html')
    except Exception as e:
        logger.error(f"预测页面加载失败: {str(e)}", exc_info=True)
        return render(request, 'error.html', {"error_msg": "预测页面加载失败，请稍后重试"}, status=500)


# 3.2 预测接口（处理AJAX请求）
@csrf_exempt
def predict_api(request):
    """预测接口：处理AJAX POST请求，返回旅行周期预测结果"""
    # 异常1：请求方法错误
    if request.method != 'POST':
        logger.warning(f"预测接口 - 非POST请求: {request.method}")
        return JsonResponse({
            "status": "error",
            "message": "仅支持POST请求"
        }, status=405)

    try:
        # 1. 获取输入参数（3个核心参数）
        age_str = request.POST.get('traveler_age')
        acc_cost_str = request.POST.get('accommodation_cost')
        trans_cost_str = request.POST.get('transportation_cost')

        # 异常2：参数缺失
        if not all([age_str, acc_cost_str, trans_cost_str]):
            raise ValueError("缺失必要参数：age/acc_cost/trans_cost")

        # 异常3：参数类型转换失败
        try:
            age = float(age_str)
            acc_cost = float(acc_cost_str)
            trans_cost = float(trans_cost_str)
        except ValueError:
            raise ValueError("参数格式错误：age/acc_cost/trans_cost必须为数字")

        # 异常4：参数值无效（负数/不合理值）
        if age <= 0 or age > 120:
            raise ValueError("年龄必须为0-120之间的正数")
        if acc_cost < 0 or trans_cost < 0:
            raise ValueError("住宿/交通费用不能为负数")

        # 2. 加载或训练模型
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/model')
        model_path = os.path.join(model_dir, 'travel_model.pkl')
        mae_path = os.path.join(model_dir, 'model_mae.pkl')

        if not os.path.exists(model_path) or not os.path.exists(mae_path):
            # 训练模型（首次运行时）
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/cleaned_travel_data.csv')

            # 检查数据文件
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"训练数据文件不存在：{data_path}")

            # 读取训练数据
            try:
                df = pd.read_csv(data_path)
            except Exception as e:
                raise ValueError(f"读取训练数据失败：{str(e)}")

            # 检查训练数据列
            required_cols = ['Traveler age', 'Accommodation cost', 'Transportation cost', 'Duration (days)']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"训练数据缺失列：{', '.join(missing_cols)}")

            # 处理训练数据空值
            df = df.dropna(subset=required_cols)
            if df.empty:
                raise ValueError("训练数据无有效记录（空值过滤后）")

            # 拆分训练/测试集
            x = df[['Traveler age', 'Accommodation cost', 'Transportation cost']]
            y = df['Duration (days)']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # 训练线性回归模型
            model = LinearRegression()
            model.fit(x_train, y_train)

            # 计算MAE（用于预测区间）
            y_pred = model.predict(x_test)
            mae = mean_absolute_error(y_test, y_pred)

            # 保存模型和MAE（确保目录存在）
            os.makedirs(model_dir, exist_ok=True)
            try:
                joblib.dump(model, model_path)
                joblib.dump(mae, mae_path)
            except PermissionError:
                raise PermissionError(f"无权限保存模型到：{model_dir}")
            logger.info("模型训练完成并保存！")

        # 加载模型和MAE
        try:
            model = joblib.load(model_path)
            mae = joblib.load(mae_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件缺失：{model_path} 或 {mae_path}")
        except Exception as e:
            raise ValueError(f"加载模型失败：{str(e)}")

        # 3. 预测计算
        input_data = np.array([[age, acc_cost, trans_cost]])
        try:
            pred_duration = model.predict(input_data)[0]
        except Exception as e:
            raise ValueError(f"模型预测失败：{str(e)}")
        pred_duration = round(pred_duration, 1)

        # 计算预测区间（pred ± mae）
        lower_bound = round(pred_duration - mae, 1)
        upper_bound = round(pred_duration + mae, 1)
        lower_bound = max(lower_bound, 1)  # 确保至少1天

        # 4. 结果解读
        analysis = ""
        if acc_cost > 1500:
            analysis = "住宿费用较高，旅行周期偏短（高住宿成本可能压缩旅行时长）"
        elif trans_cost > 1000:
            analysis = "交通费用较高，旅行周期偏短（长途交通可能减少停留时间）"
        elif age > 40:
            analysis = "年龄较大，旅行周期偏长（中老年旅行者节奏较慢）"
        else:
            analysis = "旅行周期合理，符合同类旅行者的平均水平"

        # 5. 返回结果
        return JsonResponse({
            "status": "success",
            "pred_duration": pred_duration,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "analysis": analysis,
            "confidence": "95%"
        })

    # 细分异常处理
    except ValueError as e:
        logger.error(f"预测接口 - 数值错误: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=400)
    except FileNotFoundError as e:
        logger.error(f"预测接口 - 文件缺失: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=404)
    except PermissionError as e:
        logger.error(f"预测接口 - 权限不足: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=403)
    except KeyError as e:
        logger.error(f"预测接口 - 列缺失: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=400)
    except Exception as e:
        logger.error(f"预测接口 - 未知错误: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": "预测服务异常，请联系管理员"
        }, status=500)


# ---------------------- 4. 旅行费用计算器视图 ----------------------
def cost_calculator(request):
    """费用计算器视图：根据时长/日均住宿/总交通计算旅行费用"""
    try:
        if request.method == 'POST':
            # 1. 获取输入参数
            duration_str = request.POST.get('duration')
            acc_cost_per_day_str = request.POST.get('acc_cost')
            trans_cost_total_str = request.POST.get('trans_cost')

            # 异常1：参数缺失
            if not all([duration_str, acc_cost_per_day_str, trans_cost_total_str]):
                raise ValueError("缺失必要参数：旅行时长/日均住宿费用/总交通费用")

            # 异常2：参数类型转换失败
            try:
                duration = float(duration_str)
                acc_cost_per_day = float(acc_cost_per_day_str)
                trans_cost_total = float(trans_cost_total_str)
            except ValueError:
                raise ValueError("参数格式错误：所有参数必须为数字")

            # 异常3：参数值无效
            if duration <= 0 or duration > 365:
                raise ValueError("旅行时长必须为0-365之间的正数")
            if acc_cost_per_day < 0 or trans_cost_total < 0:
                raise ValueError("住宿/交通费用不能为负数")

            # 2. 计算费用
            total_acc_cost = acc_cost_per_day * duration  # 总住宿费用
            total_cost = total_acc_cost + trans_cost_total  # 总预算
            daily_cost = total_cost / duration  # 日均预算

            # 3. 计算费用占比（处理除零错误）
            acc_ratio = (total_acc_cost / total_cost) * 100 if total_cost != 0 else 0
            trans_ratio = (trans_cost_total / total_cost) * 100 if total_cost != 0 else 0

            # 4. 优化建议
            suggestion = ""
            if acc_ratio > 60:
                suggestion = "住宿费用占比过高（>60%），建议选择性价比更高的住宿（如民宿、青旅），可降低10-20%成本"
            elif trans_ratio > 50:
                suggestion = "交通费用占比过高（>50%），建议提前预订机票/火车票，或选择更经济的交通方式（如高铁替代飞机）"
            elif daily_cost < 200:
                suggestion = "日均预算较低（<200元），建议提前规划行程，避免临时消费超支"
            else:
                suggestion = "费用分配合理，符合中等旅行预算水平，可按此计划出行"

            # 5. 返回结果
            context = {
                "total_cost": round(total_cost, 2),
                "daily_cost": round(daily_cost, 2),
                "acc_ratio": round(acc_ratio, 1),
                "trans_ratio": round(trans_ratio, 1),
                "suggestion": suggestion,
                # 回显输入参数
                "duration": duration,
                "acc_cost": acc_cost_per_day,
                "trans_cost": trans_cost_total
            }
            return render(request, 'cost_calculator.html', context)

        # GET请求：展示计算器页面
        return render(request, 'cost_calculator.html')

    # 细分异常处理
    except ValueError as e:
        logger.error(f"费用计算器 - 数值错误: {str(e)}", exc_info=True)
        return render(request, 'cost_calculator.html', {"error": str(e)}, status=400)
    except Exception as e:
        logger.error(f"费用计算器 - 未知错误: {str(e)}", exc_info=True)
        return render(request, 'cost_calculator.html', {"error": "费用计算异常，请联系管理员"}, status=500)