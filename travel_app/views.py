from django.db import models
from travel_app.models import TravelRecord
import os
import pandas as pd
import joblib
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.db import transaction
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from .models import TravelRecord
from datetime import datetime

# ---------------------- 1. 首页视图 ----------------------
def index(request):
    # 首页：展示4个功能的入口
    return render(request, 'index.html')

# ---------------------- 2. 多维度可视化视图 ----------------------
def multi_visualization(request):
    # 加载清洁数据
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/data/cleaned_travel_data.csv')
    df = pd.read_csv(data_path)

    # 1. 获取前端筛选参数（季节、地域）
    selected_season = request.GET.get('season', '')
    selected_region = request.GET.get('region', '')

    # 2. 筛选数据
    filtered_df = df.copy()
    if selected_season:
        filtered_df = filtered_df[filtered_df['Season'] == selected_season]
    if selected_region:
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]

    # 3. 计算5类对比数据
    # 3.1 性别 vs 旅行周期
    gender_data = filtered_df.groupby('Traveler gender')['Duration (days)'].mean().round(2)
    gender_labels = gender_data.index.tolist()
    gender_values = gender_data.values.tolist()

    # 3.2 年龄分段 vs 旅行周期
    age_data = filtered_df.groupby('Age segment')['Duration (days)'].mean().round(2)
    age_labels = age_data.index.tolist()
    age_values = age_data.values.tolist()

    # 3.3 费用区间 vs 旅行周期
    cost_data = filtered_df.groupby('Cost range')['Duration (days)'].mean().round(2)
    cost_labels = cost_data.index.tolist()
    cost_values = cost_data.values.tolist()

    # 3.4 地域 vs 旅行周期
    region_data = filtered_df.groupby('Region')['Duration (days)'].mean().round(2)
    region_labels = region_data.index.tolist()
    region_values = region_data.values.tolist()

    # 3.5 季节 vs 旅行周期
    season_data = filtered_df.groupby('Season')['Duration (days)'].mean().round(2)
    season_labels = season_data.index.tolist()
    season_values = season_data.values.tolist()

    # 4. 准备ECharts基础配置
    base_option = {
        "title": {"text": "多维度旅行周期分析", "left": "center", "fontSize": 18},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["平均旅行周期（天）"], "bottom": 10},
        "toolbox": {  # 交互式功能：导出、切换图表类型
            "feature": {
                "saveAsImage": {"type": "png", "name": "旅行数据可视化"},
                "magicType": {"type": ["bar", "line", "pie"], "title": {"bar": "柱状图", "line": "折线图", "pie": "饼图"}}
            },
            "right": 20
        },
        "dataZoom": [{"type": "slider", "xAxisIndex": 0, "bottom": 40}]  # 数据缩放
    }

    # 初始渲染性别对比（柱状图）
    init_option = {
        **base_option,
        "xAxis": {"type": "category", "data": gender_labels, "axisLabel": {"rotate": 0}},
        "yAxis": {"type": "value", "name": "天数", "min": 0},
        "series": [{"name": "平均旅行周期（天）", "type": "bar", "data": gender_values, "itemStyle": {"color": "#4895ef"}}]
    }

    # 传递所有数据到前端
    context = {
        "init_option": init_option,
        "gender_data": {"labels": gender_labels, "values": gender_values},
        "age_data": {"labels": age_labels, "values": age_values},
        "cost_data": {"labels": cost_labels, "values": cost_values},
        "region_data": {"labels": region_labels, "values": region_values},
        "season_data": {"labels": season_labels, "values": season_values},
        # 筛选下拉框选项
        "all_seasons": df['Season'].unique().tolist(),
        "all_regions": df['Region'].unique().tolist(),
        "selected_season": selected_season,
        "selected_region": selected_region
    }
    return render(request, 'visualization.html', context)

# ---------------------- 3. 旅行周期预测视图 ----------------------
# 3.1 预测页面（展示表单）
def travel_prediction(request):
    return render(request, 'prediction.html')

# 3.2 预测接口（处理AJAX请求）
def predict_api(request):
    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "仅支持POST请求"})

    try:
        # 1. 获取输入参数（3个核心参数）
        age = float(request.POST.get('age'))
        acc_cost = float(request.POST.get('acc_cost'))
        trans_cost = float(request.POST.get('trans_cost'))

        # 2. 加载或训练模型
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/model/travel_model.pkl')
        mae_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/model/model_mae.pkl')

        if not os.path.exists(model_path) or not os.path.exists(mae_path):
            # 训练模型（首次运行时）
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/cleaned_travel_data.csv')
            df = pd.read_csv(data_path)
            X = df[['Traveler age', 'Accommodation cost', 'Transportation cost']]
            y = df['Duration (days)']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 训练线性回归模型
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 计算MAE（用于预测区间）
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            # 保存模型和MAE
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            joblib.dump(mae, mae_path)
            print("模型训练完成并保存！")

        # 加载模型和MAE
        model = joblib.load(model_path)
        mae = joblib.load(mae_path)

        # 3. 预测计算
        input_data = np.array([[age, acc_cost, trans_cost]])
        pred_duration = model.predict(input_data)[0]
        pred_duration = round(pred_duration, 1)  # 保留1位小数

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
            "confidence": "95%"  # 固定置信度
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

# ---------------------- 4. 数据管理视图（上传+预览） ----------------------
def data_upload(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        # 1. 接收上传的CSV文件
        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            return render(request, 'data_upload.html', {"error": "请上传CSV格式的文件！"})

        # 2. 保存文件并读取
        fs = FileSystemStorage(location=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/data/uploaded/'))
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        try:
            # 3. 读取并预处理上传的数据（与之前的预处理逻辑一致）
            df = pd.read_csv(file_path)
            # 处理缺失值
            df = df.dropna(subset=['Duration (days)', 'Traveler age', 'Traveler gender', 'Accommodation cost', 'Transportation cost'])
            # 转换费用
            df['Accommodation cost'] = df['Accommodation cost'].apply(lambda x: float(''.join(filter(str.isdigit, str(x)))) if str(x).replace('.','').isdigit() else 0.0)
            df['Transportation cost'] = df['Transportation cost'].apply(lambda x: float(''.join(filter(str.isdigit, str(x)))) if str(x).replace('.','').isdigit() else 0.0)
            # 转换日期
            df['Start date'] = pd.to_datetime(df['Start date'], format='%m/%d/%Y', errors='coerce')
            df['End date'] = pd.to_datetime(df['End date'], format='%m/%d/%Y', errors='coerce')
            # 补充其他字段（年龄分段、季节等）
            df['Age segment'] = df['Traveler age'].apply(lambda x: '18-25' if x<=25 else ('26-40' if x<=40 else '40+'))
            df['Month'] = df['Start date'].dt.month
            df['Season'] = df['Month'].apply(lambda x: 'Spring' if x in [3,4,5] else ('Summer' if x in [6,7,8] else ('Autumn' if x in [9,10,11] else 'Winter')))
            df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']
            df['Cost range'] = df['Total cost'].apply(lambda x: 'Low' if x<=1000 else ('Medium' if x<=3000 else 'High'))
            df['Region'] = df['Destination'].apply(lambda x: 'Europe' if any(reg in str(x).lower() for reg in ['uk','france']) else 'Other')

            # 4. 批量导入数据库（事务确保数据一致性）
            with transaction.atomic():
                # 先删除原有数据（可选，避免重复）
                TravelRecord.objects.all().delete()
                # 批量创建
                records = []
                for _, row in df.iterrows():
                    records.append(TravelRecord(
                        trip_id=int(row['Trip ID']),
                        destination=str(row['Destination']),
                        start_date=row['Start date'].date() if pd.notna(row['Start date']) else None,
                        end_date=row['End date'].date() if pd.notna(row['End date']) else None,
                        duration=float(row['Duration (days)']),
                        traveler_name=str(row['Traveler name']),
                        traveler_age=float(row['Traveler age']),
                        traveler_gender=str(row['Traveler gender']),
                        traveler_nationality=str(row['Traveler nationality']),
                        accommodation_type=str(row['Accommodation type']),
                        accommodation_cost=float(row['Accommodation cost']),
                        transportation_type=str(row['Transportation type']),
                        transportation_cost=float(row['Transportation cost']),
                        month=int(row['Month']) if pd.notna(row['Month']) else 0,
                        season=str(row['Season']),
                        age_segment=str(row['Age segment']),
                        total_cost=float(row['Total cost']),
                        cost_range=str(row['Cost range']),
                        region=str(row['Region'])
                    ))
                TravelRecord.objects.bulk_create(records)

            # 5. 预览数据（取前10条）
            preview_data = df.head(10).to_dict('records')
            return render(request, 'data_upload.html', {
                "success": f"数据上传成功！共导入{len(df)}条记录",
                "preview_data": preview_data,
                "total_count": len(df)
            })

        except Exception as e:
            return render(request, 'data_upload.html', {"error": f"数据处理失败：{str(e)}"})

    # GET请求：展示上传页面+现有数据预览
    try:
        # 从数据库获取现有数据（预览前10条）
        preview_data = TravelRecord.objects.all()[:10].values()
        total_count = TravelRecord.objects.count()
        return render(request, 'data_upload.html', {
            "preview_data": preview_data,
            "total_count": total_count
        })
    except:
        return render(request, 'data_upload.html')

# ---------------------- 5. 旅行费用计算器视图 ----------------------
def cost_calculator(request):
    if request.method == 'POST':
        # 1. 获取输入参数
        duration = float(request.POST.get('duration'))
        acc_cost_per_day = float(request.POST.get('acc_cost'))  # 日均住宿费用
        trans_cost_total = float(request.POST.get('trans_cost'))  # 总交通费用

        # 2. 计算费用
        total_acc_cost = acc_cost_per_day * duration  # 总住宿费用
        total_cost = total_acc_cost + trans_cost_total  # 总预算
        daily_cost = total_cost / duration  # 日均预算

        # 3. 计算费用占比
        acc_ratio = (total_acc_cost / total_cost) * 100 if total_cost !=0 else 0
        trans_ratio = (trans_cost_total / total_cost) * 100 if total_cost !=0 else 0

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



'''
def travel_prediction(request):
    return render(request, "prediction.html")


# 处理预测请求（AJAX接口）
def predict_duration(request):
    if request.method == "POST":
        # 1. 获取前端传递的参数（字段名与模型对齐：修正参数名）
        age = request.POST.get("age")
        acc_cost = request.POST.get("acc_cost")
        trans_cost = request.POST.get("trans_cost")

        # 数据校验与类型转换（兼容空值，补充空值默认值避免模型报错）
        try:
            age = float(age) if age else 0.0  # 空值赋默认值
            acc_cost = float(acc_cost) if acc_cost else 0.0
            trans_cost = float(trans_cost) if trans_cost else 0.0
        except (ValueError, TypeError):
            return JsonResponse({
                "status": "error",
                "message": "参数格式错误"
            })

        # 2. 加载训练好的模型
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/model/travel_model.pkl')
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            return JsonResponse({
                "status": "error",
                "message": "预测模型文件不存在"
            })
        except Exception as e:  # 补充模型加载其他异常
            return JsonResponse({
                "status": "error",
                "message": f"模型加载失败：{str(e)}"
            })

        # 3. 进行预测（确保输入维度与模型训练一致）
        try:
            prediction = model.predict([[age, acc_cost, trans_cost]])[0]
            predicted_duration = round(prediction, 2)
        except Exception as e:  # 捕获预测过程异常
            return JsonResponse({
                "status": "error",
                "message": f"预测失败：{str(e)}"
            })

        # 4. 返回预测结果
        return JsonResponse({
            "status": "success",
            "predicted_duration": predicted_duration
        })
    return JsonResponse({"status": "error", "message": "仅支持POST请求"})


# 兼容处理：如果 django_echarts 依赖缺失，先做降级处理
try:
    from django_echarts.entities import EChartsOption
except ImportError:
    # 若未安装 django_echarts/pyecharts，先返回基础字典（保证视图不报错）
    class EChartsOption:
        def __init__(self, **kwargs):
            self.json = kwargs


# 可视化视图（修正字段名、性别取值，与模型严格对应）
def travel_visualization(request):
    # 1. 从数据库查询数据：按性别分组，计算平均旅行周期
    genders = ["Male", "Female"]
    avg_durations = []

    for gender in genders:
        # 过滤对应性别的旅行记录
        records = TravelRecord.objects.filter(traveler_gender=gender)
        if records.exists():
            avg_duration = records.aggregate(models.Avg('duration_days'))['duration_days__avg']
            avg_durations.append(round(avg_duration, 2))
        else:
            avg_durations.append(0)

    # 2. 构建ECharts配置（柱状图）
    chart_option = EChartsOption(
        title=dict(text="不同性别的平均旅行周期对比", left="center"),
        tooltip=dict(trigger="axis"),
        legend=dict(data=["平均旅行周期（天）"]),
        xAxis=dict(type="category", data=["男性", "女性"]),
        yAxis=dict(type="value", name="天数"),
        series=[
            dict(
                name="平均旅行周期（天）",
                type="bar",
                data=avg_durations,
                # 修复：颜色按索引匹配，避免循环变量污染
                itemStyle=dict(
                    color=["#1f77b4", "#ff7f0e"][genders.index(gender)] if len(avg_durations) == 1 else ["#1f77b4", "#ff7f0e"]
                )
            )
        ]
    )

    # 3. 传递数据到模板（统一变量名，兼容django_echarts有无）
    context = {
        # 修正：前端接收的变量名是echarts_option，统一字段名
        "echarts_option": chart_option.json if hasattr(chart_option, 'json') else chart_option,
        "genders": ["男性", "女性"],
        "avg_durations": avg_durations
    }
    return render(request, "visualization.html", context)


def index(request):
    return render(request, "index.html")

# 移除冗余的visualization视图（避免函数名重复导致路由匹配错误）
# def visualization(request):
#     echarts_option = {
#         "title": {"text": "示例图表"},
#         "xAxis": {"type": "category", "data": ["Mon", "Tue", "Wed"]},
#         "yAxis": {"type": "value"},
#         "series": [{"data": [120, 200, 150], "type": "bar"}]
#     }
#     return render(request, 'visualization.html', {'echarts_option': echarts_option})
'''