"""
URL configuration for travel_analysis project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

'''
from django.contrib import admin
from django.urls import path
from travel_app.views import travel_visualization  # 导入视图函数
from travel_app.views import travel_prediction  # 导入视图函数
from travel_app.views import predict_duration  # 导入视图函数
from travel_app.views import index
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name="index"),
    path('visualization/', travel_visualization, name="visualization"),
    path('prediction/', travel_prediction, name="prediction"),  # 预测页面路由
    path('predict/', predict_duration, name="predict"),  # 预测接口路由
]
'''

from django.contrib import admin
from django.urls import path
from travel_app import views  # 导入应用的视图函数

urlpatterns = [
    path('admin/', admin.site.urls),  # Django后台（可选）
    path('', views.index, name='index'),  # 首页（入口）
    path('visualization/', views.multi_visualization, name='visualization'),  # 多维度可视化
    path('prediction/', views.travel_prediction, name='prediction'),  # 旅行周期预测页面
    path('predict-api/', views.predict_api, name='predict_api'),  # 预测接口
    path('cost-calculator/', views.cost_calculator, name='cost_calculator'),  # 费用计算器
]