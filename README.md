### 旅行期间分析可视化系统

#### 项目详情

##### 项目简介

旅行期间分析可视化系统是一款基于 Django 和 ECharts 技术栈构建的 B/S 架构 Web 应用，遵循 MVT（Model-View-Template）设计原则。系统聚焦旅行数据的全生命周期处理，涵盖数据导入、预处理、模型训练、多维度可视化分析、旅行周期预测及费用计算等核心功能，为用户提供数据驱动的旅行决策支持，适用于旅行爱好者、行业分析师等各类用户。

##### 技术栈

后端框架：Django （Python Web 开发框架，提供完整 MVT 架构支持）

数据处理：Pandas、NumPy （数据清洗、转换与分析）

机器学习：Scikit-learn （线性回归、随机森林等模型训练与预测）

可视化工具：ECharts （交互式数据可视化图表生成）

数据库：SQLite （轻量级关系型数据库，存储旅行数据与预测记录）

前端技术：HTML、CSS、JavaScript （页面构建与交互逻辑）

其他依赖：Joblib（模型持久化）、Django 内置工具（URL 路由、视图函数等）

##### 项目结构

travel\_analysis/

├── travel\_analysis/          # 项目配置目录

│   ├── settings.py           # 项目核心配置（数据库、静态资源、应用注册等）

│   ├── urls.py               # 项目总路由配置

│   └── wsgi.py               # WSGI 应用入口

├── travel\_app/               # 核心应用目录

│   ├── models.py             # 数据模型定义（TravelRecord 表结构）

│   ├── views.py              # 视图函数（业务逻辑处理）

│   ├── urls.py               # 应用路由配置

│   ├── import\_data.py        # CSV 数据导入脚本

│   ├── data\_preprocess.py    # 数据预处理脚本

│   ├── train\_model.py        # 模型训练脚本

│   └── model\_mae.py          # 模型 MAE 评估脚本

├── data/                     # 数据存储目录

│   ├── Travel details dataset.csv  # 原始旅行数据 CSV 文件

│   └── cleaned\_travel\_data.csv     # 预处理后清洁数据

├── static/                   # 静态资源目录

│   ├── model/                # 训练好的模型文件（travel\_model.pkl、model\_mae.pkl）

│   └── echarts/              # ECharts 可视化资源

├── templates/                # 页面模板目录

│   ├── index.html            # 系统首页

│   ├── visualization.html    # 多维度可视化页面

│   ├── prediction.html       # 旅行周期预测页面

│   ├── cost\_calculator.html  # 旅行费用计算器页面

│   └── error.html            # 错误提示页面

├── db.sqlite3                # SQLite 数据库文件

├── manage.py                 # Django 项目管理脚本

└── requirements.txt          # 项目依赖库清单

##### 主要功能

###### 1\. 数据导入与预处理

数据导入：支持 CSV 格式旅行数据文件批量导入 SQLite 数据库，涵盖旅行 ID、目的地、旅行者信息、费用、周期等核心字段。

数据预处理：通过脚本自动完成缺失值处理（删除 / 填充）、费用字段格式标准化（移除货币符号 / 空格）、日期解析、年龄分段、地域划分、季节提取等操作，生成清洁数据用于分析与建模。

###### 2\. 多维度可视化分析

支持 5 类核心对比分析：性别 vs 旅行周期、年龄分段 vs 旅行周期、费用区间 vs 旅行周期、地域 vs 旅行周期、季节 vs 旅行周期。

交互式操作：提供季节、地域筛选功能，支持图表类型切换（柱状图 / 折线图 / 饼图）、缩放、导出图片等，直观呈现数据规律。

###### 3\. 旅行周期预测

基于线性回归 / 随机森林模型，输入旅行者年龄、住宿费用、交通费用 3 个关键参数，输出预测旅行周期及 95% 置信区间。

结果解读：根据输入参数自动生成分析建议（如高交通费用对周期的影响），并保存预测历史记录供回溯查看。

###### 4\. 旅行费用计算器

输入旅行时长、日均住宿费用、总交通费用，自动计算总预算、日均预算、住宿 / 交通费用占比。

优化建议：根据费用占比给出合理性评估与成本优化建议（如住宿费用占比过高时推荐高性价比住宿）。

###### 5\. 系统辅助功能

完善的错误处理：针对文件缺失、参数错误、权限不足等场景提供明确提示。

跨浏览器兼容：支持主流浏览器（Chrome、Firefox、Edge 等）正常访问。

#### 安装与运行

##### 环境要求

Python 版本：3.7+

依赖库：见 requirements.txt（Django、pandas、numpy、scikit-learn、joblib 等）

##### 安装步骤

###### 1\.克隆 / 下载项目

将项目文件解压至本地目录，或通过版本控制工具克隆：

git clone <https://github.com/huaiyu-sy/XMUT-travel\_analysis.git>

cd travel\_analysis

###### 2\.安装依赖库

建议使用虚拟环境隔离依赖，执行以下命令安装所需库：

\# 创建虚拟环境（可选）

python -m venv venv

\# 激活虚拟环境（Windows）

venv\\Scripts\\activate

\# 激活虚拟环境（Mac/Linux）

source venv/bin/activate

\# 安装依赖

pip install django pandas numpy scikit-learn joblib

###### 3\.配置数据文件

将旅行数据 CSV 文件（命名为 Travel details dataset.csv）放入 data/ 目录，确保包含以下核心字段：

Trip ID、Destination、Start date、End date、Duration (days)、Traveler name、Traveler age、Traveler gender、Traveler nationality、Accommodation type、Accommodation cost、Transportation type、Transportation cost。

执行数据预处理脚本，生成清洁数据：

python travel\_app/data\_preprocess.py

###### 4\.导入数据到数据库

执行数据导入脚本，将 CSV 数据批量写入 SQLite 数据库：

python travel\_app/import\_data.py

###### 5\.训练预测模型

执行模型训练脚本，生成并保存模型文件（默认存储于 static/model/）：

python travel\_app/train\_model.py

##### 运行步骤

###### 1\.启动 Django 开发服务器

python manage.py runserver 8000

（若 8000 端口被占用，可更换端口，如 python manage.py runserver 8001）

###### 2\.访问系统

打开浏览器，输入地址 http://127.0.0.1:8000/，即可进入系统首页，开始使用各项功能。

##### 使用指南

###### 系统首页

提供 4 个功能入口：多维度可视化分析、旅行周期预测、旅行费用计算器、数据导入（后台支持）。

###### 多维度可视化分析

选择季节、地域筛选条件，点击「筛选数据」。

切换顶部标签（性别对比 / 年龄分段对比等）查看不同维度图表。

使用图表工具栏导出图片、切换图表类型或缩放数据。

###### 旅行周期预测

在输入框中填写旅行者年龄（0-120 岁）、住宿费用（非负）、交通费用（非负）。

点击「预测旅行周期」，查看预测结果、置信区间及分析建议。

预测历史记录可在页面下方查看，支持删除单条记录或清空全部。

###### 旅行费用计算器

输入旅行时长（0-365 天）、日均住宿费用（非负）、总交通费用（非负）。

点击「计算费用」，查看总预算、日均预算、费用占比及优化建议。

##### 注意事项

数据文件路径：确保 CSV 原始数据、清洁数据路径与脚本中配置一致，否则会导致数据读取失败。

模型文件：首次运行预测功能时，若未提前训练模型，系统会自动训练并保存模型，耗时稍长，请耐心等待。

参数输入：所有数值型参数需输入合理范围（如年龄 0-120 岁），否则会提示参数错误。

浏览器兼容性：建议使用 Chrome 浏览器以获得最佳交互体验。

##### 问题排查

服务器启动失败：检查端口是否被占用，或 Django 环境配置是否正确。

数据导入失败：检查 CSV 文件格式是否正确、字段是否完整，或数据库权限是否足够。

图表无法加载：检查清洁数据文件是否存在，或 ECharts 静态资源是否正确引入。

预测结果异常：检查模型是否已训练，或输入参数是否在合理范围内。

##### 致谢

本项目基于 Django、ECharts、Scikit-learn 等开源技术构建，感谢相关技术社区的支持与贡献。

