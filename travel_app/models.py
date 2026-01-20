from django.db import models
from django.core.validators import MinValueValidator
import re

class TravelRecord(models.Model):
    trip_id = models.IntegerField(verbose_name="旅行ID")
    destination = models.CharField(max_length=100, verbose_name="目的地")
    start_date = models.DateField(verbose_name="开始日期")
    end_date = models.DateField(verbose_name="结束日期")
    duration = models.FloatField(verbose_name="旅行周期（天）")
    traveler_name = models.CharField(max_length=50, verbose_name="旅行者姓名")
    traveler_age = models.FloatField(verbose_name="旅行者年龄")
    traveler_gender = models.CharField(max_length=10, verbose_name="旅行者性别")
    traveler_nationality = models.CharField(max_length=30, verbose_name="旅行者国籍")
    accommodation_type = models.CharField(max_length=30, verbose_name="住宿类型")
    accommodation_cost = models.FloatField(verbose_name="住宿费用")
    transportation_type = models.CharField(max_length=30, verbose_name="交通类型")
    transportation_cost = models.FloatField(verbose_name="交通费用")
    month = models.IntegerField(verbose_name="出行月份")
    season = models.CharField(max_length=10, verbose_name="出行季节")
    age_segment = models.CharField(max_length=10, verbose_name="年龄分段")
    total_cost = models.FloatField(verbose_name="总费用")
    cost_range = models.CharField(max_length=10, verbose_name="费用区间")
    region = models.CharField(max_length=20, verbose_name="地域")

    class Meta:
        verbose_name = "旅行记录"
        verbose_name_plural = "旅行记录"

    def __str__(self):
        return f"{self.destination}-{self.traveler_name}-{self.duration}天"

'''class TravelRecord(models.Model):
    """
    旅行记录数据模型
    对应Travel details dataset.csv文件的所有字段
    """

    # 1. 旅行基本信息字段
    trip_id = models.IntegerField(
        verbose_name="旅行ID",
        unique=True,
        validators=[MinValueValidator(1)],
        help_text="旅行记录的唯一标识ID"
    )
    destination = models.CharField(
        max_length=100,
        verbose_name="目的地",
        null=True,
        blank=True,
        help_text="格式：城市, 国家（如London, UK）"
    )
    start_date = models.DateField(
        verbose_name="开始日期",
        null=True,
        blank=True,
        help_text="旅行开始日期，格式：YYYY-MM-DD"
    )
    end_date = models.DateField(
        verbose_name="结束日期",
        null=True,
        blank=True,
        help_text="旅行结束日期，格式：YYYY-MM-DD"
    )
    duration_days = models.FloatField(
        verbose_name="旅行周期（天）",
        null=True,
        blank=True,
        validators=[MinValueValidator(1)],
        help_text="旅行总天数，预测目标字段"
    )

    # 2. 旅行者信息字段
    traveler_name = models.CharField(
        max_length=100,
        verbose_name="旅行者姓名",
        null=True,
        blank=True,
        help_text="旅行者的全名"
    )
    traveler_age = models.FloatField(
        verbose_name="旅行者年龄",
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="旅行者的年龄，范围：20-60岁"
    )

    # 性别选择项，基于数据中的实际值
    GENDER_CHOICES = [
        ('Male', '男性'),
        ('Female', '女性'),
    ]
    traveler_gender = models.CharField(
        max_length=10,
        verbose_name="旅行者性别",
        choices=GENDER_CHOICES,
        null=True,
        blank=True,
        help_text="旅行者的性别"
    )
    traveler_nationality = models.CharField(
        max_length=50,
        verbose_name="旅行者国籍",
        null=True,
        blank=True,
        help_text="旅行者的国籍（如American, Chinese）"
    )

    # 住宿类型选择项（修正：Hostel对应的中文去掉多余空格）
    ACCOMMODATION_CHOICES = [
        ('Hotel', '酒店'),
        ('Airbnb', '民宿'),
        ('Hostel', '青年旅社'),  # 修正：原' hostel'空格错误+翻译优化
        ('Resort', '度假村'),
        ('Villa', '别墅'),
        ('Vacation rental', '度假租赁'),
        ('Riad', '摩洛哥传统住宅'),
        ('Guesthouse', '宾馆'),
    ]
    accommodation_type = models.CharField(
        max_length=50,
        verbose_name="住宿类型",
        choices=ACCOMMODATION_CHOICES,
        null=True,
        blank=True,
        help_text="旅行者选择的住宿类型"
    )

    # 交通类型选择项，基于数据中的实际值（已合并相似项）
    TRANSPORTATION_CHOICES = [
        ('Plane', '飞机'),
        ('Flight', '航班'),  # 与Plane含义相似，保留原始数据
        ('Train', '火车'),
        ('Car rental', '租车'),
        ('Bus', '巴士'),
        ('Airplane', '飞机'),  # 与Plane含义相似，保留原始数据
        ('Car', '汽车'),
        ('Subway', '地铁'),
        ('Ferry', '轮渡'),
    ]
    transportation_type = models.CharField(
        max_length=50,
        verbose_name="交通类型",
        choices=TRANSPORTATION_CHOICES,
        null=True,
        blank=True,
        help_text="旅行者选择的交通方式"
    )
    transportation_cost = models.FloatField(
        verbose_name="交通费用",
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="交通总费用（单位：货币单位）"
    )

    # 4. 记录管理字段
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="创建时间",
        help_text="记录首次创建的时间"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name="更新时间",
        help_text="记录最后更新的时间"
    )

    class Meta:
        verbose_name = "旅行记录"
        verbose_name_plural = "旅行记录"
        ordering = ['-start_date', 'trip_id']  # 按开始日期降序、ID升序排列
        indexes = [
            models.Index(fields=['trip_id']),  # 为唯一ID创建索引
            models.Index(fields=['destination']),  # 为目的地创建索引
            models.Index(fields=['start_date']),  # 为开始日期创建索引
            models.Index(fields=['traveler_gender', 'traveler_age']),  # 为性别和年龄组合创建索引
        ]

    def __str__(self):
        """模型实例的字符串表示"""
        return f"旅行{self.trip_id}: {self.destination} ({self.duration_days}天)"

    def clean_cost_field(self, cost_value):
        """
        清理费用字段的辅助方法
        处理包含货币符号（如$）和空格的费用值
        """
        if not cost_value:
            return None

        # 如果是字符串类型，清理货币符号和空格
        if isinstance(cost_value, str):
            # 移除所有非数字和小数点的字符
            cleaned = re.sub(r'[^\d.]', '', cost_value.strip())
            try:
                return float(cleaned)
            except ValueError:
                return None
        return float(cost_value)

    def save(self, *args, **kwargs):
        """重写保存方法，自动清理费用字段"""
        # 清理住宿费用
        if self.accommodation_cost is not None:
            self.accommodation_cost = self.clean_cost_field(self.accommodation_cost)

        # 清理交通费用
        if self.transportation_cost is not None:
            self.transportation_cost = self.clean_cost_field(self.transportation_cost)

        super().save(*args, **kwargs)

    def get_travel_period(self):
        """计算并返回旅行时间段的字符串表示"""
        if self.start_date and self.end_date:
            return f"{self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}"
        return "日期未完整记录"

    def get_total_cost(self):
        """计算并返回总费用"""
        total = 0
        if self.accommodation_cost:
            total += self.accommodation_cost
        if self.transportation_cost:
            total += self.transportation_cost
        return round(total, 2)
'''