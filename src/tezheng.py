# -*- coding: utf-8 -*-
"""
tezheng.py —— 预处理与模型的组合（Pipeline）。

本实验最核心的一条工程原则：
    预处理和模型必须在同一个 Pipeline 里，一起 fit、一起 dump。
    分成两步保存的话，训练时用了 StandardScaler，推理时忘了用，
    结果不报错但预测全错——这就是"训练/推理预处理不一致"。

wine 是纯数值数据，没有类别列，所以实际用的是：
    SimpleImputer(median) -> StandardScaler -> 分类器
下面那个 ColumnTransformer 版本是为了在报告里展示通用写法，
万一以后换成有类别列的数据集可以直接改着用。
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 模型标识。写进元数据，界面按这个显示"当前加载的是哪一版"
MOXING_A_MING = "logistic_regression"
MOXING_B_MING = "random_forest"


def goujian_yuchuli(tezheng_ming):
    """
    数值特征的预处理器。

    SimpleImputer 放在前面是防御性的：wine 没有缺失值，
    但真实数据里经常有。留着它，换成别的数据集时不用改结构。
    """
    return Pipeline([
        ("buchong", SimpleImputer(strategy="median")),
        ("biaozhunhua", StandardScaler()),
    ])


def goujian_yuchuli_tongyong(shuzhi_lie, leibie_lie=None):
    """
    通用版预处理器：数值列走中位数填充+标准化，类别列走众数填充+OneHot。

    ★ OneHotEncoder 必须写 handle_unknown="ignore"。
      否则推理时遇到训练阶段没见过的类别会直接抛异常，
      整个应用挂掉。ignore 之后是输出全 0 向量，不会崩。
      代价是"未知类别"被静默当成了一种特征缺失，所以界面层要另外提示。

    本实验用的是 wine，没有类别列，这个函数实际没被调用；
    留着是为了说明"如果数据有类别列该怎么写"，不是凑复杂度。
    """
    shuzhi_guan = Pipeline([
        ("buchong", SimpleImputer(strategy="median")),
        ("biaozhunhua", StandardScaler()),
    ])
    guan_list = [("shuzhi", shuzhi_guan, list(shuzhi_lie))]

    if leibie_lie:
        leibie_guan = Pipeline([
            ("buchong", SimpleImputer(strategy="most_frequent")),
            ("duhuo", OneHotEncoder(handle_unknown="ignore")),
        ])
        guan_list.append(("leibie", leibie_guan, list(leibie_lie)))

    return ColumnTransformer(guan_list)


def goujian_moxing_a(zhongzi=42):
    """
    算法 A：逻辑回归。

    为什么选它当基线：线性模型、训练快、系数可解释，
    是分类任务最经典的"先跑一个看看"的选择。
    max_iter 调到 2000 是因为默认 100 在这份数据上会收敛警告。
    """
    return LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        random_state=zhongzi,
    )


def goujian_moxing_b(zhongzi=42):
    """
    算法 B：随机森林。

    选它是为了和 A 形成对比：A 是线性、B 是非线性；
    A 需要标准化、B 理论上不需要（但 Pipeline 里统一做了也不影响）。
    两个都跑一遍才能回答"这份数据到底线性可分吗"。
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=zhongzi,
        n_jobs=-1,
    )


def goujian_pipeline(tezheng_ming, moxing):
    """把预处理器和分类器串成一个完整 Pipeline。这是保存和加载的单元。"""
    return Pipeline([
        ("yuchuli", goujian_yuchuli(tezheng_ming)),
        ("moxing", moxing),
    ])
