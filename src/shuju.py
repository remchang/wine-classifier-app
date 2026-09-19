# -*- coding: utf-8 -*-
"""
shuju.py —— 数据加载、质量检查与划分。

数据集：sklearn 自带的 wine（load_wine），共 178 个样本、13 个数值特征、3 个类别。
用它的好处是不用联网下载，CPU 上几秒就能跑完，适合课堂。

重要：本模块只负责"拿数据 + 划分"，不碰任何预处理。
      预处理必须放进 Pipeline 里（见 tezheng.py），否则就是数据泄漏。
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 固定随机种子。写死它，任何人任何时候跑出来的划分都一模一样。
SUIJI_ZHONGZI = 42

# 测试集占比。178 * 0.25 ≈ 45 个测试样本，够算混淆矩阵了。
CESHI_ZhanBI = 0.25


class ShujuCuowu(Exception):
    """数据层面的问题（缺列、全空、标签异常）统一抛这个。"""
    pass


def jiazai_shuju():
    """
    加载 wine 数据集。

    返回 (tezheng_biao, biaoqian, leibie_ming)
        tezheng_biao  : DataFrame，178 行 × 13 列
        biaoqian      : Series，值域 {0,1,2}
        leibie_ming   : list[str]，下标即类别编号
    """
    raw = load_wine(as_frame=True)
    tezheng_biao = raw.data.copy()
    biaoqian = raw.target.copy()

    # 列名里有空格和斜杠，统一换成下划线，避免以后当关键字用的时候出问题
    tezheng_biao.columns = [c.replace(" ", "_").replace("/", "_")
                            for c in tezheng_biao.columns]

    leibie_ming = [str(x) for x in raw.target_names]
    return tezheng_biao, biaoqian, leibie_ming


def jiancha_shuju(tezheng_biao, biaoqian):
    """
    基础质量检查。返回一个字典，直接给报告和界面用。

    实验要求"Schema 检查"，所以这里把形状、缺失、重复、标签分布都查一遍。
    """
    wenti = []

    if len(tezheng_biao) != len(biaoqian):
        wenti.append("特征行数与标签行数不一致")

    kong_lie = [c for c in tezheng_biao.columns if tezheng_biao[c].isna().all()]
    if kong_lie:
        wenti.append("存在整列为空的特征：%s" % ", ".join(kong_lie))

    if tezheng_biao.isna().any().any():
        wenti.append("存在缺失值（本数据集理论上不该有）")

    chongfu = int(tezheng_biao.duplicated().sum())

    return {
        "yangben_shu": int(len(tezheng_biao)),
        "tezheng_shu": int(tezheng_biao.shape[1]),
        "leibie_shu": int(biaoqian.nunique()),
        "leibie_fenbu": {int(k): int(v) for k, v in biaoqian.value_counts().sort_index().items()},
        "quexian_zhi": int(tezheng_biao.isna().sum().sum()),
        "chongfu_hang": chongfu,
        "wenti": wenti,
        "tezheng_ming": list(tezheng_biao.columns),
    }


def huafen_shuju(tezheng_biao, biaoqian, zhongzi=SUIJI_ZHONGZI):
    """
    分层划分训练集 / 测试集。

    为什么用 stratify：
        wine 三个类别是 59 / 71 / 48，不算特别不平衡但也不均匀。
        不做分层的话，小样本情况下可能出现某个类在测试集里只有一两个，
        指标波动会很大。分层能保证两边类别比例一致。

    为什么是分层而不是 GroupShuffleSplit：
        wine 的每一行是一个独立的酒样本，不存在"同一个主体有多行"的问题，
        所以不需要分组划分。如果换成"同一用户多条记录"的数据集，
        就必须换成 GroupShuffleSplit，否则会数据泄漏。
    """
    if tezheng_biao is None or len(tezheng_biao) == 0:
        raise ShujuCuowu("特征表是空的，没法划分")

    X_xunlian, X_ceshi, y_xunlian, y_ceshi = train_test_split(
        tezheng_biao,
        biaoqian,
        test_size=CESHI_ZhanBI,
        random_state=zhongzi,
        stratify=biaoqian,       # ★ 关键
    )
    return X_xunlian, X_ceshi, y_xunlian, y_ceshi


def shuju_zhiwen(tezheng_biao, biaoqian):
    """
    给数据算一个 SHA256 指纹，写进模型元数据。

    目的：以后模型表现不对时，能确认"当时用的是不是这份数据"。
    做法是把所有数值按固定精度四舍五入后拼成字符串再哈希——
    直接哈希浮点数的二进制表示在不同平台可能不一致。
    """
    h = hashlib.sha256()
    h.update(np.round(tezheng_biao.to_numpy(dtype=float), 6).tobytes())
    h.update(np.asarray(biaoqian).tobytes())
    h.update(("|".join(tezheng_biao.columns)).encode("utf-8"))
    return h.hexdigest()


def zu_canshu_zi(lie_ming, zhi):
    """
    把界面传进来的字典整理成一条 DataFrame，列顺序严格按训练时的顺序。

    为什么要这么小心：sklearn 的 Pipeline 内部是位置敏感的。
    如果用户传的 dict 顺序和训练时不同，预测结果会静默错掉——
    不报错，但结果是垃圾。所以必须按 lie_ming 重新排一次。
    """
    if not isinstance(zhi, dict):
        raise ShujuCuowu("输入必须是字典")
    que = [c for c in lie_ming if c not in zhi]
    if que:
        raise ShujuCuowu("缺少字段：%s" % ", ".join(que))

    cuo_lie = [c for c in lie_ming if not _ke_zhuan_shuzi(zhi[c])]
    if cuo_lie:
        raise ShujuCuowu("这些字段不是数字：%s" % ", ".join(cuo_lie))

    # ★ 按 lie_ming 的顺序重建，这才是不出错的关键
    return pd.DataFrame([[float(zhi[c]) for c in lie_ming]], columns=list(lie_ming))


def _ke_zhuan_shuzi(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False
