# -*- coding: utf-8 -*-
"""
test_shuju.py —— 数据与划分测试（实验要求 ≥5 条）。

重点验三件事：
  ① Schema 对不对（行数、列数、类别数）
  ② 划分可不可复现（同种子同结果）
  ③ 有没有明显泄漏（训练集和测试集不重叠、分层比例一致）
"""
import pandas as pd
import pytest

from shuju import (
    SUIJI_ZHONGZI,
    ShujuCuowu,
    huafen_shuju,
    jiancha_shuju,
    jiazai_shuju,
    shuju_zhiwen,
    zu_canshu_zi,
)


def test_jiazai_shuju_schema(yuanshi_shuju):
    """内置 wine 应该是 178 行 × 13 列，3 个类别。"""
    tb, bq, lm = yuanshi_shuju
    assert tb.shape == (178, 13)
    assert len(lm) == 3
    assert sorted(bq.unique().tolist()) == [0, 1, 2]


def test_lie_ming_yi_qingli(yuanshi_shuju):
    """列名里的空格和斜杠应该被换成下划线（不然当关键字用会出问题）。"""
    tb, _, _ = yuanshi_shuju
    for c in tb.columns:
        assert " " not in c
        assert "/" not in c


def test_jiancha_shuju_meiyou_wenti(yuanshi_shuju):
    """wine 是干净数据，质量检查不该报问题。"""
    tb, bq, _ = yuanshi_shuju
    zhiliang = jiancha_shuju(tb, bq)
    assert zhiliang["wenti"] == []
    assert zhiliang["quexian_zhi"] == 0
    assert zhiliang["leibie_shu"] == 3
    assert sum(zhiliang["leibie_fenbu"].values()) == zhiliang["yangben_shu"]


def test_huafen_kesu (huafende):
    """同一种子 + 同一份数据，两次划分结果必须一模一样。"""
    X_xl, X_cs, y_xl, y_cs = huafende
    tb, bq, _ = jiazai_shuju()
    X_xl2, X_cs2, y_xl2, y_cs2 = huafen_shuju(tb, bq, SUIJI_ZHONGZI)
    pd.testing.assert_frame_equal(X_xl, X_xl2)
    pd.testing.assert_series_equal(y_cs, y_cs2)


def test_huafen_bili(huafende):
    """测试集占比应该接近 25%。"""
    X_xl, X_cs, _, _ = huafende
    bili = len(X_cs) / (len(X_xl) + len(X_cs))
    assert 0.20 <= bili <= 0.30


def test_huafen_fenceng_youxiao(huafende):
    """分层划分：训练集和测试集的类别比例应该接近。"""
    _, _, y_xl, y_cs = huafende
    p_xl = y_xl.value_counts(normalize=True).sort_index()
    p_cs = y_cs.value_counts(normalize=True).sort_index()
    for k in p_xl.index:
        assert abs(p_xl[k] - p_cs[k]) < 0.08, "类别 %s 比例差太大" % k


def test_xunlian_ceshi_wu_chongdie(huafende):
    """
    无重叠检查。这是防泄漏最基础的一条：
    同一个样本不能既在训练集又在测试集。
    """
    X_xl, X_cs, _, _ = huafende
    zl = set(map(tuple, X_xl.round(9).to_numpy()))
    cs = set(map(tuple, X_cs.round(9).to_numpy()))
    assert len(zl & cs) == 0


def test_kong_biao_baocuo():
    """空数据要明确报错，不能静默返回一个空划分。"""
    with pytest.raises(ShujuCuowu):
        huafen_shuju(pd.DataFrame(), pd.Series(dtype=int))


def test_shuju_zhiwen_wending(yuanshi_shuju):
    """数据指纹同样数据必须一致，改一个数就变。"""
    tb, bq, _ = yuanshi_shuju
    a = shuju_zhiwen(tb, bq)
    b = shuju_zhiwen(tb.copy(), bq.copy())
    assert a == b

    tb2 = tb.copy()
    tb2.iloc[0, 0] = tb2.iloc[0, 0] + 1.0
    assert shuju_zhiwen(tb2, bq) != a


def test_zu_canshu_zi_an_lie_ming_paixu(yuanshi_shuju):
    """
    ★ 关键测试：传进去的字典顺序打乱，输出 DataFrame 的列顺序也必须正确。
    如果这里错了，预测会静默出错——不报错但结果是垃圾。
    """
    tb, _, _ = yuanshi_shuju
    lie = list(tb.columns)
    zheng = {c: 1.0 for c in lie}
    luan = {c: 1.0 for c in reversed(lie)}

    a = zu_canshu_zi(lie, zheng)
    b = zu_canshu_zi(lie, luan)
    assert list(a.columns) == lie
    assert list(b.columns) == lie
    pd.testing.assert_frame_equal(a, b)


def test_zu_canshu_zi_que_ziduan_baocuo(yuanshi_shuju):
    tb, _, _ = yuanshi_shuju
    lie = list(tb.columns)
    zhi = {c: 1.0 for c in lie}
    zhi.pop(lie[3])
    with pytest.raises(ShujuCuowu) as e:
        zu_canshu_zi(lie, zhi)
    assert lie[3] in str(e.value)


def test_zu_canshu_zi_feishuzi_baocuo(yuanshi_shuju):
    tb, _, _ = yuanshi_shuju
    lie = list(tb.columns)
    zhi = {c: 1.0 for c in lie}
    zhi[lie[0]] = "不是数字"
    with pytest.raises(ShujuCuowu):
        zu_canshu_zi(lie, zhi)


def test_zu_canshu_zi_feizidian_baocuo(yuanshi_shuju):
    tb, _, _ = yuanshi_shuju
    with pytest.raises(ShujuCuowu):
        zu_canshu_zi(list(tb.columns), [1, 2, 3])
