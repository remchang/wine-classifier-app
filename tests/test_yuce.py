# -*- coding: utf-8 -*-
"""
test_yuce.py —— 推理接口与应用级输入测试（实验要求：应用测试 ≥6 条）。

覆盖：合法输入、非法输入、边界值、批量输入、坏行隔离、阈值行为。
"""
import pandas as pd
import pytest

from shuju import ShujuCuowu, huafen_shuju, jiazai_shuju
from yuce import MoxingCuowu, jiazai_moxing, pinggu_yuzhi, yuce_dan_tiao, yuce_piliang


@pytest.fixture
def zhengchang_canshu(yuanshi_shuju):
    """一批合法参数（取第一条真实样本，转成字典）。"""
    tb, _, _ = yuanshi_shuju
    return {c: float(tb.iloc[0][c]) for c in tb.columns}


# ==================== 单条预测 ====================

def test_dan_tiao_zhengchang(jiazai_haode, zhengchang_canshu):
    guan, meta = jiazai_haode
    jieguo = yuce_dan_tiao(guan, meta, zhengchang_canshu, 0.0)
    assert jieguo["leibie"] in meta["leibie_ming"]
    assert 0.0 <= jieguo["kexindu"] <= 1.0
    # 概率之和应该是 1
    assert abs(sum(jieguo["gailv"].values()) - 1.0) < 1e-6


def test_dan_tiao_que_ziduan_baocuo(jiazai_haode, zhengchang_canshu):
    guan, meta = jiazai_haode
    huai = dict(zhengchang_canshu)
    huai.pop(list(huai.keys())[2])
    with pytest.raises(ShujuCuowu) as e:
        yuce_dan_tiao(guan, meta, huai, 0.0)
    assert "缺少字段" in str(e.value)


def test_dan_tiao_bianliang_weikong_baocuo(jiazai_haode, zhengchang_canshu):
    """缺失值不能静默通过——wine 没缺失，但接口得挡住。"""
    guan, meta = jiazai_haode
    huai = dict(zhengchang_canshu)
    huai[list(huai.keys())[0]] = None
    with pytest.raises(ShujuCuowu):
        yuce_dan_tiao(guan, meta, huai, 0.0)


def test_dan_tiao_feishuzi_baocuo(jiazai_haode, zhengchang_canshu):
    guan, meta = jiazai_haode
    huai = dict(zhengchang_canshu)
    huai[list(huai.keys())[1]] = "abc"
    with pytest.raises(ShujuCuowu):
        yuce_dan_tiao(guan, meta, huai, 0.0)


def test_dan_tiao_canshu_shunxu_buyingxiang(jiazai_haode, zhengchang_canshu):
    """字典顺序打乱，预测结果必须一样。"""
    guan, meta = jiazai_haode
    a = yuce_dan_tiao(guan, meta, zhengchang_canshu, 0.0)
    luan = dict(reversed(list(zhengchang_canshu.items())))
    b = yuce_dan_tiao(guan, meta, luan, 0.0)
    assert a["leibie"] == b["leibie"]
    assert a["kexindu"] == b["kexindu"]


def test_dan_tiao_bianjie_zhi_bu_bengkui(jiazai_haode, zhengchang_canshu):
    """
    边界值：极小 / 极大。
    这里明确接受"模型会给一个答案"这个现实——
    越界输入不会报错，这是当前版本的已知短板，写在模型卡里了。
    测试只要求它别崩、概率仍然合法。
    """
    guan, meta = jiazai_haode
    for ya in (1e-6, 0.0, 1e4):
        bian = {c: ya for c in zhengchang_canshu}
        jieguo = yuce_dan_tiao(guan, meta, bian, 0.0)
        assert jieguo["leibie"] in meta["leibie_ming"]


# ==================== 阈值（自主功能） ====================

def test_yuzhi_hen_gao_jiao_ren_he(jiazai_haode, zhengchang_canshu):
    """阈值设成 1.01（实际上不可能达到）时，应该一律标为不可信。"""
    guan, meta = jiazai_haode
    jieguo = yuce_dan_tiao(guan, meta, zhengchang_canshu, 1.01)
    assert jieguo["keyi_xinren"] is False
    assert "复核" in jieguo["tishi"]


def test_yuzhi_hen_di_dou_keyi(jiazai_haode, zhengchang_canshu):
    guan, meta = jiazai_haode
    jieguo = yuce_dan_tiao(guan, meta, zhengchang_canshu, 0.0)
    assert jieguo["keyi_xinren"] is True


def test_pinggu_yuzhi_qushi_heli(jiazai_haode):
    """
    ★ 自主功能的量化证据。
    阈值升高时：覆盖率必须单调不增（可以不变），转人工数必须单调不减。
    如果这个趋势反了，说明阈值逻辑写反了。
    """
    guan, meta = jiazai_haode
    tb, bq, _ = jiazai_shuju()
    _, X_cs, _, y_cs = huafen_shuju(tb, bq)

    jieguo = pinggu_yuzhi(guan, meta, X_cs, y_cs, [0.0, 0.4, 0.6, 0.8, 0.9])
    fugai = [j["fugai_lv"] for j in jieguo]
    for i in range(1, len(fugai)):
        assert fugai[i] <= fugai[i - 1] + 1e-9, "覆盖率不该随阈值上升"

    # 阈值 0 时覆盖率必须是 100%
    assert jieguo[0]["fugai_lv"] == 1.0
    # 阈值 0 时的自动准确率 = 整体 accuracy
    assert jieguo[0]["zidong_zhunque_lv"] == meta["ceshi_ji"]["accuracy"]


# ==================== 批量预测（自主功能） ====================

def test_piliang_zhengchang(jiazai_haode, yuanshi_shuju):
    guan, meta = jiazai_haode
    tb, _, _ = yuanshi_shuju
    biao = tb.head(10)
    jieguo, cuowu, tongji = yuce_piliang(guan, meta, biao, 0.0)
    assert len(jieguo) == 10
    assert cuowu == []
    assert tongji["chenggong"] == 10


def test_piliang_huai_hang_bu_yingxiang_zhengpi(jiazai_haode, yuanshi_shuju):
    """
    ★ 实验明确要求："批量预测要返回每行错误，不因一行坏数据使整批无结果。"
    这里故意把第 3、5 行弄坏，另外 8 行必须照常出结果。

    注意：pandas 3.0 开始不允许往 float64 列里直接塞字符串
    （会抛 LossySetitemError），所以先 astype(object) 再改。
    这不影响被测逻辑——真实场景里 CSV 读进来的坏列本来就可能已经是 object。
    """
    guan, meta = jiazai_haode
    tb, _, _ = yuanshi_shuju
    biao = tb.head(10).copy().astype(object)
    biao.iloc[2, 0] = "坏数据"
    biao.iloc[4, 1] = None

    jieguo, cuowu, tongji = yuce_piliang(guan, meta, biao, 0.0)
    assert tongji["zong_hang"] == 10
    assert tongji["shibai"] == 2
    assert tongji["chenggong"] == 8
    assert len(jieguo) == 8
    assert {c["hang"] for c in cuowu} == {3, 5}


def test_piliang_que_lie_quanbu_shibai(jiazai_haode, yuanshi_shuju):
    """整批都缺列时，返回空结果 + 全部错误，而不是抛异常。"""
    guan, meta = jiazai_haode
    tb, _, _ = yuanshi_shuju
    biao = tb[["alcohol", "malic_acid"]].head(5)

    jieguo, cuowu, tongji = yuce_piliang(guan, meta, biao, 0.0)
    assert jieguo.empty
    assert tongji["chenggong"] == 0
    assert len(cuowu) == 5


def test_piliang_he_dan_tiao_jieguo_yizhi(jiazai_haode, zhengchang_canshu):
    """同一行数据，走批量通道和走单条通道，结果必须一致。"""
    guan, meta = jiazai_haode
    biao = pd.DataFrame([zhengchang_canshu])
    piliang, _, _ = yuce_piliang(guan, meta, biao, 0.0)
    dan = yuce_dan_tiao(guan, meta, zhengchang_canshu, 0.0)
    assert piliang.iloc[0]["置信度"] == dan["kexindu"]


def test_piliang_yuzhi_shengxiao(jiazai_haode, yuanshi_shuju):
    """阈值提高后，"待复核"的行数必须变多。"""
    guan, meta = jiazai_haode
    tb, _, _ = yuanshi_shuju
    biao = tb.head(40)

    _, _, t0 = yuce_piliang(guan, meta, biao, 0.0)
    _, _, t9 = yuce_piliang(guan, meta, biao, 0.95)
    assert t9["di_yuzhi"] >= t0["di_yuzhi"]


# ==================== 模型加载的失败路径 ====================

def test_moxing_wenjian_bucunzai_baocuo(tmp_path):
    with pytest.raises(MoxingCuowu) as e:
        jiazai_moxing(str(tmp_path))
    assert "找不到模型文件" in str(e.value)


def test_meta_wenjian_bucunzai_baocuo(tmp_path):
    """只有模型没有元数据，也必须拒绝加载——缺一个都不能用。"""
    import joblib
    from tezheng import goujian_moxing_a, goujian_pipeline

    tb, bq, _ = jiazai_shuju()
    guan = goujian_pipeline(list(tb.columns), goujian_moxing_a(42))
    guan.fit(tb, bq)
    joblib.dump(guan, str(tmp_path / "model.joblib"))

    with pytest.raises(MoxingCuowu) as e:
        jiazai_moxing(str(tmp_path))
    assert "元数据" in str(e.value)
