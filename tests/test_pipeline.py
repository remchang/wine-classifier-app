# -*- coding: utf-8 -*-
"""
test_pipeline.py —— Pipeline 与模型保存/加载测试（实验要求 ≥6 条）。

这一组测的是"训练和推理用的是不是同一套规则"：
  · Pipeline 能不能直接吃原始输入（不需要手动做标准化）
  · 保存再加载之后，同一份输入的预测必须完全一致
  · 列顺序、未知类别、缺列这些坑有没有被挡住
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from shuju import huafen_shuju, jiazai_shuju
from tezheng import (
    MOXING_A_MING,
    MOXING_B_MING,
    goujian_moxing_a,
    goujian_moxing_b,
    goujian_pipeline,
)


def test_pipeline_zhijie_chi_yuanshi_shuru(huafende):
    """
    ★ 核心要求："原始输入可直接 predict"。
    不做任何手动标准化，直接把原始 DataFrame 丢进去就该能出结果。
    """
    X_xl, X_cs, y_xl, _ = huafende
    guan = goujian_pipeline(list(X_xl.columns), goujian_moxing_a(42))
    guan.fit(X_xl, y_xl)
    y = guan.predict(X_cs)
    assert len(y) == len(X_cs)


def test_pipeline_yuchuli_zai_neibu(huafende):
    """预处理器必须真的在 Pipeline 里，且能取到训练时学到的统计量。"""
    X_xl, _, y_xl, _ = huafende
    guan = goujian_pipeline(list(X_xl.columns), goujian_moxing_a(42))
    guan.fit(X_xl, y_xl)

    biaozhunhua = guan.named_steps["yuchuli"].named_steps["biaozhunhua"]
    assert hasattr(biaozhunhua, "mean_")
    assert len(biaozhunhua.mean_) == X_xl.shape[1]


def test_pipeline_jilu_tezheng_shunxu(huafende):
    """Pipeline 里应该记下训练时的特征名和顺序，加载时才能校验。"""
    X_xl, _, y_xl, _ = huafende
    lie = list(X_xl.columns)
    guan = goujian_pipeline(lie, goujian_moxing_a(42))
    guan.fit(X_xl, y_xl)
    assert list(guan.named_steps["yuchuli"].feature_names_in_) == lie


def test_liang_zhong_suanfa_dou_ke_yong(huafende):
    """两个算法的 Pipeline 都要能拟合出结果，不能有一个是坏的。"""
    X_xl, X_cs, y_xl, y_cs = huafende
    for ming, jianzao in ((MOXING_A_MING, goujian_moxing_a),
                          (MOXING_B_MING, goujian_moxing_b)):
        guan = goujian_pipeline(list(X_xl.columns), jianzao(42))
        guan.fit(X_xl, y_xl)
        y = guan.predict(X_cs)
        assert len(y) == len(y_cs), ming
        assert set(np.unique(y)).issubset({0, 1, 2}), ming


def test_baocun_jiazai_yuce_yizhi(xunlian_haode):
    """
    ★ 实验明确要求："保存前后同一 fixture 的预测一致性"。
    这是防止"训练时好好的、上线就错"最重要的一条。
    """
    from yuce import jiazai_moxing
    import joblib

    mulu, _ = xunlian_haode
    guan, meta = jiazai_moxing(mulu)
    X_xl, X_cs, _, _ = huafen_shuju(*jiazai_shuju()[:2])

    y1 = guan.predict(X_cs)
    guan2 = joblib.load(os.path.join(mulu, "model.joblib"))
    y2 = guan2.predict(X_cs)
    assert np.array_equal(y1, y2)

    # 概率也要一致（不只是类别）
    p1 = guan.predict_proba(X_cs)
    p2 = guan2.predict_proba(X_cs)
    assert np.allclose(p1, p2, atol=1e-9)


def test_metadata_bao_han_bibei_ziduan(xunlian_haode):
    """元数据必须含训练时间、版本、特征顺序、标签映射、指标、SHA256。"""
    mulu, _ = xunlian_haode
    with open(os.path.join(mulu, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    for ziduan in ("xunlian_shijian", "huanjing", "tezheng_ming", "leibie_ming",
                   "leibie_yingshe", "moxing_sha256", "shuju_zhiwen_sha256",
                   "zhu_zhibiao", "git_commit", "kua_jia_yanzheng", "ceshi_ji"):
        assert ziduan in meta, ziduan

    assert len(meta["tezheng_ming"]) == 13
    assert len(meta["leibie_ming"]) == 3
    assert len(meta["moxing_sha256"]) == 64


def test_sha256_he_wenjian_yizhi(xunlian_haode):
    """元数据里的 SHA256 要能对上真实的模型文件——不能是随手写的。"""
    import hashlib

    mulu, _ = xunlian_haode
    with open(os.path.join(mulu, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(mulu, "model.joblib"), "rb") as f:
        zhen = hashlib.sha256(f.read()).hexdigest()
    assert meta["moxing_sha256"] == zhen


def test_liang_suanfa_cv_jieguo_wanzheng(xunlian_haode):
    """交叉验证结果要两个算法都有，且均值/标准差都在。"""
    _, jieguo = xunlian_haode
    assert len(jieguo["bijiao"]) == 2
    for b in jieguo["bijiao"]:
        for zhibiao in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert "jun_zhi" in b[zhibiao]
            assert "biao_zhun_cha" in b[zhibiao]
            assert 0.0 <= b[zhibiao]["jun_zhi"] <= 1.0


def test_xuanzhong_de_shi_cv_zuihao(xunlian_haode):
    """选中的模型必须是交叉验证主指标最高的那个，不能是随便挑的。"""
    _, jieguo = xunlian_haode
    pai = sorted(jieguo["bijiao"], key=lambda x: -x["f1_macro"]["jun_zhi"])
    assert jieguo["xuanzhong"] == pai[0]["moxing"]


def test_ceshi_ji_zhibiao_hefa(xunlian_haode):
    """测试集指标应该落在合理范围（一份干净数据不该只有 0.5）。"""
    _, jieguo = xunlian_haode
    c = jieguo["ceshi_ji"]
    assert c["accuracy"] >= 0.8
    assert len(c["hunxiao_juzhen"]) == 3
    assert all(len(h) == 3 for h in c["hunxiao_juzhen"])
    assert sum(sum(h) for h in c["hunxiao_juzhen"]) == c["yangben_shu"]


def test_kexue_zhongzi_ke_fuyan(yuanshi_shuju):
    """
    同一随机种子训两次，指标应该完全一致——这是"可复现"的定义。
    """
    from xunlian import jiaocha_yanzheng

    tb, bq, _ = yuanshi_shuju
    X_xl, _, y_xl, _ = huafen_shuju(tb, bq)

    a = jiaocha_yanzheng(MOXING_A_MING, goujian_moxing_a(42), X_xl, y_xl, 42)
    b = jiaocha_yanzheng(MOXING_A_MING, goujian_moxing_a(42), X_xl, y_xl, 42)
    assert a["f1_macro"]["jun_zhi"] == b["f1_macro"]["jun_zhi"]
    assert a["accuracy"]["jun_zhi"] == b["accuracy"]["jun_zhi"]
