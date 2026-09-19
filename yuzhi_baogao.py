# -*- coding: utf-8 -*-
"""一次性脚本：打印阈值分析表（用于写报告）。"""
import os
import sys

GEN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(GEN, "src"))

from shuju import huafen_shuju, jiazai_shuju      # noqa: E402
from yuce import jiazai_moxing, pinggu_yuzhi       # noqa: E402

guan, meta = jiazai_moxing(os.path.join(GEN, "models"))
tb, bq, _ = jiazai_shuju()
_, X_cs, _, y_cs = huafen_shuju(tb, bq)

print("选中算法:", meta["moxing_ming"])
print("测试集样本数:", len(y_cs))
print()
print("%-8s %-10s %-10s %-14s %-10s %-16s" % (
    "阈值", "自动判定数", "覆盖率", "自动准确率", "转人工数", "转人工中本来对的"))
for j in pinggu_yuzhi(guan, meta, X_cs, y_cs, [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    print("%-8.2f %-10d %-10.4f %-14s %-10d %-16d" % (
        j["yuzhi"], j["fugai_shu"], j["fugai_lv"],
        ("%.4f" % j["zidong_zhunque_lv"]) if j["zidong_zhunque_lv"] is not None else "-",
        j["jujue_shu"], j["jujue_zhong_ben_ke_dui"]))

print()
print("模型 SHA256:", meta["moxing_sha256"])
print("数据指纹:", meta["shuju_zhiwen_sha256"])
