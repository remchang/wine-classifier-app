# -*- coding: utf-8 -*-
"""
xunlian.py —— 训练脚本：比较两种算法、锁定模型、保存 Pipeline 与元数据。

跑法：
    python src/xunlian.py                 # 默认输出到 models/
    python src/xunlian.py --zhongzi 7     # 换随机种子，验证可复现性

流程（严格按实验要求）：
    1. 加载数据 + 质量检查
    2. 先划分训练/测试（分层），测试集从此封存
    3. 在训练集上做 5 折交叉验证，比较两个算法
    4. 按交叉验证的主指标选模型
    5. 在测试集上做且仅做一次最终评价
    6. 保存完整 Pipeline + 元数据（含 SHA256）
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# 让 `python src/xunlian.py` 和 `from src.xunlian import ...` 都能跑
_zhe_ge_wenjian = os.path.dirname(os.path.abspath(__file__))
if _zhe_ge_wenjian not in sys.path:
    sys.path.insert(0, _zhe_ge_wenjian)

from shuju import (  # noqa: E402
    SUIJI_ZHONGZI,
    huafen_shuju,
    jiancha_shuju,
    jiazai_shuju,
    shuju_zhiwen,
)
from tezheng import (  # noqa: E402
    MOXING_A_MING,
    MOXING_B_MING,
    goujian_moxing_a,
    goujian_moxing_b,
    goujian_pipeline,
)

# 主指标：macro F1。
# 为什么不用 accuracy：accuracy 会被大类带偏；wine 三类不算均衡，
# macro F1 对每个类同样看重，更适合当"选哪个模型"的依据。
ZHU_Zhibiao = "f1_macro"


def _git_commit():
    """拿当前 Commit，写进元数据。拿不到就返回 unknown，不要报错中断训练。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def jiaocha_yanzheng(moxing_ming, moxing, X_xunlian, y_xunlian, zhongzi):
    """
    在训练集上做 5 折分层交叉验证。

    注意：cross_validate 内部对每一折都会重新 fit 一遍 Pipeline，
    所以 StandardScaler 只会在该折的训练部分上拟合，不会泄漏。
    这也是"必须把预处理放进 Pipeline"的原因之一。
    """
    guan = goujian_pipeline(list(X_xunlian.columns), moxing)
    fen = StratifiedKFold(n_splits=5, shuffle=True, random_state=zhongzi)

    kaishi = time.time()
    jieguo = cross_validate(
        guan, X_xunlian, y_xunlian,
        cv=fen,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        n_jobs=1,          # 关掉并行，保证耗时数据可比
        return_train_score=False,
    )
    haoshi = time.time() - kaishi

    return {
        "moxing": moxing_ming,
        "zhe_shu": 5,
        "kua_jia_yanzheng_miao": round(haoshi, 3),
        "accuracy": {
            "jun_zhi": round(float(np.mean(jieguo["test_accuracy"])), 4),
            "biao_zhun_cha": round(float(np.std(jieguo["test_accuracy"])), 4),
        },
        "precision_macro": {
            "jun_zhi": round(float(np.mean(jieguo["test_precision_macro"])), 4),
            "biao_zhun_cha": round(float(np.std(jieguo["test_precision_macro"])), 4),
        },
        "recall_macro": {
            "jun_zhi": round(float(np.mean(jieguo["test_recall_macro"])), 4),
            "biao_zhun_cha": round(float(np.std(jieguo["test_recall_macro"])), 4),
        },
        "f1_macro": {
            "jun_zhi": round(float(np.mean(jieguo["test_f1_macro"])), 4),
            "biao_zhun_cha": round(float(np.std(jieguo["test_f1_macro"])), 4),
        },
    }


def zui_zhong_pingjia(guan, X_ceshi, y_ceshi, leibie_ming):
    """
    最终测试集评价。整个项目里只会调用一次。

    实验要求"测试集只用于最终一次评价"——所以这个函数不应该在调参循环里被调用。
    """
    y_yuce = guan.predict(X_ceshi)
    gailv = guan.predict_proba(X_ceshi)

    return {
        "yangben_shu": int(len(y_ceshi)),
        "accuracy": round(float(accuracy_score(y_ceshi, y_yuce)), 4),
        "precision_macro": round(float(precision_score(
            y_ceshi, y_yuce, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(
            y_ceshi, y_yuce, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(
            y_ceshi, y_yuce, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(
            y_ceshi, y_yuce, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(
            y_ceshi, y_yuce, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_ceshi, y_yuce, average="weighted", zero_division=0)), 4),
        "hunxiao_juzhen": confusion_matrix(y_ceshi, y_yuce).tolist(),
        "mei_lei_baogao": classification_report(
            y_ceshi, y_yuce, target_names=leibie_ming,
            zero_division=0, output_dict=True),
        "pingjun_kexindu": round(float(np.mean(np.max(gailv, axis=1))), 4),
    }


def shiji_xunlian(shuchu_mulu="models", zhongzi=SUIJI_ZHONGZI, shuchu_baogao=None):
    """
    完整训练流程。返回一个结果字典（也写成 JSON 存盘，方便报告和 PPT 引用）。
    """
    tezheng_biao, biaoqian, leibie_ming = jiazai_shuju()
    zhiliang = jiancha_shuju(tezheng_biao, biaoqian)

    if zhiliang["wenti"]:
        # 有问题不静默继续——宁可报错，也不要训出一个说不清来历的模型
        raise RuntimeError("数据质量检查未通过：%s" % "；".join(zhiliang["wenti"]))

    X_xunlian, X_ceshi, y_xunlian, y_ceshi = huafen_shuju(
        tezheng_biao, biaoqian, zhongzi
    )

    # ---- 比较两个算法（只在训练集上，用交叉验证）----
    bijiao = []
    for ming, jianzao in ((MOXING_A_MING, goujian_moxing_a),
                          (MOXING_B_MING, goujian_moxing_b)):
        bijiao.append(jiaocha_yanzheng(
            ming, jianzao(zhongzi), X_xunlian, y_xunlian, zhongzi
        ))

    # ---- 按主指标选最优 ----
    you_lie = sorted(bijiao, key=lambda x: -x[ZHU_Zhibiao]["jun_zhi"])
    zui_hao = you_lie[0]["moxing"]

    # ---- 在完整训练集上重新拟合最佳模型 ----
    if zui_hao == MOXING_A_MING:
        guan = goujian_pipeline(list(X_xunlian.columns), goujian_moxing_a(zhongzi))
    else:
        guan = goujian_pipeline(list(X_xunlian.columns), goujian_moxing_b(zhongzi))

    kaishi = time.time()
    guan.fit(X_xunlian, y_xunlian)
    xunlian_miao = time.time() - kaishi

    # ---- 测试集：只评一次 ----
    ceshi_jieguo = zui_zhong_pingjia(guan, X_ceshi, y_ceshi, leibie_ming)

    # ---- 保存 ----
    os.makedirs(shuchu_mulu, exist_ok=True)
    moxing_lujing = os.path.join(shuchu_mulu, "model.joblib")
    joblib.dump(guan, moxing_lujing)

    with open(moxing_lujing, "rb") as f:
        moxing_sha256 = hashlib.sha256(f.read()).hexdigest()

    meta = {
        "moxing_ming": zui_hao,
        "moxing_lujing": "models/model.joblib",
        "moxing_sha256": moxing_sha256,
        "moxing_daxiao_zijie": os.path.getsize(moxing_lujing),
        "xunlian_shijian": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "xunlian_miao": round(xunlian_miao, 3),
        "suiji_zhongzi": zhongzi,
        "shuju_ji": "sklearn.datasets.load_wine",
        "shuju_zhiwen_sha256": shuju_zhiwen(tezheng_biao, biaoqian),
        "yangben_shu": zhiliang["yangben_shu"],
        "tezheng_ming": zhiliang["tezheng_ming"],
        "leibie_ming": leibie_ming,
        "leibie_yingshe": {str(i): n for i, n in enumerate(leibie_ming)},
        "zhu_zhibiao": ZHU_Zhibiao,
        "huanjing": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "git_commit": _git_commit(),
        "kua_jia_yanzheng": bijiao,
        "ceshi_ji": ceshi_jieguo,
        "gailv_jiaozhun_shuoming": (
            "未做概率校准。树模型的 predict_proba 倾向于给出接近 0/1 的值，"
            "不能把概率当成'事实确定性'来解读。界面上的置信度只作为提示。"
        ),
    }

    with open(os.path.join(shuchu_mulu, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    baogao = {
        "zhiliang_jiancha": zhiliang,
        "bijiao": bijiao,
        "xuanzhong": zui_hao,
        "ceshi_ji": ceshi_jieguo,
        "xunlian_miao": round(xunlian_miao, 3),
        "moxing_sha256": moxing_sha256,
    }
    if shuchu_baogao:
        with open(shuchu_baogao, "w", encoding="utf-8") as f:
            json.dump(baogao, f, ensure_ascii=False, indent=2)

    return baogao


def main():
    p = argparse.ArgumentParser(description="训练并比较两种分类算法")
    p.add_argument("--shuchu", default="models", help="模型输出目录")
    p.add_argument("--zhongzi", type=int, default=SUIJI_ZHONGZI, help="随机种子")
    p.add_argument("--baogao", default="models/xunlian_baogao.json", help="结果 JSON 路径")
    a = p.parse_args()

    jieguo = shiji_xunlian(a.shuchu, a.zhongzi, a.baogao)

    print("=" * 66)
    print("数据：%d 个样本，%d 个特征，%d 个类别" % (
        jieguo["zhiliang_jiancha"]["yangben_shu"],
        jieguo["zhiliang_jiancha"]["tezheng_shu"],
        jieguo["zhiliang_jiancha"]["leibie_shu"]))
    print("类别分布：", jieguo["zhiliang_jiancha"]["leibie_fenbu"])
    print("-" * 66)
    print("5 折交叉验证（只在训练集上做）：")
    print("  %-22s %8s %8s %8s %8s %10s" % (
        "算法", "acc", "P(macro)", "R(macro)", "F1(macro)", "耗时(s)"))
    for b in jieguo["bijiao"]:
        print("  %-22s %8.4f %8.4f %8.4f %8.4f %10.3f" % (
            b["moxing"], b["accuracy"]["jun_zhi"], b["precision_macro"]["jun_zhi"],
            b["recall_macro"]["jun_zhi"], b["f1_macro"]["jun_zhi"],
            b["kua_jia_yanzheng_miao"]))
    print("-" * 66)
    print("按 %s 选中：%s" % (ZHU_Zhibiao, jieguo["xuanzhong"]))
    c = jieguo["ceshi_ji"]
    print("测试集（只评这一次，%d 个样本）：" % c["yangben_shu"])
    print("  Accuracy         = %.4f" % c["accuracy"])
    print("  Precision(macro) = %.4f" % c["precision_macro"])
    print("  Recall(macro)    = %.4f" % c["recall_macro"])
    print("  F1(macro)        = %.4f" % c["f1_macro"])
    print("  混淆矩阵：")
    for hang in c["hunxiao_juzhen"]:
        print("    ", hang)
    print("=" * 66)
    print("模型已保存到 %s，元数据写入 %s" % (a.shuchu, os.path.join(a.shuchu, "model_meta.json")))


if __name__ == "__main__":
    main()
