# -*- coding: utf-8 -*-
"""
yuce.py —— 推理封装：加载模型、单条预测、批量预测、置信度阈值。

这个模块是"训练"和"应用"之间的唯一接口。
app.py 只调这里的函数，不直接碰 sklearn —— 这样以后换模型不用改界面。
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn

from shuju import ShujuCuowu, zu_canshu_zi

# 置信度低于这个值就提示"建议人工复核"
MOREN_YUZHI = 0.60


class MoxingCuowu(Exception):
    """模型加载或元数据不匹配时抛这个。"""
    pass


def jiazai_moxing(mulu="models"):
    """
    加载 Pipeline 和元数据。

    加载前会校验三件事，这是实验要求的"避免旧界面调用新 Schema"：
      ① 元数据里的 scikit-learn 版本和当前环境一致（大版本不同直接拒）
      ② 元数据里的特征名顺序和 Pipeline 里记录的一致
      ③ 元数据里的类别名数量正确

    ★ 安全提示：joblib/pickle 反序列化时会执行代码。
      所以这里只允许加载"本仓库 src/xunlian.py 训练出来的模型"，
      绝不接受用户上传的任意模型文件。界面里也没有上传模型的入口。
    """
    moxing_lujing = os.path.join(mulu, "model.joblib")
    meta_lujing = os.path.join(mulu, "model_meta.json")

    if not os.path.exists(moxing_lujing):
        raise MoxingCuowu(
            "找不到模型文件 %s。先跑： python src/xunlian.py" % moxing_lujing
        )
    if not os.path.exists(meta_lujing):
        raise MoxingCuowu(
            "找不到元数据 %s。模型和元数据必须成对存在，缺一个都不能用。" % meta_lujing
        )

    with open(meta_lujing, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # ① 版本校验：只比大版本号，小版本差异不阻断，但会记下来
    meta_banben = meta.get("huanjing", {}).get("scikit_learn", "unknown")
    dangqian_banben = sklearn.__version__

    def _da_banben(v):
        return str(v).split(".")[0]

    if meta_banben != "unknown" and _da_banben(meta_banben) != _da_banben(dangqian_banben):
        raise MoxingCuowu(
            "模型是用 scikit-learn %s 训练的，当前环境是 %s，主版本不一致。"
            "跨主版本反序列化不保证兼容，请用相同环境重新训练。" % (meta_banben, dangqian_banben)
        )

    guan = joblib.load(moxing_lujing)

    # ② 特征名与顺序校验
    meta_tezheng = list(meta.get("tezheng_ming", []))
    guan_tezheng = []
    try:
        guan_tezheng = list(guan.named_steps["yuchuli"].feature_names_in_)
    except Exception:
        guan_tezheng = []

    if meta_tezheng and guan_tezheng and meta_tezheng != guan_tezheng:
        raise MoxingCuowu(
            "元数据里的特征顺序和 Pipeline 里的不一致，"
            "这会导致预测静默出错。请重新训练。"
        )

    # ③ 类别
    if not meta.get("leibie_ming"):
        raise MoxingCuowu("元数据缺少 leibie_ming，无法把预测结果翻译成类别名。")

    meta["_dangqian_banben"] = dangqian_banben
    meta["_banben_yizhi"] = (meta_banben == dangqian_banben)
    return guan, meta


def _zhengli_gailv(guan, X):
    """把 predict_proba 的结果整理成 (最大概率, 最大概率对应的类别下标)。"""
    gailv = guan.predict_proba(X)
    zuida = np.max(gailv, axis=1)
    xiabiao = np.argmax(gailv, axis=1)
    return gailv, zuida, xiabiao


def yuce_dan_tiao(guan, meta, zidian, yuzhi=MOREN_YUZHI):
    """
    单条预测。

    返回字典：
        leibie       预测类别名
        leibie_xiabiao
        gailv        每个类别的概率（list）
        kexindu      最大概率
        keyi_xinren  是否达到置信度阈值（自主功能）
        tishi        给用户看的一句话
    """
    tezheng_ming = meta["tezheng_ming"]
    X = zu_canshu_zi(tezheng_ming, zidian)      # 内部会校验缺字段/非数字
    gailv, zuida, xiabiao = _zhengli_gailv(guan, X)

    leibie_ming = meta["leibie_ming"]
    kb = int(xiabiao[0])
    kexindu = float(zuida[0])
    keyi = kexindu >= float(yuzhi)

    if keyi:
        tishi = "置信度 %.1f%%，达到阈值 %.0f%%，可以按此结果处理。" % (
            kexindu * 100, float(yuzhi) * 100)
    else:
        tishi = ("置信度只有 %.1f%%，低于阈值 %.0f%%。"
                 "模型只是倾向这个类别，建议人工复核，不要直接采信。"
                 % (kexindu * 100, float(yuzhi) * 100))

    return {
        "leibie": leibie_ming[kb],
        "leibie_xiabiao": kb,
        "gailv": {leibie_ming[i]: round(float(gailv[0][i]), 4)
                  for i in range(len(leibie_ming))},
        "kexindu": round(kexindu, 4),
        "keyi_xinren": keyi,
        "yuzhi": float(yuzhi),
        "tishi": tishi,
    }


def yuce_piliang(guan, meta, biao, yuzhi=MOREN_YUZHI):
    """
    批量预测（自主功能）。

    实验要求："批量预测要返回每行错误，不因一行坏数据使整批无结果。"

    所以这里逐行 try/except：
      坏行标出来是哪一列出的问题，好行照常预测。

    返回 (jieguo_biao, cuowu_liebiao, tongji)
    """
    tezheng_ming = meta["tezheng_ming"]
    leibie_ming = meta["leibie_ming"]

    hao_hang = []      # (原始行号, 参数dict)
    cuowu_liebiao = []

    for i, (_, hang) in enumerate(biao.iterrows()):
        yuan_hang = int(i) + 1        # 给用户看的行号从 1 开始
        try:
            zidian = {}
            for c in tezheng_ming:
                if c not in biao.columns:
                    raise ShujuCuowu("缺少列 %s" % c)
                zidian[c] = hang[c]
            # 用和单条预测完全相同的整理逻辑，保证两条路径结果一致
            zu_canshu_zi(tezheng_ming, zidian)
            hao_hang.append((yuan_hang, zidian))
        except ShujuCuowu as e:
            cuowu_liebiao.append({"hang": yuan_hang, "yuanyin": str(e)})
        except Exception as e:                      # noqa: BLE001
            cuowu_liebiao.append({"hang": yuan_hang, "yuanyin": "无法解析：%s" % e})

    if not hao_hang:
        return pd.DataFrame(), cuowu_liebiao, {
            "zong_hang": len(biao), "chenggong": 0, "shibai": len(cuowu_liebiao),
            "di_yuzhi": 0,
        }

    X = pd.DataFrame(
        [[float(z[c]) for c in tezheng_ming] for _, z in hao_hang],
        columns=list(tezheng_ming),    # ★ 列顺序和训练时一致
    )
    gailv, zuida, xiabiao = _zhengli_gailv(guan, X)

    jieguo = []
    di_yuzhi_shu = 0
    for k, (yuan_hang, _) in enumerate(hao_hang):
        kb = int(xiabiao[k])
        kexindu = float(zuida[k])
        keyi = kexindu >= float(yuzhi)
        if not keyi:
            di_yuzhi_shu += 1
        jieguo.append({
            "原始行号": yuan_hang,
            "预测类别": leibie_ming[kb] if keyi else ("待复核（%s?）" % leibie_ming[kb]),
            "置信度": round(kexindu, 4),
            "达到阈值": keyi,
        })

    return pd.DataFrame(jieguo), cuowu_liebiao, {
        "zong_hang": len(biao),
        "chenggong": len(hao_hang),
        "shibai": len(cuowu_liebiao),
        "di_yuzhi": di_yuzhi_shu,
    }


def pinggu_yuzhi(guan, meta, X_ceshi, y_ceshi, yuzhi_liebiao):
    """
    自主功能的量化证据：阈值怎么影响"覆盖率"和"在覆盖率内的准确率"。

    逻辑：
       置信度 < 阈值的样本，我们拒绝自动判定，交给人工。
       于是阈值越高 → 拒绝得越多（覆盖率下降），但剩下的越准（准确率上升）。
       这就是 Precision / Recall 之外，实际业务里更该看的权衡。
    """
    gailv = guan.predict_proba(X_ceshi)
    zuida = np.max(gailv, axis=1)
    yuce = np.argmax(gailv, axis=1)
    y_true = np.asarray(y_ceshi)

    jieguo = []
    for yz in yuzhi_liebiao:
        baoliu = zuida >= float(yz)
        fg = int(baoliu.sum())
        if fg == 0:
            jieguo.append({
                "yuzhi": round(float(yz), 2), "fugai_shu": 0, "fugai_lv": 0.0,
                "zidong_zhunque_lv": None, "jujue_shu": int(len(baoliu)),
                "jujue_zhong_ben_ke_dui": 0,
            })
            continue
        zhengque = int((yuce[baoliu] == y_true[baoliu]).sum())
        jujue = ~baoliu
        jieguo.append({
            "yuzhi": round(float(yz), 2),
            "fugai_shu": fg,
            "fugai_lv": round(fg / len(baoliu), 4),
            "zidong_zhunque_lv": round(zhengque / fg, 4),
            "jujue_shu": int(jujue.sum()),
            "jujue_zhong_ben_ke_dui": int((yuce[jujue] == y_true[jujue]).sum()),
        })
    return jieguo
