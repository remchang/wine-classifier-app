# -*- coding: utf-8 -*-
"""
app.py —— Streamlit 推理界面。

跑法：
    streamlit run app.py

设计要点：
  · 模型用 st.cache_resource 只加载一次，不在每次点击时重新训练/加载；
  · 表单字段严格按训练 Schema 生成，不手写字段名，避免和训练不一致；
  · 所有异常都转成友好提示，不把堆栈甩到页面上；
  · 明确标注"仅用于课程演示"，不承诺生产可用。
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

# 让 app.py 能 import 到 src/ 下的模块
GEN = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(GEN, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from shuju import ShujuCuowu, jiazai_shuju, jiancha_shuju  # noqa: E402
from yuce import (  # noqa: E402
    MOREN_YUZHI,
    MoxingCuowu,
    jiazai_moxing,
    pinggu_yuzhi,
    yuce_dan_tiao,
    yuce_piliang,
)

st.set_page_config(
    page_title="葡萄酒分类系统 · 实验04",
    page_icon="🍷",
    layout="wide",
)

MOXING_MULU = os.path.join(GEN, "models")


# ---------------- 资源加载（只跑一次） ----------------

@st.cache_resource(show_spinner="正在加载模型…")
def qu_moxing():
    """缓存模型。返回 (pipeline, meta, cuowu_wenben)。"""
    try:
        guan, meta = jiazai_moxing(MOXING_MULU)
        return guan, meta, None
    except MoxingCuowu as e:
        return None, None, str(e)


def qu_ceshi_ji():
    """拿测试集，给阈值分析用。同样缓存，避免每次重算。"""
    from shuju import huafen_shuju
    tb, bq, lm = jiazai_shuju()
    _, X_ceshi, _, y_ceshi = huafen_shuju(tb, bq)
    return X_ceshi, y_ceshi, lm


# ---------------- 顶栏 ----------------

guan, meta, jiazai_cuowu = qu_moxing()

st.title("🍷 葡萄酒产地分类系统")
st.caption(
    "《开源软件与新技术》实验04 · scikit-learn Pipeline + Streamlit ｜ "
    "⚠️ 仅用于课程演示，不构成任何实际判定依据"
)

if jiazai_cuowu:
    st.error(jiazai_cuowu)
    st.info("先在项目根目录执行： `python src/xunlian.py`，训练出模型后再刷新本页。")
    st.stop()

tezheng_ming = meta["tezheng_ming"]
leibie_ming = meta["leibie_ming"]

# ---------------- 侧边栏：模型信息 + 阈值 ----------------

with st.sidebar:
    st.header("模型信息")
    st.write("**算法**：`%s`" % meta["moxing_ming"])
    st.write("**主指标**：`%s`" % meta["zhu_zhibiao"])
    st.write("**训练时间**：%s" % meta["xunlian_shijian"])
    st.write("**样本数**：%d" % meta["yangben_shu"])
    st.write("**scikit-learn**：%s" % meta["huanjing"]["scikit_learn"])
    st.write("**Commit**：`%s`" % str(meta.get("git_commit", "unknown"))[:10])
    st.caption("模型 SHA256（前16位）：`%s`" % meta["moxing_sha256"][:16])

    if not meta.get("_banben_yizhi", True):
        st.warning("当前 scikit-learn 版本和训练时不完全一致，结果可能有细微差异。")

    st.divider()
    st.header("置信度阈值")
    yuzhi = st.slider(
        "低于此阈值就转人工复核", min_value=0.0, max_value=1.0,
        value=MOREN_YUZHI, step=0.05,
        help="越高越保守：自动判定的样本变少，但留下来的更可靠。",
    )

    with st.expander("这个阈值是干什么的？", expanded=False):
        st.write(
            "模型对每一类都会给一个概率。取最大的那个作为**置信度**。\n\n"
            "置信度低于阈值的样本，我们**不自动给结论**，而是标成「待复核」。\n\n"
            "这是**自主功能**：因为实际业务里，判错的代价是不对称的——"
            "把 A 产地当成 B 产地卖出去，比多找个人看一眼严重得多。"
        )

# ---------------- 标签页 ----------------

tab_dan, tab_pi, tab_yz, tab_mo = st.tabs(
    ["单条预测", "批量预测", "阈值分析", "模型信息"]
)


# ==================== ① 单条预测 ====================

with tab_dan:
    st.subheader("输入 13 个特征，实时预测产地")

    # 从真实数据里取典型值当默认，比自己编一个像样
    tb, bq, lm = jiazai_shuju()
    zhongweishu = tb.median()
    zuixiaozhi = tb.min()
    zuidazhi = tb.max()

    with st.form("dan_tiao_biaodan"):
        st.caption("字段与训练 Schema 完全一致，范围和单位见输入框提示。")
        zhi = {}
        lie = st.columns(3)
        for i, c in enumerate(tezheng_ming):
            with lie[i % 3]:
                zhi[c] = st.number_input(
                    c.replace("_", " "),
                    value=float(zhongweishu[c]),
                    min_value=float(zuixiaozhi[c]) - abs(float(zuixiaozhi[c])) - 1,
                    max_value=float(zuidazhi[c]) * 2 + 1,
                    format="%.5f",
                    help="训练数据范围：%.4f ~ %.4f，中位数 %.4f"
                         % (zuixiaozhi[c], zuidazhi[c], zhongweishu[c]),
                )
        tijiao = st.form_submit_button("预测", use_container_width=True)

    if tijiao:
        try:
            jieguo = yuce_dan_tiao(guan, meta, zhi, yuzhi)
            a, b = st.columns([1, 2])
            with a:
                st.metric("预测类别", jieguo["leibie"])
                st.metric("置信度", "%.1f%%" % (jieguo["kexindu"] * 100))
            with b:
                st.write("**各类别概率：**")
                st.bar_chart(pd.Series(jieguo["gailv"], name="概率"))
                if jieguo["keyi_xinren"]:
                    st.success(jieguo["tishi"])
                else:
                    st.warning(jieguo["tishi"])
            st.caption(
                "⚠️ 概率来自 %.0f%% 的训练集分布，**未做概率校准**；"
                "树模型的概率往往偏向 0 或 1，请勿当作事实确定性。"
                % (100 - 25)
            )
        except ShujuCuowu as e:
            st.error("输入有问题：%s" % e)
        except Exception as e:                     # noqa: BLE001
            st.error("预测失败：%s" % e)


# ==================== ② 批量预测（自主功能） ====================

with tab_pi:
    st.subheader("上传 CSV，逐行预测")
    st.caption(
        "需要包含这 13 列：%s。**一行坏数据不会让整批失败**，"
        "坏行会单独列出来说明原因。" % ", ".join(tezheng_ming)
    )

    shangchuan = st.file_uploader("选择 CSV 文件", type=["csv"])

    b, c = st.columns(2)
    with b:
        if st.button("用示例数据试一下", use_container_width=True):
            yangli = tb.head(8).copy()
            shangchuan = None
            st.session_state["yangli"] = yangli
    with c:
        if st.button("清空", use_container_width=True):
            st.session_state.pop("yangli", None)

    biao = None
    if shangchuan is not None:
        try:
            # 编码问题：中文 CSV 很多是 GBK，utf-8 读不出来就换一个再试
            try:
                biao = pd.read_csv(shangchuan, encoding="utf-8")
            except UnicodeDecodeError:
                shangchuan.seek(0)
                biao = pd.read_csv(shangchuan, encoding="gbk")
                st.info("这个文件不是 UTF-8，已按 GBK 读取。")
        except Exception as e:                     # noqa: BLE001
            st.error("读文件失败：%s" % e)
    elif "yangli" in st.session_state:
        biao = st.session_state["yangli"]

    if biao is not None:
        st.write("**读到的数据（前 5 行）：**")
        st.dataframe(biao.head(), use_container_width=True)

        try:
            jieguo_biao, cuowu, tongji = yuce_piliang(guan, meta, biao, yuzhi)

            d, e, f = st.columns(3)
            d.metric("总行数", tongji["zong_hang"])
            e.metric("成功预测", tongji["chenggong"])
            f.metric("需要人工复核", tongji["di_yuzhi"])

            if cuowu:
                st.warning("有 %d 行没法处理，已在下面列出原因：" % len(cuowu))
                st.dataframe(pd.DataFrame(cuowu), use_container_width=True)

            if not jieguo_biao.empty:
                st.write("**预测结果：**")
                st.dataframe(jieguo_biao, use_container_width=True)
                st.download_button(
                    "下载结果 CSV",
                    jieguo_biao.to_csv(index=False).encode("utf-8-sig"),
                    file_name="yuce_jieguo.csv",
                    mime="text/csv",
                )
            else:
                st.error("所有行都处理失败了，请检查列名是否和训练 Schema 一致。")
        except ShujuCuowu as e:
            st.error("数据不符合 Schema：%s" % e)
        except Exception as e:                     # noqa: BLE001
            st.error("批量预测失败：%s" % e)


# ==================== ③ 阈值分析（自主功能的量化证据） ====================

with tab_yz:
    st.subheader("置信度阈值怎么影响覆盖率与准确率")
    st.write(
        "光加一个滑块不算自主功能。这里用**测试集**算出："
        "阈值提高时，自动判定的样本变少（覆盖率下降），"
        "但留下来的更准（自动准确率上升）。这是可以量化的取舍证据。"
    )

    X_ceshi, y_ceshi, _ = qu_ceshi_ji()
    yuzhi_liebiao = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    jieguo_yz = pinggu_yuzhi(guan, meta, X_ceshi, y_ceshi, yuzhi_liebiao)
    biao_yz = pd.DataFrame(jieguo_yz)
    biao_yz.columns = ["阈值", "自动判定数", "覆盖率", "自动判定准确率", "转人工数", "转人工中本来判对的"]

    st.dataframe(
        biao_yz.style.format({
            "阈值": "{:.2f}", "覆盖率": "{:.2%}", "自动判定准确率": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.write("**覆盖率 vs 自动准确率**")
    st.line_chart(
        biao_yz.set_index("阈值")[["覆盖率", "自动判定准确率"]],
        use_container_width=True,
    )

    st.info(
        "**怎么读这张表**：阈值 0（即不做任何过滤）时，覆盖率 100%%，"
        "自动准确率就等于模型的整体 accuracy。阈值越高，转人工的越多，"
        "自动准确率越接近 100%%。\n\n"
        "**边界说明**：这只是测试集（%d 个样本）上的统计，"
        "样本一少数字就会抖。真实系统里这个阈值该由业务上的误判代价来定，"
        "不能只看哪一行数字好看。" % len(y_ceshi)
    )


# ==================== ④ 模型信息 ====================

with tab_mo:
    st.subheader("模型元数据（model_meta.json）")
    st.write(
        "保存模型时把训练环境、特征顺序、标签映射、指标和 SHA256 一起存下来，"
        "加载时校验。这样以后表现不对，能确认当时用的是哪份数据和哪版代码。"
    )

    kj = st.columns(4)
    kj[0].metric("Accuracy", "%.4f" % meta["ceshi_ji"]["accuracy"])
    kj[1].metric("Precision(macro)", "%.4f" % meta["ceshi_ji"]["precision_macro"])
    kj[2].metric("Recall(macro)", "%.4f" % meta["ceshi_ji"]["recall_macro"])
    kj[3].metric("F1(macro)", "%.4f" % meta["ceshi_ji"]["f1_macro"])

    st.write("**两种算法的交叉验证对比（只在训练集上做）**")
    rows = []
    for b in meta["kua_jia_yanzheng"]:
        rows.append({
            "算法": b["moxing"],
            "Accuracy": b["accuracy"]["jun_zhi"],
            "Accuracy 标准差": b["accuracy"]["biao_zhun_cha"],
            "Precision(macro)": b["precision_macro"]["jun_zhi"],
            "Recall(macro)": b["recall_macro"]["jun_zhi"],
            "F1(macro)": b["f1_macro"]["jun_zhi"],
            "F1 标准差": b["f1_macro"]["biao_zhun_cha"],
            "耗时(秒)": b["kua_jia_yanzheng_miao"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.write("**测试集混淆矩阵**（行=真实，列=预测）")
    hm = pd.DataFrame(
        meta["ceshi_ji"]["hunxiao_juzhen"],
        index=["真实 " + n for n in leibie_ming],
        columns=["预测 " + n for n in leibie_ming],
    )
    st.dataframe(hm, use_container_width=True)

    st.write("**每类明细**")
    mei = meta["ceshi_ji"]["mei_lei_baogao"]
    st.dataframe(
        pd.DataFrame({
            "precision": [mei[n]["precision"] for n in leibie_ming],
            "recall": [mei[n]["recall"] for n in leibie_ming],
            "f1-score": [mei[n]["f1-score"] for n in leibie_ming],
            "support": [mei[n]["support"] for n in leibie_ming],
        }, index=leibie_ming),
        use_container_width=True,
    )

    with st.expander("完整 model_meta.json"):
        st.code(json.dumps(meta, ensure_ascii=False, indent=2), language="json")

    st.warning(
        "**已知限制**：\n"
        "1. 数据是 sklearn 内置的 wine（178 条，3 类），**样本量很小**，"
        "指标波动比看上去大；\n"
        "2. 没做概率校准，概率不等于真实发生率；\n"
        "3. 输入范围超出训练数据分布时，模型会照样给一个答案，"
        "**不会报错也不会提示**——这是当前版本的明显短板。"
    )
