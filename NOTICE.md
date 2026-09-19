# NOTICE —— 第三方代码、数据与许可证

本仓库是《开源软件与新技术》实验04 的二次开发成品。
下面的库是**通过包管理器正常安装**的依赖，不是复制源码进本仓库；
数据集有单独的许可证，不能和代码许可证混为一谈。

## 一、上游项目

| 项目 | 来源链接 | 版本 | 许可证 | 在本项目里的角色 |
| --- | --- | --- | --- | --- |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | 1.7.2 | BSD-3-Clause | 提供 Pipeline / 算法 / 指标 |
| Streamlit | https://github.com/streamlit/streamlit | 1.50.0 | Apache-2.0 | Web 推理界面 |

**选择理由**：scikit-learn 的估计器接口统一，Pipeline 能把预处理和模型绑在一起，
正好用来解决本实验最核心的问题——训练和推理预处理不一致。
Streamlit 能把固定模型几行代码变成普通人能操作的界面，省掉前端工程。

## 二、数据集

| 数据集 | 来源 | 许可证 | 说明 |
| --- | --- | --- | --- |
| Wine | UCI Machine Learning Repository，https://archive.ics.uci.edu/dataset/109/wine | **CC BY 4.0** | 178 个样本，13 个数值特征，3 个类别 |

本项目通过 `sklearn.datasets.load_wine()` 读取该数据集的**内置副本**，
不联网下载。原始来源和许可证按 UCI 页面记录。

> **许可证边界**：scikit-learn 是 BSD-3-Clause，但 **Wine 数据本身是 CC BY 4.0**，
> 两者不能互相替代。如果本项目的成果要对外发布，
> 引用 Wine 数据时必须保留 UCI 的署名要求。

## 三、间接依赖

| 库 | 许可证 |
| --- | --- |
| NumPy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| joblib | BSD-3-Clause |
| pytest | MIT |

（具体版本以实际安装为准，见 `requirements.txt` 与 `pip freeze` 输出。）

## 四、代码来源声明

- `src/`、`app.py`、`tests/`、`scripts/` 全部为本人编写。
- `src/tezheng.py` 里 `goujian_yuchuli_tongyong()` 的 ColumnTransformer 写法
  参考了 scikit-learn 官方 User Guide 的 compose 章节（
  https://scikit-learn.org/stable/modules/compose.html ），
  按本实验的数据特点改写，不是整段复制。
- `src/xunlian.py` 里 `goujian_moxing_b()` 的随机森林超参数取的是
  scikit-learn 文档里的常用默认值（`n_estimators=300`），没有照抄任何示例代码。

## 五、模型文件

`models/model.joblib` 和 `models/model_meta.json` 是**本人用本仓库的
`src/xunlian.py` 在本地 CPU 上训练出来的**，不包含任何第三方预训练权重。

joblib/pickle 反序列化会执行代码，所以本应用**只加载自己训练并校验过的模型**，
不提供"上传模型文件"的入口。

## 六、本仓库的许可证

**MIT**，见 [LICENSE](LICENSE)，与上游 scikit-learn（BSD-3-Clause）兼容。
