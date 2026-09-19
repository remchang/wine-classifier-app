# 个人开发过程与 Git 证据对照表（实验04）

仓库：https://github.com/remchang/wine-classifier-app

---

## 一、Issue 与任务拆解

| Issue | 标题 | 对应阶段 | 关联 Commit |
| --- | --- | --- | --- |
| #1 | 任务定义与数据基线 | 基线 | `chore(baseline)` |
| #2 | 基线模型与划分策略 | 核心功能 | `feat(pipeline)` |
| #3 | 两种算法比较与主指标解释 | 核心功能 | `feat(train)` |
| #4 | 模型保存、元数据与推理应用 | 核心功能 | `feat(infer)` / `feat(app)` |
| #5 | 自主功能：置信度阈值 + 批量预测 | 自主功能 | `feat(ext)` |
| #6 | 测试、模型卡与文档 | 测试 / 文档 | `test(tests)` / `docs` |

Issue 正文留档在 `docs/issue.md`。

## 二、Commit 对照表

| 序号 | 类型 | 覆盖阶段 | 主要内容 |
| --- | --- | --- | --- |
| 1 | `chore(baseline)` | 基线 | 依赖、许可证、NOTICE（区分代码与数据许可）、基线说明 |
| 2 | `feat(pipeline)` | 核心功能 | `shuju.py`（Schema/划分/防泄漏）、`tezheng.py`（Pipeline 构造） |
| 3 | `feat(train)` | 核心功能 | `xunlian.py`：CV 比较两算法 → 按 macro F1 选优 → 一次测试 → 保存 |
| 4 | `feat(infer)` | 核心功能 | `yuce.py`：加载校验、单条/批量推理 |
| 5 | `feat(app)` | 核心功能 | `app.py`：Streamlit 四个标签页 |
| 6 | `feat(ext)` | 自主功能 | 置信度阈值 + 批量 CSV + 阈值分析页 |
| 7 | `test(tests)` | 测试 | 40 条 pytest，含假数据与非侵入式训练 fixture |
| 8 | `feat(scripts)` | 交付 | 安装/训练/启动/测试一键脚本 |
| 9 | `docs` | 文档 | README、模型卡、实验报告、测试记录 |

覆盖实验要求的五类：**基线 / 核心功能 / 自主功能 / 测试 / 文档**。

## 三、分支与 PR

```powershell
git switch -c feature/model-application
# …开发…
git push -u origin feature/model-application
```

- 特性分支：`feature/model-application`
- PR 关联 Issue #4 #5 #6，描述里附了 `pytest` 输出和阈值分析表
- 合并前完成一次有文字记录的自我 Code Review（见下节）

验证命令：

```powershell
git log --oneline --graph --decorate --all -n 30
git shortlog -sne HEAD
```

## 四、自我 Code Review 检查清单（合并前逐项过）

| # | 检查项 | 结论 |
| --- | --- | --- |
| 1 | 预处理器是在全量数据上 fit 的吗？ | ✅ 不是。全部包在 Pipeline 内，`cross_validate` 每折单独拟合 |
| 2 | 测试集被用了多少次？ | ✅ 只有一次，代码结构上 `zui_zhong_pingjia()` 不在任何循环里 |
| 3 | 划分单位想清楚了吗？会不会有主体跨集合？ | ✅ 想清楚了。wine 每行独立，无需分组；判断依据写在 `shuju.py` 注释 |
| 4 | 界面传的字典顺序变了会不会出错？ | ✅ 不会。`zu_canshu_zi()` 按训练列序重建 DataFrame，有专门测试 |
| 5 | 保存的是完整 Pipeline 还是分开的？ | ✅ 完整 Pipeline 一起 dump，不可能忘记做预处理 |
| 6 | 加载时校验了什么？ | ✅ scikit-learn 主版本、特征顺序、标签映射；不一致直接拒 |
| 7 | 元数据里的 SHA256 是真的算出来的吗？ | ✅ 有一条测试拿真实文件重算比对 |
| 8 | 批量预测一行坏了会不会整批失败？ | ✅ 不会，逐行 try/except，行号和原因都报出来 |
| 9 | 自主功能有没有量化证据？ | ✅ 有。阈值分析页给出 8 个阈值下的覆盖率与准确率 |
| 10 | 有没有把概率说成"确定性"？ | ✅ 没有。界面和模型卡都写了"未做概率校准" |
| 11 | 会不会加载用户上传的模型？ | ✅ 不会，界面上没有这个入口（pickle 反序列化会执行代码） |
| 12 | 报告里有没有把"可能更好"写成"更好"？ | ✅ 没有。明确写了样本量太小时算法差异可能在噪声范围内 |
| 13 | 数据许可证和代码许可证分开了吗？ | ✅ 分开了。NOTICE 里明确 Wine 是 CC BY 4.0，代码是 MIT/BSD |
| 14 | 测试能不能从一个干净克隆跑通？ | ✅ 能。conftest 现场训练一个模型，不依赖 `models/` 目录 |

## 五、踩过的坑

1. **差点写成在全量数据上 fit scaler**。
   写完第一版划分代码才反应过来，如果先把 scaler 应用于整个 X 再划分，
   就是在测试集上做过拟合。改成 Pipeline 之后才安全。
2. **字典顺序问题**。最初 `yuce_dan_tiao` 直接用 `pd.DataFrame([zidian])`，
   列顺序就是字典顺序。用户从界面输入的字段顺序不保证和训练一致——
   这会导致预测静默出错。补了 `zu_canshu_zi()` 强制按 `lie_ming` 排序。
3. **测试依赖 `models/` 目录**。第一版测试直接加载仓库里的模型文件，
   在一个干净克隆上跑必然失败。改成在 conftest 里现场训练。
4. **置信度阈值的定义自洽性**。写完之后想验证"阈值有没有写反"，
   于是加了一条断言：阈值 0 时的自动准确率必须等于整体 accuracy。
   这条断言同时验证了定义和实现。
5. **依赖版本漂移**。`requirements.txt` 一开始按文档写了推测的版本号
   （scikit-learn 1.7.2 / pandas 2.3.3），实际安装到的是别的版本。
   改成按实际 `pip freeze` 的结果写，并把差异记在测试记录里。
