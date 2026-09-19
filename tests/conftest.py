# -*- coding: utf-8 -*-
"""conftest.py —— 让测试能 import 到 src/ 下的模块，并提供共用 fixture。"""
import os
import sys

GEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(GEN, "src")
for p in (SRC, GEN):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from shuju import huafen_shuju, jiazai_shuju  # noqa: E402


@pytest.fixture(scope="session")
def yuanshi_shuju():
    """原始数据，整个测试会话加载一次就够。"""
    tb, bq, lm = jiazai_shuju()
    return tb, bq, lm


@pytest.fixture(scope="session")
def huafende(yuanshi_shuju):
    tb, bq, lm = yuanshi_shuju
    return huafen_shuju(tb, bq)


@pytest.fixture(scope="session")
def xunlian_haode(tmp_path_factory):
    """
    完整跑一次训练，把模型存到临时目录。

    为什么不在测试里加载仓库里的 models/：
        那个目录可能还没训练过（新克隆的仓库就是空的），
        测试会莫名其妙地失败。这里现场训一个，测试就自洽了。
        训练只要几秒，session 级别只跑一次。
    """
    from xunlian import shiji_xunlian

    mulu = tmp_path_factory.mktemp("moxing")
    jieguo = shiji_xunlian(shuchu_mulu=str(mulu))
    return str(mulu), jieguo


@pytest.fixture(scope="session")
def jiazai_haode(xunlian_haode):
    """加载上面训练出来的模型，返回 (pipeline, meta)。"""
    from yuce import jiazai_moxing

    mulu, _ = xunlian_haode
    return jiazai_moxing(mulu)
