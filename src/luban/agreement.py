"""Versioned local usage agreement; acceptance does not grant third-party rights."""
from __future__ import annotations

from datetime import datetime

AGREEMENT_VERSION = "2026-10-09"
AGREEMENT_TEXT = """鲁班零号使用协议
版本：2026-10-09

1. 数据与第三方权利
您自行确认输入、保存、训练和使用的数据，以及教师服务输出的使用方式，符合适用法律、第三方权利和对应服务条款。点击同意本协议不授予任何第三方数据、模型或教师输出的权利，也不表示本应用已核实供应商许可。

2. 教师服务与费用
您自行选择和配置教师服务，处理其账号、调用额度、费用及服务条款。请求走教师路由时，相关对话上下文会发送至您配置的教师地址；教师可能按其自身规则处理这些数据。

3. 本地保存与学习
本应用在配置的本地存储目录保存对话、教师回答、使用量、学习材料和模型模块；接口密钥及本协议接受记录保存在本机私有应用目录。接受本协议后，符合当前结构检查的完整教师回答可自动进入学习或检验材料；这些检查不证明事实正确、供应商许可或第三方权利。您自行管理材料并决定学习操作。

4. 输出核验与研究范围
模型输出可能错误或不完整，您应按实际用途核验内容后再使用。“无限进化”是研究愿景，不是已证明的能力或效果保证。当前版本不保证新增学习在所有任务上有效。

5. 责任范围
您对自己选择的数据、教师服务、学习材料及输出用途负责。本协议不排除或限制适用法律规定不能免除的责任和义务，不影响您依法享有的权利。

点击“我已阅读并同意”表示您已阅读并接受此版本协议。"""


def agreement_status(data):
    record = data.get("agreement", {})
    accepted = False
    if isinstance(record, dict) and record.get("version") == AGREEMENT_VERSION:
        try:
            accepted = datetime.fromisoformat(record["accepted_at"]).tzinfo is not None
        except (KeyError, TypeError, ValueError):
            pass
    return {"version": AGREEMENT_VERSION, "accepted": accepted, "text": AGREEMENT_TEXT}
