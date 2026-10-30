export const AGREEMENT_VERSION = "2026-10-09";
// Python datetime.fromisoformat accepts basic/week dates and optional seconds.
function awareISO(value: unknown): boolean {
  if (typeof value !== "string") return false;
  const m = /^(\d{4}-\d{2}-\d{2}|\d{8}|\d{4}-W\d{2}(?:-[1-7])?|\d{4}W\d{2}[1-7]?)[\s\S](\d{2}(?::\d{2}(?::\d{2})?|(?:\d{2}){1,2})?(?:[.,]\d+)?)(Z|[+-]\d{2}(?::\d{2}(?::\d{2})?|(?:\d{2}){1,2})?(?:[.,]\d+)?)$/u.exec(value);
  if (!m) return false;
  const date = m[1]!, year = Number(date.slice(0, 4));
  const time = m[2]!.split(/[.,]/, 1)[0]!.replaceAll(":", "");
  if (year < 1 || Number(time.slice(0, 2)) > 23 || Number(time.slice(2, 4) || 0) > 59 || Number(time.slice(4, 6) || 0) > 59) return false;
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  if (!date.includes("W")) {
    const digits = date.replaceAll("-", ""), month = Number(digits.slice(4, 6)), day = Number(digits.slice(6, 8));
    if (month < 1 || month > 12 || day < 1 || day > [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]!) return false;
  } else {
    const jan1 = new Date(0); jan1.setUTCFullYear(year, 0, 1);
    const weekday = jan1.getUTCDay() || 7;
    const weeks = weekday === 4 || (weekday === 3 && leap) ? 53 : 52;
    const parts = date.replaceAll("-", "").split("W")[1]!;
    const week = Number(parts.slice(0, 2)), day = Number(parts.slice(2) || 1);
    if (week < 1 || week > weeks) return false;
    // A final ISO week can extend into year 10000, outside datetime's range.
    if (year === 9999 && week === 52 && day > 5) return false;
  }
  const zone = m[3]!;
  if (zone !== "Z") {
    const parts = zone.slice(1).split(/[.,]/, 1)[0]!.replaceAll(":", "");
    // fromisoformat normalizes offset minutes/seconds but rejects a full day.
    if (Number(parts.slice(0, 2)) * 3600 + Number(parts.slice(2, 4) || 0) * 60 + Number(parts.slice(4, 6) || 0) >= 86400) return false;
  }
  return true;
}

export function agreementStatus(data: Record<string, unknown>) {
  const record = data.agreement;
  const accepted = record !== null && typeof record === "object" && !Array.isArray(record)
    && (record as Record<string, unknown>).version === AGREEMENT_VERSION
    && awareISO((record as Record<string, unknown>).accepted_at);
  return { version: AGREEMENT_VERSION, accepted: Boolean(accepted), text: AGREEMENT_TEXT };
}
export const agreement_status = agreementStatus;
export const AGREEMENT_TEXT = "鲁班零号使用协议\n版本：2026-10-09\n\n1. 数据与第三方权利\n您自行确认输入、保存、训练和使用的数据，以及教师服务输出的使用方式，符合适用法律、第三方权利和对应服务条款。点击同意本协议不授予任何第三方数据、模型或教师输出的权利，也不表示本应用已核实供应商许可。\n\n2. 教师服务与费用\n您自行选择和配置教师服务，处理其账号、调用额度、费用及服务条款。请求走教师路由时，相关对话上下文会发送至您配置的教师地址；教师可能按其自身规则处理这些数据。\n\n3. 本地保存与学习\n本应用在配置的本地存储目录保存对话、教师回答、使用量、学习材料和模型模块；接口密钥及本协议接受记录保存在本机私有应用目录。接受本协议后，符合当前结构检查的完整教师回答可自动进入学习或检验材料；这些检查不证明事实正确、供应商许可或第三方权利。您自行管理材料并决定学习操作。\n\n4. 输出核验与研究范围\n模型输出可能错误或不完整，您应按实际用途核验内容后再使用。“无限进化”是研究愿景，不是已证明的能力或效果保证。当前版本不保证新增学习在所有任务上有效。\n\n5. 责任范围\n您对自己选择的数据、教师服务、学习材料及输出用途负责。本协议不排除或限制适用法律规定不能免除的责任和义务，不影响您依法享有的权利。\n\n点击“我已阅读并同意”表示您已阅读并接受此版本协议。";
