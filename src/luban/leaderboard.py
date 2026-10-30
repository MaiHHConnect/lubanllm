"""Opt-in automatic branch-upgrade ranking; installation credentials stay private."""
from __future__ import annotations

import copy
import json
import math
import re
import secrets
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path

import httpx

from .settings import private_json
from .name_filter import mask_name
from .upgrades import read_upgrade_parameter_total, read_upgrade_records

SITE = "https://ai.maihh.net"
PROTOCOL_VERSION = 2
COUNTRY_CODES = frozenset("AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW".split())
COUNTRY_TRACE = "https://www.cloudflare.com/cdn-cgi/trace"


def validated_country(value):
    if value is not None and (not isinstance(value, str) or value not in COUNTRY_CODES):
        raise ValueError("国家/地区需要有效的 ISO 两字母代码")
    return value


def detect_country(transport=None):
    """Only consume loc; do not retain the trace IP or send credentials."""
    try:
        with httpx.Client(transport=transport, timeout=3, follow_redirects=False) as client:
            response = client.get(COUNTRY_TRACE)
        if response.status_code != 200 or len(response.content) > 4096:
            return None
        locations = [line[4:] for line in response.text.splitlines() if line.startswith("loc=")]
        return validated_country(locations[0]) if len(locations) == 1 else None
    except (httpx.RequestError, ValueError):
        return None


def normalized_username(value, limit=16):
    if not isinstance(value, str):
        raise ValueError("用户名需要文本")
    username = unicodedata.normalize("NFKC", value).strip()
    if not 1 <= len(username) <= limit or any(unicodedata.category(c).startswith("C") for c in username):
        raise ValueError(f"用户名需要 1–{limit} 个字符，不能含控制字符")
    username = mask_name(username)
    if len(username) > limit and limit == 32:
        return "***"  # Legacy records remain editable even when masking expands them.
    if len(username) > limit:
        raise ValueError(f"用户名屏蔽后超过 {limit} 个字符，请缩短后重试")
    return username


def validated_profile(body):
    if not isinstance(body, dict) or "username" not in body or set(body) - {"username", "country_code"}:
        raise ValueError("排行设置只接受用户名和国家/地区，升级次数由模型记录自动统计")
    return {"username": normalized_username(body["username"]),
            **({"country_code": validated_country(body["country_code"])} if "country_code" in body else {})}


def validated_entry(body, ranked=True):
    entry = {"username": normalized_username(body["username"], limit=32)}
    count = body.get("evolution_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 1_000_000_000:
        raise ValueError("无效的排行次数")
    entry["evolution_count"] = count
    entry["country_code"] = validated_country(body.get("country_code"))
    token_total = body.get("token_total")
    if token_total is not None and (type(token_total) is not int or not 0 <= token_total <= 10**15):
        raise ValueError("无效的累计 Token")
    entry["token_total"] = token_total
    parameter_total = body.get("parameter_total")
    if parameter_total is not None and (not isinstance(parameter_total, str) or not re.fullmatch(r"0|[1-9][0-9]{0,24}", parameter_total)):
        raise ValueError("无效的累计进化参数")
    entry["parameter_total"] = parameter_total
    for key in ("benchmark", "generalization", "benchmark_delta", "generalization_delta"):
        value = body.get(key)
        lower = -100 if key.endswith("_delta") else 0
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not lower <= value <= 100):
            raise ValueError("无效的评测成绩")
        entry[key] = value
    for key in ("suite_id", "model_version"):
        value = body.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 128):
            raise ValueError("无效的评测版本")
        entry[key] = value
    if "growth" in body or "growth_profile" in body:
        from .growth_service import validated_growth, DOMAINS
        entry["growth"] = validated_growth(body["growth"], public=True) if body.get("growth") else None
        profiles = body.get("growth_profile", [])
        if (not isinstance(profiles, list) or len(profiles) != 3
                or {p.get("domain") for p in profiles if isinstance(p, dict)} != set(DOMAINS)):
            raise ValueError("个人能力画像领域不完整")
        entry["growth_profile"] = [{"domain": p["domain"], "growth": validated_growth(p["growth"], public=True)
                                   if p.get("growth") else None} for p in profiles]
        if any(p["growth"] and p["growth"]["domain"] != p["domain"] for p in entry["growth_profile"]):
            raise ValueError("个人能力画像领域不匹配")
    if ranked:
        rank = body.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("无效的排行名次")
        entry["rank"] = rank
    return entry


class Leaderboard:
    def __init__(self, home: Path, transport=None, storage_root=None, country_transport=None,
                 token_provider=None, evaluation_provider=None):
        self.path = Path(home) / "leaderboard.json"
        self.lock = threading.RLock()
        self.transport = transport
        self.country_transport = country_transport
        self.storage_root = storage_root or (lambda: Path(home))
        self.storage_error = None
        self.token_provider = token_provider
        self.evaluation_provider = evaluation_provider
        self.growth_provider = None
        self.country_retry_after = 0.0
        self.data = {"protocol_version": PROTOCOL_VERSION, "username": "", "token": "", "published_profile": None,
                     "country_code": None, "country_detected": False, "country_manual": False}
        try:
            if self.path.exists():
                saved = json.loads(self.path.read_text())
                username = saved.get("username", "")
                profile = {"username": ""} if username == "" else {"username": normalized_username(username, limit=32)}
                token = saved.get("token", "")
                if not isinstance(token, str) or (token and not re.fullmatch(r"[0-9a-f]{64}", token)):
                    raise ValueError("Invalid installation credential")
                migrated = {**self.data, **profile, "token": token}
                if "country_code" in saved:
                    migrated["country_code"] = validated_country(saved["country_code"])
                    detected = saved.get("country_detected", True)
                    if not isinstance(detected, bool):
                        raise ValueError("Invalid country detection status")
                    manual = saved.get("country_manual", False)
                    if not isinstance(manual, bool):
                        raise ValueError("Invalid country selection status")
                    migrated["country_manual"] = manual
                    # Older versions marked failed lookups as completed, leaving null forever.
                    migrated["country_detected"] = detected and (manual or migrated["country_code"] is not None)
                if saved.get("protocol_version") == PROTOCOL_VERSION:
                    published = saved.get("published_profile")
                    migrated["published_profile"] = validated_entry(published, ranked=False) if published else None
                # Drop all legacy self-reported counts and snapshots; preserve installation identity.
                if saved != migrated:
                    self._save(migrated)
                else:
                    self.data = migrated
        except RuntimeError as error:
            self.storage_error = str(error)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            self.storage_error = "本地排行资料不可读或损坏，请保留 leaderboard.json 文件并检查"

    def records(self):
        root = self.storage_root() if callable(self.storage_root) else self.storage_root
        return read_upgrade_records(root)

    def profile(self, records=None):
        with self.lock:
            result = {"username": self.data["username"], "country_code": self.data["country_code"], "published_profile": copy.deepcopy(self.data["published_profile"]),
                      "evolution_count": None, "count_error": self.storage_error}
            if not self.storage_error:
                try:
                    result["evolution_count"] = len(self.records() if records is None else records)
                except RuntimeError as error:
                    result["count_error"] = str(error)
            result["token_total"] = self.token_total()
            if not self.storage_error:
                try:
                    result["parameter_total"] = str(read_upgrade_parameter_total(self.storage_root() if callable(self.storage_root) else self.storage_root))
                except RuntimeError as error:
                    result["parameter_total"] = None
                    result["count_error"] = str(error)
            else:
                result["parameter_total"] = None
            result.update(self.evaluation_provider() if self.evaluation_provider else {})
            return result

    def token_total(self):
        if self.token_provider is None:
            return None
        try:
            value = self.token_provider()
            return value if type(value) is int and 0 <= value <= 10**15 else None
        except (OSError, RuntimeError, ValueError, AttributeError, TypeError, sqlite3.Error):
            return None

    def ensure_token(self):
        with self.lock:
            if self.storage_error:
                raise RuntimeError(self.storage_error)
            if not self.data["token"]:
                self._save({**self.data, "token": secrets.token_hex(32)})

    def _save(self, data):
        if self.storage_error:
            raise RuntimeError(self.storage_error)
        try:
            private_json(self.path, data)
        except OSError:
            raise RuntimeError("本地排行资料保存失败，原设置已保留；已发送的操作可重试确认") from None
        self.data = data

    def save_profile(self, body):
        profile = validated_profile(body)
        with self.lock:
            self._save({**self.data, **profile, **({"country_detected": True, "country_manual": True} if "country_code" in profile else {})})
            return {"profile": self.profile()}

    def _request(self, method, authenticated=False, body=None, page=1, sort=None, endpoint=None, domain=None):
        headers = {"Authorization": "Bearer " + self.data["token"]} if authenticated else {}
        path = endpoint or ("/api/leaderboard/me" if authenticated else "/api/leaderboard")
        try:
            with httpx.Client(transport=self.transport, timeout=12, follow_redirects=False) as client:
                response = client.request(method, SITE + path, headers=headers, json=body,
                                          params={"page": page, **({"sort": sort} if sort else {}), **({"domain": domain} if domain else {})} if method == "GET" else None)
        except httpx.RequestError:
            raise RuntimeError("无法连接全球排行榜，本地设置已保留，请稍后重试") from None
        if not response.is_success:
            messages = {409: "用户名或升级记录已被其他安装使用，请检查后重试", 429: "操作过于频繁，请稍后重试",
                        400: "官网拒绝了排行资料，请检查用户名和升级记录",
                        401: "排行凭据未通过验证，请保留本地文件并检查"}
            raise RuntimeError(messages.get(response.status_code, "全球排行榜暂不可用，本地设置已保留"))
        try:
            return response.json()
        except (ValueError, UnicodeError):
            raise RuntimeError("全球排行榜返回格式异常，本地设置已保留") from None

    def read(self, page=1, sort="benchmark", domain="mathematics"):
        if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= 999999:
            raise ValueError("页码需要是 1–999999 的整数")
        if sort not in {"growth", "benchmark", "generalization", "evolution_count"}:
            raise ValueError("无效的排行排序")
        if domain not in {"mathematics", "programming", "data"}:
            raise ValueError("无效的成长领域")
        with self.lock:
            if (not self.storage_error and not self.data["country_manual"] and self.data["country_code"] is None
                    and time.monotonic() >= self.country_retry_after):
                self.country_retry_after = time.monotonic() + 60
                try:
                    code = detect_country(self.country_transport)
                    if code is not None:
                        self._save({**self.data, "country_code": code, "country_detected": True})
                except RuntimeError:
                    pass  # Optional detection never blocks reporting; failed saves remain retryable.
        result = {"profile": self.profile(), "entries": [], "total": 0, "updated_at": None, "error": None, "page": page, "page_size": 100, "sort": sort, "domain": domain}
        try:
            data = self._request("GET", page=page, sort=sort, domain=domain)
            if (type(data.get("protocol_version")) is not int or data["protocol_version"] != PROTOCOL_VERSION
                    or data.get("measurement") != "accepted_branch_upgrade_records"):
                raise RuntimeError("全球排行榜协议不兼容，暂不显示旧版自报次数，请稍后重试")
            if sort == "growth" and (data.get("sort") != sort or data.get("domain") != domain):
                raise ValueError("成长排行返回了不同领域或协议")
            entries = data["entries"]
            total = data["total"]
            if (not isinstance(entries, list) or len(entries) > 100 or isinstance(total, bool) or not isinstance(total, int) or total < 0
                    or type(data.get("page", page)) is not int or data.get("page", page) != page
                    or type(data.get("page_size", 100)) is not int or data.get("page_size", 100) != 100):
                raise ValueError("Invalid ranking")
            result.update(entries=[validated_entry(entry) for entry in entries], total=total,
                          updated_at=data.get("updated_at"))
        except RuntimeError as error:
            result["error"] = str(error)
        except (ValueError, KeyError, TypeError, AttributeError):
            result["error"] = "全球排行榜返回格式异常，请稍后重试"
        return result

    def publish(self, evaluation=None, growth=None, core_version=None):
        with self.lock:
            if self.storage_error:
                raise RuntimeError(self.storage_error)
            profile = validated_profile({"username": self.data["username"], "country_code": self.data["country_code"]})
            records = self.records()  # Verify the entire history before any network write.
            self.ensure_token()
            total = self.token_total()
            extra = {"token_total": total} if total is not None else {}
            parameter_total = read_upgrade_parameter_total(self.storage_root() if callable(self.storage_root) else self.storage_root)
            if parameter_total:
                extra["parameter_total"] = str(parameter_total)
            if evaluation:
                extra.update(model_version=evaluation["model_version"], evaluation_receipt_id=evaluation["receipt_id"])
            elif core_version:
                extra["model_version"] = core_version
            if growth is None and self.growth_provider:
                growth = self.growth_provider()
            if growth:
                extra.update(growth_model_version=growth["model_version"], growth_receipt_id=growth["receipt_id"])
            entry = None
            for offset in range(0, max(1, len(records)), 50):
                response = self._request("PUT", authenticated=True,
                                         body={**profile, **extra, "upgrades": records[offset:offset + 50]})
                try:
                    if type(response.get("protocol_version")) is not int or response["protocol_version"] != PROTOCOL_VERSION:
                        raise ValueError("Incompatible leaderboard protocol")
                    entry = validated_entry(response["entry"])
                    if (entry["username"] != profile["username"] or entry["country_code"] != profile["country_code"]
                            or entry["evolution_count"] < min(offset + 50, len(records))):
                        raise ValueError("Profile mismatch")
                except (ValueError, KeyError, TypeError, AttributeError):
                    raise RuntimeError("上报结果格式异常；官网可能已接收，可重试上报确认") from None
                self._save({**self.data, "published_profile": {k: v for k, v in entry.items() if k != "rank"}})
            return {"profile": self.profile(records), "entry": entry}

    def unpublish(self):
        with self.lock:
            if self.storage_error:
                raise RuntimeError(self.storage_error)
            # Also delete after a failed publish: the remote may have committed before a timeout.
            if self.data["token"]:
                response = self._request("DELETE", authenticated=True)
                if not isinstance(response, dict) or response.get("ok") is not True:
                    raise RuntimeError("退出结果格式异常，原发布记录已保留；请重试确认")
            self._save({**self.data, "published_profile": None})
            return {"profile": self.profile()}
