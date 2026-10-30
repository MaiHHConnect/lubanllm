"""Reviewed private screen/action demonstrations for image-conditioned learning.

No recorder or executor. The existing image-learning owner performs the fixed
round; a saved demonstration is input data, never evidence of learned behaviour.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import stat

ROUND_STEPS = 5
MAX_BYTES = 272 * 1024 * 1024
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
ACTIONS = {"click", "type", "key", "scroll"}
KEYS = {"Enter", "Tab", "Escape", "Backspace", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _keys(value, names):
    _need(type(value) is dict and set(value) == set(names.split()), "示范字段缺失或含未知字段")


def _text(value, maximum=4096):
    _need(isinstance(value, str) and value.strip() and len(value) <= maximum
          and not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value), "示范文字无效")
    value.encode("utf-8", "strict")


def _id(value):
    _text(value, 160)
    _need(re.fullmatch(r"[A-Za-z0-9._:-]+", value), "示范标识无效")


def _integer(value, minimum, maximum=9007199254740991):
    _need(type(value) is int and minimum <= value <= maximum, "示范整数越界")


def _strings(value, maximum):
    _need(type(value) is list and 0 < len(value) <= maximum, "示范列表无效")
    for item in value:
        _id(item)
    _need(len(set(value)) == len(value), "示范列表重复")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _target(action):
    fields = {"click": ("x", "y", "button", "count"), "type": ("text",),
              "key": ("key",), "scroll": ("x", "y", "delta_x", "delta_y")}
    return {key: action[key] for key in ("task_id", "app_id", "type", *fields[action["type"]])}


def validate_demonstration(d):
    """Same recorder v1 contract as the accepted native action-sample helper."""
    _keys(d, "schema id session_id source source_family approved training_allowed rights_basis "
          "user_data_scope_confirmed sensitive_reviewed sensitive recording task outcome steps")
    _need(d["schema"] == "luban-operation-demonstration-v1" and d["source"] == "user_demonstration", "只接收真实录制示范")
    for key in ("id", "session_id", "source_family"):
        _id(d[key])
    _need(all(d[key] is True for key in ("approved", "training_allowed", "user_data_scope_confirmed", "sensitive_reviewed"))
          and d["sensitive"] is False, "示范须核对正确、许可且不含敏感信息")
    _text(d["rights_basis"])
    recording, task, outcome = d["recording"], d["task"], d["outcome"]
    _keys(recording, "start stop")
    for event in recording.values():
        _keys(event, "explicit at_ms")
        _need(event["explicit"] is True, "必须明确开始和停止录制")
        _integer(event["at_ms"], 0)
    start, stop = recording["start"]["at_ms"], recording["stop"]["at_ms"]
    _need(stop > start, "录制时间无效")
    _keys(task, "id family goal app_ids allowed_actions")
    _id(task["id"]); _id(task["family"]); _text(task["goal"])
    _strings(task["app_ids"], 16); _strings(task["allowed_actions"], 4)
    _need(set(task["allowed_actions"]) <= ACTIONS, "不支持的操作")
    _need(len(task["app_ids"]) == 1 and d["source_family"] == task["family"] == task["app_ids"][0], "录制任务沿用原软件家族，不自动捏造独立任务族")
    _keys(outcome, "task_id verified_by success at_ms evidence")
    _need(outcome["task_id"] == task["id"] and outcome["verified_by"] == "user" and outcome["success"] is True, "必须由用户核对任务成功")
    _integer(outcome["at_ms"], stop); _text(outcome["evidence"])
    _need(type(d["steps"]) is list and 0 < len(d["steps"]) <= 256, "示范轨迹为空或过长")
    seen, previous, total_rgb = set(), None, 0
    for step in d["steps"]:
        _keys(step, "observation action")
        frame, action = step["observation"], step["action"]
        _keys(frame, "id task_id app_id session_id captured_at_ms phase previous_action_id width height rgb reviewed sensitive")
        _need(type(action) is dict, "动作无效")
        _id(frame["id"]); _id(action.get("id"))
        _need(frame["id"] != action["id"] and not {frame["id"], action["id"]} & seen, "画面或动作标识重复")
        seen.update((frame["id"], action["id"]))
        _need(frame["task_id"] == action.get("task_id") == task["id"]
              and frame["app_id"] == action.get("app_id") == task["app_ids"][0]
              and frame["session_id"] == d["session_id"], "画面任务或软件不匹配")
        _need(frame["phase"] == "before_action" and frame["previous_action_id"] == (previous["id"] if previous else None)
              and frame["reviewed"] is True and frame["sensitive"] is False, "需要核对过的动作前画面及完整历史")
        _integer(frame["width"], 1, 8192); _integer(frame["height"], 1, 8192)
        _need(frame["width"] * frame["height"] <= 16000000 and type(frame["rgb"]) is list
              and len(frame["rgb"]) == frame["width"] * frame["height"] * 3, "需要完整RGB画面")
        total_rgb += len(frame["rgb"])
        _need(total_rgb <= 64 * 1024 * 1024, "示范画面超过上限")
        for byte in frame["rgb"]:
            _integer(byte, 0, 255)
        _integer(frame["captured_at_ms"], previous["finished_at_ms"] if previous else start, stop)
        _integer(action.get("started_at_ms"), frame["captured_at_ms"] + 1, stop)
        _integer(action.get("finished_at_ms"), action["started_at_ms"], stop)
        kind = action.get("type")
        _need(isinstance(kind, str) and kind in task["allowed_actions"] and action["started_at_ms"] - frame["captured_at_ms"] <= 2000, "动作超出许可或画面过期")
        fields = {"click": "x y button count", "type": "text sensitive", "key": "key", "scroll": "x y delta_x delta_y"}
        _keys(action, "id task_id app_id started_at_ms finished_at_ms type " + fields[kind])
        if kind == "click":
            _need(action["button"] in {"left", "right"} and type(action["count"]) is int and action["count"] in {1, 2}, "点击格式无效")
        elif kind == "type":
            _text(action["text"], 8192); _need(action["sensitive"] is False, "拒绝敏感输入")
        elif kind == "key":
            _need(action["key"] in KEYS, "不支持的按键")
        else:
            _integer(action["delta_x"], -8192, 8192); _integer(action["delta_y"], -8192, 8192)
            _need(action["delta_x"] or action["delta_y"], "滚动为空")
        if kind in {"click", "scroll"}:
            _integer(action["x"], 0, frame["width"] - 1); _integer(action["y"], 0, frame["height"] - 1)
        previous = action
    _need(outcome["at_ms"] >= previous["finished_at_ms"], "任务结果早于动作")
    return d


def _directory(home):
    path = os.fspath(home)
    _need(os.path.isabs(path) and "\0" not in path and not {".", ".."} & set(path.split("/")), "需要绝对私有目录")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for part in Path(path).parts[1:]:
            following = os.open(part, flags, dir_fd=fd)
            os.close(fd); fd = following
        _need(os.fstat(fd).st_uid == os.getuid(), "私有目录不属于当前用户")
        following = os.open("operation-demonstrations", flags, dir_fd=fd)
        os.close(fd); fd = following
        info = os.fstat(fd)
        _need(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700, "示范目录需要0700权限")
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)


def prepare_demonstrations(home, selected):
    """Read only explicit private UUID files, once, then build complete RGB+JSON."""
    _need(type(selected) is list and 2 <= len(selected) <= 1024, "需要独立训练和留出示范")
    seen_files, seen_demos, sessions, families, tasks, records, used = set(), set(), set(), {}, {}, [], 0
    directory = _directory(home)
    try:
        for item in selected:
            _keys(item, "file_id split")
            file_id, split = item["file_id"], item["split"]
            _need(isinstance(file_id, str) and UUID.fullmatch(file_id) and file_id not in seen_files
                  and isinstance(split, str) and split in {"TRAIN", "HOLDOUT"}, "示范UUID或固定拆分无效")
            seen_files.add(file_id)
            fd = os.open(file_id + ".json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
            try:
                before = os.fstat(fd)
                _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
                      and stat.S_IMODE(before.st_mode) == 0o600 and 0 < before.st_size <= MAX_BYTES, "示范必须是私有0600有界普通文件")
                used += before.st_size
                _need(used <= MAX_BYTES, "所选示范总量超过上限")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    raw = stream.read(before.st_size + 1)
                after = os.fstat(fd)
                fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
                _need(len(raw) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in fields), "示范入场读取时变化")
                def unique(pairs):
                    result = {}
                    for key, value in pairs:
                        _need(key not in result, "示范JSON字段重复")
                        result[key] = value
                    return result
                d = validate_demonstration(json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=unique,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("示范含非有限值"))))
            finally:
                os.close(fd)
            _need(d["id"] not in seen_demos and d["session_id"] not in sessions, "示范或录制会话重复")
            seen_demos.add(d["id"]); sessions.add(d["session_id"])
            for mapping, key in ((families, d["source_family"]), (families, d["task"]["family"]), (tasks, d["task"]["id"])):
                _need(key not in mapping or mapping[key] == split, "同软件或任务家族不能跨训练和留出；当前不支持同软件任务族独立证明")
                mapping[key] = split
            history = []
            for step in d["steps"]:
                frame, action = step["observation"], step["action"]
                prefix = {"instruction": "Predict the next complete action JSON for the observed screen and task.",
                    "image": "attached_pre_action_rgb", "coordinate_system": "image_pixels_top_left",
                    "screen": {"width": frame["width"], "height": frame["height"]}, "task": d["task"],
                    "app_id": frame["app_id"], "history": copy.deepcopy(history)}
                records.append({"id": d["id"] + ":" + action["id"], "kind": "operation", "split": "training" if split == "TRAIN" else "validation", "family": d["source_family"],
                    "image": {"width": frame["width"], "height": frame["height"], "rgb": bytes(frame["rgb"])},
                    "question": _canonical(prefix), "answer": _canonical(_target(action)),
                    "provenance": {"kind": "operation", "file_id": file_id, "demonstration_id": d["id"],
                        "session_id": d["session_id"], "rights_basis": d["rights_basis"], "user_verified": True,
                        "saved_file_stat": {key: getattr(before, key) for key in fields},
                        "training_allowed": True, "sensitive": False}})
                history.append(_target(action))
        _need({r["split"] for r in records} == {"training", "validation"}, "训练和留出均不能为空")
        return records
    finally:
        os.close(directory)


def train_round(runtime, samples, saved_candidate=None, cancel=None, progress=None):
    """Use the one image owner/job; never interpret action logs as text training."""
    _need(type(samples) is list and samples and all(type(s) is dict and s.get("kind") == "operation" for s in samples),
          "操作训练需要完整图像条件操作示范")
    from .image_learning import train_round as image_round
    return image_round(runtime, samples, saved_candidate=saved_candidate, cancel=cancel, progress=progress)


def capability(runtime):
    from .image_learning import capability as image_capability
    result = dict(image_capability(runtime))
    result.update(kind="operation", round_steps=ROUND_STEPS, external_execution=False,
                  same_app_task_holdout=False)
    return result
