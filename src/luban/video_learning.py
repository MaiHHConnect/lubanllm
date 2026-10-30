"""Bounded local video frames for the existing image-teacher lesson, no model load."""
from __future__ import annotations

import json
import math
from pathlib import Path
import stat
import subprocess


def extract_frames(source: Path, directory: Path):
    """Keep one video's frames in one source family; never infer its sound/actions."""
    root = Path(__file__).resolve().parents[2]
    helper = next((path for path in (root / "native/luban-video-frames", root / "build/native/luban-video-frames")
                   if path.is_file() and not path.is_symlink()), None)
    if helper is None:
        raise ValueError("视频画面解码器未就绪，请更新应用")
    directory.mkdir(mode=0o700)
    source, directory = source.resolve(), directory.resolve()
    try:
        result = subprocess.run([str(helper), str(source), str(directory)], stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=35, check=False)
        if result.returncode != 0 or len(result.stdout) > 16384:
            raise ValueError("视频无法解码；请使用不超过30秒的MP4、MOV或受支持的视频")
        body = json.loads(result.stdout)
        duration, frames = body.get("duration_seconds"), body.get("frames")
        if (type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 30
                or not isinstance(frames, list) or not 1 <= len(frames) <= 4):
            raise ValueError("视频解码结果无效")
        from PIL import Image
        seen, total, output, last = set(), 0, [], -1.0
        for frame in frames:
            path, stamp = Path(frame["path"]), frame["timestamp_seconds"]
            if (type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0 or not last < stamp <= duration
                    or path.parent != directory or path.suffix != ".jpg" or path in seen
                    or path.resolve() != directory.resolve() / path.name):
                raise ValueError("视频画面路径或时间戳无效")
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError("视频画面必须私有保存")
            total += info.st_size
            if total > 4 * 1024 * 1024:
                raise ValueError("视频画面超过本轮大小限制")
            with Image.open(path) as image:
                if image.format != "JPEG" or not 1 <= min(image.size) <= max(image.size) <= 768:
                    raise ValueError("视频画面尺寸或格式无效")
                image.verify()
            seen.add(path); last = stamp
            output.append({"path": str(path), "timestamp_seconds": stamp})
        return output
    except (subprocess.SubprocessError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("视频画面提取失败；原文件已保留，不会发送教师请求") from error


def frame_rows(record):
    frames = record.get("video_frames")
    if not isinstance(frames, list) or not 2 <= len(frames) <= 4:
        raise ValueError("视频联合学习需要2至4张有序画面")
    last = -1.0
    for row in frames:
        stamp = row.get("timestamp_seconds") if isinstance(row, dict) else None
        if (type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= stamp <= 30
                or stamp <= last or not isinstance(row.get("asset_path"), str)):
            raise ValueError("视频画面路径或时间戳无效")
        last = stamp
    return frames


def teacher_content(record, storage_root):
    """One ordered sequence per paid attempt; all frames belong to this clip."""
    import base64
    import io
    from PIL import Image
    from .speech_learning import _read
    frames = frame_rows(record)
    root = Path(storage_root).resolve() / "app/media-lessons"
    source = Path(record["asset_path"])
    if not source.is_relative_to(root) or source.resolve() != source:
        raise ValueError("教师资料不在本批私有保存目录")
    directory = source.parent / (source.stem + "-frames")
    content = [{"type": "text", "text": "以下是同一视频按时间顺序抽取的画面。请联合画面和时间关系回答；"
                "只判断可见变化，不猜测未采样过程；未提供音轨。问题：" + record["question"]}]
    total = 0
    for row in frames:
        path = Path(row["asset_path"])
        if path.parent != directory or path.resolve() != path:
            raise ValueError("教师资料不在本批私有保存目录")
        raw = _read(path, 4 * 1024 * 1024)
        total += len(raw)
        if total > 4 * 1024 * 1024:
            raise ValueError("视频画面超过本轮大小限制")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or not 1 <= min(image.size) <= max(image.size) <= 768:
                raise ValueError("视频画面尺寸或格式无效")
            image.verify()
        content.extend([{"type": "text", "text": str(row["timestamp_seconds"]) + "秒"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")}}])
    return content
