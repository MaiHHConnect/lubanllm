"""Separate streamed thought tags, including tags split across token chunks."""
from __future__ import annotations


class ThinkingStream:
    def __init__(self, emit):
        self.emit = emit
        self.pending = ""
        self.thinking = False

    def _emit(self, text):
        if text:
            self.emit({"reasoning_content" if self.thinking else "content": text})

    def feed(self, text):
        self.pending += text
        tags = ("<think>", "</think>")
        while self.pending:
            positions = [(self.pending.find(tag), tag) for tag in tags if tag in self.pending]
            if positions:
                index, tag = min(positions)
                self._emit(self.pending[:index])
                self.pending = self.pending[index + len(tag):]
                self.thinking = tag == "<think>"
                continue
            keep = max((size for tag in tags for size in range(1, len(tag))
                        if self.pending.endswith(tag[:size])), default=0)
            self._emit(self.pending[:-keep] if keep else self.pending)
            self.pending = self.pending[-keep:] if keep else ""
            break

    def finish(self):
        self._emit(self.pending)
        self.pending = ""
