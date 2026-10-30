import unittest

from luban.thinking import ThinkingStream


class ThinkingTests(unittest.TestCase):
    def split(self, chunks):
        output = []
        stream = ThinkingStream(output.append)
        for chunk in chunks:
            stream.feed(chunk)
        stream.finish()
        return {key: "".join(part.get(key, "") for part in output)
                for key in ("content", "reasoning_content")}

    def test_tags_across_every_stream_boundary(self):
        text = "<think>先计算，再核验。</think>答案是5。"
        for width in (1, 2, 5, 11, len(text)):
            with self.subTest(width=width):
                result = self.split([text[i:i + width] for i in range(0, len(text), width)])
                self.assertEqual(result, {"content": "答案是5。", "reasoning_content": "先计算，再核验。"})

    def test_literal_partial_tag_and_comparison_are_preserved(self):
        self.assertEqual(self.split(["x <", "5; <thi"]),
                         {"content": "x <5; <thi", "reasoning_content": ""})

    def test_unfinished_thinking_has_no_invented_answer(self):
        self.assertEqual(self.split(["<thi", "nk>还在计算"]),
                         {"content": "", "reasoning_content": "还在计算"})

    def test_multiple_blocks_and_stray_closing_tag(self):
        self.assertEqual(self.split(["</think>答<think>想</think>案<think>验</think>。"]),
                         {"content": "答案。", "reasoning_content": "想验"})
