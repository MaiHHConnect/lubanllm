import unittest
from luban.name_filter import mask_name

class NameFilterTests(unittest.TestCase):
    def test_normal_names_and_masks_remain(self):
        for name in ['vinson','鲁班零号','林浩','学习者','***玩家','Player13','Player14','进化100']:
            self.assertEqual(mask_name(name),name)
    def test_comment_and_advertising_words_are_masked(self):
        for name,expected in [('习大大','***'),('习-大-大','***'),('最强玩家','***玩家'),('傻-逼用户','***用户'),('ＦＵＣＫ','***'),('世界第一人','***人'),('色情玩家','***玩家'),('İ最强','İ***'),('😀最强','😀***')]:
            with self.subTest(name=name):self.assertEqual(mask_name(name),expected)
    def test_masking_is_idempotent(self):
        for name in ['最强第一','傻 逼玩家','ＦＵＣＫ最强','İ最强']:
            first=mask_name(name);self.assertEqual(mask_name(first),first)
