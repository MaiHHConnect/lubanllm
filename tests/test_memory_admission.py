"""Memory admission uses the OS available estimate plus free swap."""
import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from luban import context


class MemoryAdmissionTests(unittest.TestCase):
    def mach(self, *, free=100, inactive=800, external=300, speculative=20,
             purgeable=0, count=40, result=0):
        def read(host, flavor, out, size):
            self.assertEqual((host, flavor), (12, 4))
            stats = ctypes.cast(out, ctypes.POINTER(context._VMStatistics64)).contents
            for name, value in dict(free=free, inactive=inactive, external=external,
                                    speculative=speculative, purgeable=purgeable).items():
                setattr(stats, name, value)
            ctypes.cast(size, ctypes.POINTER(ctypes.c_uint32)).contents.value = count
            return result
        api = SimpleNamespace(mach_host_self=Mock(return_value=12),
                              host_statistics64=Mock(side_effect=read),
                              mach_port_deallocate=Mock(return_value=0))
        for mock in (patch("luban.context._mach_memory_api", return_value=(api, 3)),
                     patch("luban.context.os.sysconf", return_value=16384)):
            mock.start()
            self.addCleanup(mock.stop)
        return api

    def test_inactive_anonymous_allowance_is_bounded_and_speculative_not_counted_twice(self):
        api = self.mach(purgeable=500)
        self.assertEqual(context._darwin_memory_bytes(), 380 * 16384)
        # Purgeable may overlap inactive; do not add it on top.
        api.mach_port_deallocate.assert_called_once_with(3, 12)

    def test_no_file_pages_does_not_credit_inactive_and_all_file_pages_preserve_old_limit(self):
        for external, expected in ((0, 100), (20, 100), (820, 900)):
            with self.subTest(external=external):
                self.mach(external=external)
                self.assertEqual(context._darwin_memory_bytes(), expected * 16384)

    def test_old_revision_allowed_but_short_or_failed_stats_release_host(self):
        for count, result, valid in ((38, 0, True), (37, 0, False), (41, 0, False), (40, 5, False)):
            with self.subTest(count=count, result=result):
                api = self.mach(count=count, result=result)
                if valid:
                    self.assertGreater(context._darwin_memory_bytes(), 0)
                else:
                    with self.assertRaises(OSError):
                        context._darwin_memory_bytes()
                api.mach_port_deallocate.assert_called_once_with(3, 12)

    def test_invalid_page_counts_and_port_errors_do_not_supply_a_budget(self):
        api = self.mach(speculative=101)
        with self.assertRaises(OSError):
            context._darwin_memory_bytes()
        api.mach_port_deallocate.assert_called_once_with(3, 12)
        api = self.mach()
        api.mach_port_deallocate.return_value = 5
        with self.assertRaises(OSError):
            context._darwin_memory_bytes()
        api = self.mach()
        api.mach_host_self.return_value = 0
        with self.assertRaises(OSError):
            context._darwin_memory_bytes()
        api.mach_port_deallocate.assert_not_called()

    def test_failed_telemetry_uses_only_free_memory_and_never_increases_old_estimate(self):
        memory = SimpleNamespace(available=900, free=80)
        with patch("luban.context.sys.platform", "darwin"), \
                patch("luban.context.psutil.virtual_memory", return_value=memory), \
                patch("luban.context.psutil.swap_memory", return_value=SimpleNamespace(free=0)), \
                patch("luban.context._darwin_memory_bytes", side_effect=[400, 1200, OSError("unavailable"), 0]):
            self.assertEqual([context.available_memory_bytes() for _ in range(4)], [900, 1200, 900, 900])

    def test_non_mac_keeps_platform_estimate_without_mach(self):
        with patch("luban.context.sys.platform", "linux"), \
                patch("luban.context.psutil.virtual_memory", return_value=SimpleNamespace(available=900)), \
                patch("luban.context.psutil.swap_memory", return_value=SimpleNamespace(free=100)), \
                patch("luban.context._mach_memory_api", side_effect=AssertionError("Mac only")):
            self.assertEqual(context.available_memory_bytes(), 1000)

    def test_free_swap_is_available_to_context_admission(self):
        with patch("luban.context.sys.platform", "linux"), \
                patch("luban.context.psutil.virtual_memory", return_value=SimpleNamespace(available=900)), \
                patch("luban.context.psutil.swap_memory", return_value=SimpleNamespace(free=500)):
            self.assertEqual(context.available_memory_bytes(), 1400)

    def test_context_keeps_os_available_estimate_when_mach_reading_is_lower(self):
        gib = context.GIB
        settings = {"context_mode": "auto", "context_tokens": 262144}
        model = SimpleNamespace(max_position_embeddings=262144, hidden_size=2048,
                                num_attention_heads=8, num_key_value_heads=2, head_dim=256,
                                layer_types=["full_attention"] * 6)
        # Both snapshots have the same nominal psutil available memory. A lower
        # Mach reading must not discard the OS estimate; no large allocation occurs.
        with patch("luban.context.sys.platform", "darwin"), \
                patch("luban.context.psutil.virtual_memory", return_value=SimpleNamespace(available=8*gib, free=gib)), \
                patch("luban.context.psutil.swap_memory", return_value=SimpleNamespace(free=0)), \
                patch("luban.context._darwin_memory_bytes", side_effect=[8*gib, 2*gib]):
            first = context.window(settings, model, context.available_memory_bytes(), device="mps", weight_bytes=3*gib)
            later = context.window(settings, model, context.available_memory_bytes(), device="mps", weight_bytes=3*gib)
        context.check_input(100, 128, first)
        context.check_input(100, 128, later)
        self.assertEqual(first["memory_estimate_limit"], later["memory_estimate_limit"])


if __name__ == "__main__":
    unittest.main()
