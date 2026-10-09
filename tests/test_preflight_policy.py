import unittest
from primate_ai.preflight_policy import preflight_candidates

class CapturePolicyTests(unittest.TestCase):
    def test_automatic_fallback_deduplicates(self):
        self.assertEqual(preflight_candidates(1280,720,30),[(1280,720,30),(640,480,30)])
        self.assertEqual(preflight_candidates(1920,1080,30),[(1920,1080,30),(1280,720,30),(640,480,30)])
    def test_fixed_never_downgrades(self):
        self.assertEqual(preflight_candidates(1920,1080,60,automatic=False),[(1920,1080,60)])
    def test_low_res_saved_mode_does_not_try_high_mode_by_default(self):
        self.assertEqual(preflight_candidates(640,480,30),[(640,480,30),(1280,720,30)])

if __name__=='__main__':unittest.main()
