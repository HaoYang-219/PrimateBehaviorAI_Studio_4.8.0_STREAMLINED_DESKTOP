import unittest
import numpy as np
from primate_ai.roi_assist import image_reference_quality, suggest_reference_time

class ReferenceFrameTests(unittest.TestCase):
    def test_empty_image_is_rejected(self):
        self.assertEqual(image_reference_quality(None),-1.)

    def test_exposure_and_sharpness(self):
        blank=np.zeros((80,100,3),dtype=np.uint8)
        checker=((np.indices((80,100)).sum(axis=0)%2)*180+30).astype(np.uint8)
        clear=np.stack([checker]*3,axis=2)
        self.assertGreater(image_reference_quality(clear),image_reference_quality(blank))

    def test_frame_suggestion_prefers_quality(self):
        blank=np.zeros((80,100,3),dtype=np.uint8)
        checker=((np.indices((80,100)).sum(axis=0)%2)*180+30).astype(np.uint8)
        clear=np.stack([checker]*3,axis=2)
        selected=suggest_reference_time(lambda t:clear if t>=5 else blank,20)
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(selected,5.)

if __name__=='__main__': unittest.main()
