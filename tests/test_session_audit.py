# coding: utf-8
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from primate_ai.session_audit import assess_session


class SessionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / '_metadata').mkdir()
        self.video = self.root / 'cam01_front.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (64, 48))
        assert writer.isOpened(), 'Test MJPEG writer unavailable'
        for i in range(16):
            writer.write(np.full((48, 64, 3), 30 + i * 5, dtype=np.uint8))
        writer.release()
        self.ts = self.root / '_metadata' / 'timestamps_cam01_front.csv'
        self._timestamps()
        self._metadata()

    def tearDown(self):
        self.temp.cleanup()

    def _timestamps(self, nonmonotonic=False, count=16):
        with self.ts.open('w', encoding='utf-8-sig', newline='') as f:
            out = csv.writer(f)
            out.writerow(['frame_index', 'session_time_s'])
            for i in range(count):
                out.writerow([i, '%.6f' % ((i - 2 if nonmonotonic and i == 8 else i) / 30.)])

    def _metadata(self, state='finished'):
        (self.root / 'session.json').write_text(json.dumps({
            'session_state': state, 'camera_specs': [
                {'slot_id': 0, 'view_key': 'front', 'width': 64, 'height': 48, 'fps': 30}
            ], 'camera_results': {'0': {'video': str(self.video), 'timestamps': str(self.ts)}}
        }), encoding='utf-8')

    def test_valid_capture(self):
        result = assess_session(self.root)
        self.assertEqual(result['overall_status'], '通过')
        self.assertEqual(result['cameras'][0]['timestamps']['rows'], 16)
        self.assertTrue((self.root / '采集验收报告.html').exists())
        self.assertTrue((self.root / '_metadata' / 'capture_acceptance.json').exists())

    def test_time_log_corruption_is_not_silent(self):
        self._timestamps(nonmonotonic=True)
        result = assess_session(self.root, save=False)
        self.assertEqual(result['overall_status'], '未通过')
        self.assertGreater(result['cameras'][0]['timestamps']['nonmonotonic'], 0)

    def test_incomplete_session_requires_review(self):
        self._metadata(state='recording_or_initializing')
        result = assess_session(self.root, save=False)
        self.assertEqual(result['overall_status'], '需复核')

    def test_video_missing_fails(self):
        self.video.unlink()
        result = assess_session(self.root, save=False)
        self.assertEqual(result['overall_status'], '未通过')

    def test_multi_camera_large_sync_spread_requests_review(self):
        second = self.root / 'cam02_left.avi'
        second.write_bytes(self.video.read_bytes())
        second_t = self.root / '_metadata' / 'timestamps_cam02_left.csv'
        second_t.write_bytes(self.ts.read_bytes())
        manifest = json.loads((self.root / 'session.json').read_text(encoding='utf-8'))
        manifest['camera_specs'].append({'slot_id': 1, 'view_key': 'left', 'width': 64, 'height': 48, 'fps': 30})
        manifest['camera_results']['1'] = {'video': str(second), 'timestamps': str(second_t)}
        manifest['software_sync_quality'] = {'p95_spread_ms': 100.0}
        manifest['requested_fps'] = 30.0
        (self.root / 'session.json').write_text(json.dumps(manifest), encoding='utf-8')
        result = assess_session(self.root, save=False)
        self.assertEqual(result['overall_status'], '需复核')
        self.assertTrue(any('P95' in x for x in result['warnings']))

    def test_bad_video_and_html_escape(self):
        (self.root / 'session.json').write_text(json.dumps({
            'session_state': 'finished', 'camera_specs': [
                {'slot_id': 0, 'view_key': '<script>alert(1)</script>', 'width': 64, 'height': 48}
            ]
        }), encoding='utf-8')
        result = assess_session(self.root)
        self.assertEqual(result['overall_status'], '未通过')
        markup = (self.root / '采集验收报告.html').read_text(encoding='utf-8')
        self.assertNotIn('<script>', markup)


if __name__ == '__main__':
    unittest.main()
