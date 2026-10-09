import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from primate_ai import session_templates as st

class TemplateTests(unittest.TestCase):
    def test_local_template_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(st, 'TEMPLATE_DIR', Path(d)):
                name='猴行为_40Hz'
                path=st.save_template(name, {'protocol_code':'EMF40', 'fps':1, 'camera_id':99})
                self.assertTrue(path.exists())
                self.assertEqual(st.list_templates(), [name])
                self.assertEqual(st.load_template(name), {'protocol_code':'EMF40','fps':1})
                self.assertNotIn('camera_id', path.read_text(encoding='utf-8'))
    def test_name_cannot_escape_template_folder(self):
        self.assertEqual(st.safe_name('../outside'), 'outside')
        with self.assertRaises(ValueError):
            st.safe_name('../')

if __name__=='__main__': unittest.main()
