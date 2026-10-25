"""Controlled failure-injection checks; no real assets altered or accuracy claims."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import backend
from app import analyze_request,build_demo

ATTRIBUTES = __import__('json').loads((Path(__file__).parent/'attribute_names.json').read_text())
POINTS = [[70,80],[150,80],[110,120],[80,160],[140,160]]

class FakeModel:
    failures = set()
    def __init__(self,name,archive):
        if name in self.failures:
            raise ValueError('checkpoint hash mismatch')
        self.name=name;self.attributes=ATTRIBUTES;self.thresholds=np.full(24,.5)
        self.raise_infer=False;self.nonfinite=False
    def infer(self,tensor,points=None):
        if self.raise_infer:
            raise RuntimeError('injected inference failure')
        scores=np.full(24,np.nan if self.nonfinite else .75)
        return dict(scores=scores,decisions=scores>=self.thresholds,logits=np.ones(24))

class FakeDetector:
    def __init__(self):
        self.result=dict(status='ok',face_count=1,landmarks=POINTS)
        self.fail=False
    def detect(self,image):
        if self.fail:
            raise RuntimeError('injected detector failure')
        return self.result

class Errors(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.root.joinpath('r18_seed1_hci_assets.zip').touch()
        self.root.joinpath('r50_m3_seed1_hci_assets.zip').touch()
        FakeModel.failures=set()
        self.detector=FakeDetector()
        self.model_patch=patch.object(backend,'AttributeModel',FakeModel);self.model_patch.start()
        self.detector_patch=patch.object(backend.Comparison,'_load_detector',return_value=self.detector);self.detector_patch.start()
        self.engine=backend.Comparison(self.root)
        self.image=Image.new('RGB',(178,218),(100,120,140))
    def tearDown(self):
        self.detector_patch.stop();self.model_patch.stop();self.temp.cleanup()
    def assert_m3_off(self,result):
        self.assertNotIn('m3',result['results'])
        self.assertEqual(len(result['rows']),24)
        self.assertTrue(all(r[-2] is None and r[-1]=='Unavailable' for r in result['rows']))
        self.assertEqual(result['canvas'].tobytes(),result['overlay'].tobytes())
    def test_no_face_multiple_and_bad_points(self):
        cases=[dict(status='no_face',face_count=0),dict(status='ok',face_count=0),
               dict(status='ok',face_count=2,landmarks=POINTS),
               dict(status='ok',face_count=1,landmarks=None),
               dict(status='ok',face_count=1,landmarks=[['bad',0]]*5),
               dict(status='ok',face_count=1,landmarks=[[float('nan'),0]]*5),
               dict(status='ok',face_count=1,landmarks=[[224,0]]*5),None]
        for case in cases:
            with self.subTest(case=case):
                self.detector.result=case
                result=self.engine.infer(self.image)
                self.assert_m3_off(result)
                self.assertEqual(set(result['results']),{'r18','r50'})
    def test_detector_exception_then_recovery(self):
        self.assertIn('m3',self.engine.infer(self.image)['results'])
        self.detector.fail=True;self.assert_m3_off(self.engine.infer(self.image))
        self.detector.fail=False;self.assertIn('m3',self.engine.infer(self.image)['results'])
    def test_detector_load_failure(self):
        with patch.object(backend.Comparison,'_load_detector',side_effect=RuntimeError('missing dependency')):
            self.engine.reload()
        self.assert_m3_off(self.engine.infer(self.image))
        self.assertIn('unavailable',self.engine.summary())
    def test_one_model_inference_failure_isolated(self):
        for name in backend.MODELS:
            self.engine.models[name].raise_infer=True
            result=self.engine.infer(self.image)
            self.assertEqual(set(result['results']),set(backend.MODELS)-{name})
            i=1+2*backend.MODELS.index(name)
            self.assertTrue(all(r[i] is None and r[i+1]=='Unavailable' for r in result['rows']))
            self.engine.models[name].raise_infer=False
            self.assertEqual(len(self.engine.infer(self.image)['results']),3)
    def test_nonfinite_output_is_rejected(self):
        self.engine.models['r50'].nonfinite=True
        self.assertNotIn('r50',self.engine.infer(self.image)['results'])
    def test_model_load_failure_and_reload(self):
        FakeModel.failures={'m3'};self.engine.reload()
        self.assertNotIn('m3',self.engine.models)
        result=self.engine.infer(self.image);self.assertNotIn('m3',result['results'])
        self.assertTrue(all(r[-1] is None for r in result['thresholds']))
        FakeModel.failures=set();self.engine.reload()
        self.assertEqual(len(self.engine.models),3)
    def test_missing_archives_and_recovery(self):
        for path in self.root.glob('*.zip'):path.unlink()
        self.engine.reload();self.assertEqual(self.engine.models,{})
        self.assertTrue(all('missing' in e for e in self.engine.load_errors.values()))
        self.assertEqual(self.engine.infer(self.image)['results'],{})
        self.root.joinpath('r18_seed1_hci_assets.zip').touch()
        self.root.joinpath('r50_m3_seed1_hci_assets.zip').touch()
        self.engine.reload();self.assertEqual(len(self.engine.models),3)
    def test_actual_bad_zip_classification(self):
        self.model_patch.stop()
        try:
            self.engine.reload()
            self.assertEqual(self.engine.models,{})
            self.assertTrue(all('unreadable' in e for e in self.engine.load_errors.values()))
        finally:
            self.model_patch.start()
    def test_image_errors_clear_all_old_results(self):
        path=self.root/'portrait.png';self.image.save(path)
        success=analyze_request(self.engine,str(path),'blur',0)
        self.assertEqual(len(success[3]),24)
        bad=self.root/'bad.jpg';bad.write_text('not an image')
        for candidate in (None,str(bad),str(self.root/'missing.png')):
            result=analyze_request(self.engine,candidate,'blur',0)
            self.assertEqual(result[:5],(None,None,None,[],[]))
            self.assertEqual(len(result),12)
            self.assertEqual(result[7:],('','','',[],None))
        recovered=analyze_request(self.engine,str(path),'blur',0)
        self.assertEqual(len(recovered[3]),24)
    def test_language_failure_preserves_scores(self):
        path=self.root/'portrait.png';self.image.save(path)
        with patch('app.language_outputs',side_effect=ValueError('injected evidence error')):
            result=analyze_request(self.engine,str(path),'blur',0)
        self.assertEqual(len(result[3]),24)
        self.assertTrue(all('validation failed' in x for x in result[7:10]))
        self.assertEqual(result[10:],([],None))
    def test_reload_callback_clears_results_and_unlocks(self):
        demo=build_demo(self.engine)
        functions={f.fn.__name__:f.fn for f in demo.fns.values() if f.fn is not None}
        result=functions['reload_models']()
        self.assertEqual(result[1:5],(None,None,[],[]))
        self.assertEqual(len(result),14)
        self.assertIsNone(result[11])
        self.assertTrue(all(x.get('interactive') is True for x in functions['unlock_controls']()))

    def test_ui_snapshot_filters_and_upload_preservation(self):
        demo=build_demo(self.engine)
        functions={f.fn.__name__:f.fn for f in demo.fns.values() if f.fn is not None}
        path=self.root/'original_name.jpg';self.image.save(path)
        before=path.read_bytes()
        preview=functions['upload_changed'](str(path))
        self.assertEqual(len(preview),16)
        self.assertTrue(preview[0]['visible'])
        self.assertEqual(preview[0]['value'].size,self.image.size)
        self.assertFalse(preview[14]['visible'])
        self.assertTrue(preview[15]['visible'])
        self.assertEqual(path.read_bytes(),before)
        result=functions['analyze'](str(path),'blur',0,'All','Predicted','Smiling')
        self.assertEqual(len(result),14)
        self.assertIsNotNone(result[11])
        self.assertEqual(result[4][0][0],'Smiling')
        self.assertEqual(result[4][0][1],'0.500')
        self.assertEqual(result[11]['models']['r18']['attributes'][0]['score'],.75)
        self.assertIn('The model predicts',result[7])
        self.assertIn('attribute-grid',result[12])
        cleared=functions['clear_results']()
        self.assertIsNone(cleared[11])
        self.assertNotIn('attribute-grid',cleared[12])
        reset=functions['upload_changed'](None)
        self.assertFalse(reset[0]['visible'])
        self.assertIsNone(reset[11])
        reset=functions['reset_image']()
        self.assertIsNone(reset[14]['value'])
        self.assertTrue(reset[14]['visible'])
        self.assertFalse(reset[15]['visible'])
        self.assertIsNone(reset[11])
        self.assertEqual(reset[3:5],([],[]))
        self.assertNotIn('attribute-grid',reset[12])
        config=demo.get_config_file()
        labels=[c['props'].get('label','') for c in config['components']]
        self.assertNotIn('Status',labels)
        self.assertFalse(any('Local wording' in x for x in labels))
        self.assertEqual(next(c for c in config['components'] if c['type']=='file')['props']['type'],'filepath')
        accordions={c['props'].get('label'):c['props'] for c in config['components'] if c['type']=='accordion'}
        self.assertFalse(accordions['Advanced / Troubleshooting']['open'])

if __name__=='__main__':
    unittest.main(verbosity=2)
