"""Display contract checks for the v6 interface; synthetic inputs, no real weights."""
import copy
import json
from pathlib import Path
import unittest
from ui_presenter import CATEGORIES, summaries, explorer, focus
from language import ATTRIBUTES, describe

CASES=json.loads((Path(__file__).parent/'development_cases.json').read_text())['cases']


class DisplayChecks(unittest.TestCase):
    def test_all_attributes_have_a_category(self):
        self.assertEqual(set().union(*(v for k,v in CATEGORIES.items() if k!='All')),set(ATTRIBUTES))

    def test_summaries_use_only_existing_claims_and_do_not_mutate_evidence(self):
        for case in CASES:
            payload=case['input'];before=copy.deepcopy(payload)
            views=summaries(payload)
            for i,name in enumerate(('r18','r50','m3')):
                description=describe(payload,name)
                for claim in description['claims']:
                    self.assertIn(claim['phrase'],views[i])
                if not description['claims']:
                    self.assertIn(description['text'].split('\n')[0],views[i])
            self.assertEqual(payload,before)

    def test_all_filters_and_score_views_preserve_source_data(self):
        for case in CASES:
            payload=case['input'];before=copy.deepcopy(payload)
            for category in CATEGORIES:
                for mode in ('Predicted','Differences','All attributes'):
                    self.assertIsInstance(explorer(payload,category,mode),str)
            for attribute in ATTRIBUTES:
                self.assertIn('score-detail',focus(payload,attribute))
            self.assertEqual(payload,before)

    def test_missing_evidence_has_no_old_predictions(self):
        self.assertNotIn('attribute-row',explorer(None))
        self.assertNotIn('score-fill',focus(None,'Smiling'))
        self.assertTrue(all('will appear here' in v for v in summaries(None)))

    def test_unavailable_source_is_not_reported_as_a_negative_prediction(self):
        case=next(c for c in CASES if not c['input']['models']['m3']['available'])
        html=explorer(case['input'],'All','All attributes')
        self.assertIn('Unavailable',html)
        self.assertIn('Unavailable',focus(case['input'],'Smiling'))


if __name__=='__main__':
    unittest.main(verbosity=2)
