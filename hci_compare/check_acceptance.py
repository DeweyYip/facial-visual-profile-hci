"""Controlled checks for the acceptance harness; no Qwen inference."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import run_acceptance as acceptance


class ControlledService:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def generate(self, template):
        self.calls += 1
        if self.fail:
            raise TimeoutError('controlled timeout')
        return json.dumps({'sentences':[dict(text='The model predicts '+c['phrase']+'.',
                                              evidence_id=c['evidence_id']) for c in template['claims']]}), dict(startup_s=0,generation_s=.01)


class Checks(unittest.TestCase):
    def test_frozen_suite_and_expected_templates(self):
        freeze,suite=acceptance.load_frozen()
        rows=[acceptance.run_case(case) for case in suite['cases']]
        summary=acceptance.summarize(rows,False)
        self.assertTrue(summary['system_contract_pass'])
        self.assertEqual(summary['template_contract_pass'],20)
        self.assertIsNone(summary['local_contract_pass'])
        self.assertFalse(summary['preexisting_held_out_evaluated'])

    def test_generation_and_gate_counts(self):
        _,suite=acceptance.load_frozen()
        service=ControlledService()
        rows=[acceptance.run_case(case,service) for case in suite['cases']]
        summary=acceptance.summarize(rows,True)
        self.assertEqual(service.calls,15)
        self.assertEqual(summary['accepted_local_generations'],15)
        self.assertEqual(summary['gated_without_generation'],5)
        self.assertEqual(summary['template_fallbacks'],0)
        self.assertTrue(summary['system_contract_pass'])

    def test_fallback_does_not_count_as_generation_success(self):
        _,suite=acceptance.load_frozen()
        service=ControlledService(fail=True)
        rows=[acceptance.run_case(case,service) for case in suite['cases']]
        summary=acceptance.summarize(rows,True)
        self.assertEqual(summary['template_fallbacks'],15)
        self.assertEqual(summary['accepted_local_generations'],0)
        self.assertTrue(summary['system_contract_pass'])
        self.assertFalse(summary['local_generation_without_fallback'])

    def test_incomplete_or_wrong_expectation_cannot_pass(self):
        _,suite=acceptance.load_frozen()
        case=copy.deepcopy(suite['cases'][0])
        case['expected']['claim_attributes']=['Eyeglasses']
        row=acceptance.run_case(case)
        self.assertFalse(row['template_contract_pass'])
        self.assertFalse(acceptance.summarize([row],False)['system_contract_pass'])

    def test_modified_policy_or_suite_rejected(self):
        freeze,_=acceptance.load_frozen()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in [*freeze['policy_files'],'development_cases.json']:
                shutil.copyfile(acceptance.ROOT/name,root/name)
            shutil.copytree(acceptance.SUITE,root/'acceptance_nov02',ignore=shutil.ignore_patterns('results'))
            with patch.object(acceptance,'ROOT',root),patch.object(acceptance,'SUITE',root/'acceptance_nov02'):
                acceptance.load_frozen()
                policy=root/'local_wording.py';original=policy.read_bytes()
                policy.write_bytes(original+b'\n# changed policy\n')
                with self.assertRaisesRegex(ValueError,'Frozen policy file mismatch'):
                    acceptance.load_frozen()
                policy.write_bytes(original)
                cases=root/'acceptance_nov02'/'cases.json';cases.write_text(cases.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'Acceptance input file changed'):
                    acceptance.load_frozen()


if __name__=='__main__':
    unittest.main(verbosity=2)
