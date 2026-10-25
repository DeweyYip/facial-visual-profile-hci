"""Validate 60 synthetic development fixtures and fail-closed language contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from language import (ATTRIBUTES,MODELS,validate_payload,describe,audit_description,
                      agreement,build_payload,language_outputs)

ROOT=Path(__file__).resolve().parent

def rejects(function):
    try: function()
    except (ValueError,KeyError,TypeError): return
    raise AssertionError('Invalid language data was accepted')

def main():
    suite=json.loads((ROOT/'development_cases.json').read_text())
    cases=suite['cases']
    assert len(cases)==60 and len({c['case_id'] for c in cases})==60
    assert suite['not_real_images'] and suite['not_accuracy_evaluation']
    for case in cases:
        assert case['synthetic'] is True
        payload=case['input'];validate_payload(payload)
        description=describe(payload,case['source_model'])
        expected=case['expected']
        actual=[c['attribute'] for c in description['claims']]
        assert actual==expected['claim_attributes'],(case['case_id'],actual,expected)
        assert description['status']==expected['status'],case['case_id']
        if expected['mixed_count'] is not None:
            assert sum(a['state']=='mixed' for a in agreement(payload))==expected['mixed_count'],case['case_id']
        for model in MODELS:
            audit_description(payload,describe(payload,model))
    print('PASS: 60/60 synthetic structured development cases; exact expected claims/status and evidence audits.')
    base=cases[0]['input']
    output=describe(base,'r18')
    changed=copy.deepcopy(output);changed['text']+=' The person is happy.'
    rejects(lambda:audit_description(base,changed))
    changed=copy.deepcopy(output);changed['claims'][0]['attribute']='Eyeglasses'
    rejects(lambda:audit_description(base,changed))
    for mutation in ('nan','wrong_decision','attribute_order','policy','face_gate'):
        payload=copy.deepcopy(base)
        if mutation=='nan':payload['models']['r18']['attributes'][0]['score']=float('nan')
        if mutation=='wrong_decision':payload['models']['r18']['attributes'][0]['decision']=False
        if mutation=='attribute_order':payload['models']['r18']['attributes'].reverse()
        if mutation=='policy':payload['wording_policy']['positive_margin']=0.001
        if mutation=='face_gate':payload['detection']['face_count']=2
        rejects(lambda:validate_payload(payload))
    # Exercise the actual comparison -> language input adapter with structured fixtures.
    models={m:SimpleNamespace(thresholds=[r['threshold'] for r in base['models'][m]['attributes']]) for m in MODELS}
    result=dict(results={m:dict(scores=[r['score'] for r in base['models'][m]['attributes']],
                               decisions=[r['decision'] for r in base['models'][m]['attributes']]) for m in MODELS},
        detection=dict(status='ok',face_count=1),geometry_valid=True,geometry_reason='ok',system_rows=[],
        corruption=dict(output_rgb_sha256='0'*64,kind='blur',severity=0))
    payload=build_payload(result,models)
    assert payload['models']==base['models']
    outputs=language_outputs(result,models)
    assert len(outputs)==5 and len(outputs[3])==3
    assert all(row[1]=='Smiling' for row in outputs[3])
    result.update(geometry_valid=False,geometry_reason='no_face',detection=dict(status='no_face',face_count=0))
    outputs=language_outputs(result,models)
    assert all('No facial description generated' in text for text in outputs[:3]) and outputs[3]==[]
    print('PASS: unsupported wording/evidence and malformed inputs rejected; live adapter and no-face abstention passed.')
    print('LANGUAGE CHECK PASSED. Development fixtures only; no real-image or held-out evaluation claim.')

if __name__=='__main__':
    main()
