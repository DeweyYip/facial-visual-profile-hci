"""Controlled integration checks, without downloading or loading Qwen."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
import shlex
from unittest.mock import patch
import local_wording
from language import describe, validate_payload
from local_wording import LocalWording, audit_wording, optional_description, output_schema

ROOT = Path(__file__).resolve().parent
CASES = json.loads((ROOT / 'development_cases.json').read_text())['cases']


def canonical(claims):
    return json.dumps({'sentences': [dict(text='The model predicts ' + c['phrase'],
                                         evidence_id=c['evidence_id']) for c in claims]})


class Service:
    def __init__(self, error=None, mutate=None):
        self.calls = 0
        self.error = error
        self.mutate = mutate

    def generate(self, template):
        self.calls += 1
        if self.error:
            raise self.error
        raw = canonical(template['claims'])
        if self.mutate:
            raw = self.mutate(raw)
        return raw, dict(startup_s=0, generation_s=0.01)


class Checks(unittest.TestCase):
    def test_all_development_gates_and_evidence(self):
        for case in CASES:
            service = Service()
            payload = case['input']; source = case['source_model']
            template = describe(payload, source)
            text, status, audit = optional_description(service, payload, source)
            self.assertEqual(service.calls, int(bool(template['claims'])))
            self.assertEqual(audit['evidence_ids'], [c['evidence_id'] for c in template['claims']])
            self.assertEqual(audit['status'], 'accepted_local_wording' if template['claims'] else 'gate_not_called')
            if not template['claims']:
                self.assertEqual(text, template['text'])
            else:
                self.assertTrue(all(n in text for n in template['notes']))

    def test_all_24_phrases_and_sources(self):
        for case in CASES[:24]:
            for source in ('r18','r50','m3'):
                payload = copy.deepcopy(case['input'])
                validate_payload(payload)
                template = describe(payload, source)
                service = Service()
                text, status, audit = optional_description(service, payload, source)
                self.assertEqual(audit['status'], 'accepted_local_wording')
                self.assertTrue(all(e.startswith(source + ':') for e in audit['evidence_ids']))

    def test_correct_id_with_wrong_semantics_rejected(self):
        payload = CASES[0]['input']
        service = Service(mutate=lambda raw: raw.replace('The model predicts a smile', 'The person is happy'))
        text, status, audit = optional_description(service, payload, 'r18')
        self.assertEqual(audit['status'], 'template_fallback')
        self.assertEqual(text, describe(payload, 'r18')['text'])
        self.assertNotIn('happy', text)

    def test_malformed_missing_cross_model_duplicate_and_extra(self):
        claims = describe(CASES[0]['input'], 'r18')['claims']
        for raw in ('{', '{"sentences":[]}',
                    canonical(claims).replace('r18:', 'r50:'),
                    json.dumps({'sentences': json.loads(canonical(claims))['sentences'] * 2}),
                    canonical(claims).replace('a smile', 'a smile and blue eyes')):
            with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError)):
                audit_wording(raw, claims)

    def test_network_load_timeout_and_contract_failures_fall_back(self):
        payload = CASES[0]['input']
        for error in (TimeoutError('request timeout'), RuntimeError('missing model'), OSError('server failed')):
            text, _, audit = optional_description(Service(error=error), payload, 'r18')
            self.assertEqual(audit['status'], 'template_fallback')
            self.assertEqual(text, describe(payload, 'r18')['text'])
        bad = copy.deepcopy(payload); bad['models']['r18']['attributes'][0]['score'] = float('nan')
        service = Service()
        self.assertEqual(optional_description(service, bad, 'r18')[2]['status'], 'invalid_evidence')
        self.assertEqual(optional_description(service, None, 'r18')[2]['status'], 'no_current_evidence')
        self.assertEqual(service.calls, 0)

    def test_service_cleanup_on_timeout_and_truncation(self):
        template = describe(CASES[0]['input'], 'r18')
        with tempfile.TemporaryDirectory() as folder:
            service = LocalWording(Path(folder))
            for response in (TimeoutError('timeout'), {'choices':[{'finish_reason':'length'}]}):
                with patch.object(service, 'start', return_value=0), \
                     patch.object(service, 'request', side_effect=response if isinstance(response,Exception) else None,
                                  return_value=response), patch.object(service, 'close') as close:
                    with self.assertRaises((TimeoutError, ValueError)):
                        service.generate(template)
                    close.assert_called_once()

    def test_recorded_mac_outputs_against_stricter_policy(self):
        # Expected stricter acceptance of the reviewed equivalent eyeglasses form.
        claims = describe(CASES[2]['input'], 'r18')['claims']
        raw = canonical(claims).replace('The model predicts eyeglasses',
                                       'The model predicts eyeglasses are present in the image.')
        self.assertEqual(len(audit_wording(raw, claims)), 1)

    def test_owned_http_service_start_generation_reuse_and_release(self):
        program = '''import argparse,json
from http.server import BaseHTTPRequestHandler,HTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int);args,_=p.parse_known_args()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*unused):pass
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ok"}')
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  facts=json.loads(body['messages'][1]['content'])['allowed_facts']
  schema=body['response_format']['schema']
  assert schema['required']==['sentences']
  item=schema['properties']['sentences']['items']
  assert set(item['required'])=={'text','evidence_id'}
  assert item['additionalProperties'] is False
  raw=json.dumps({'sentences':[{'text':'The model predicts '+f['phrase'],'evidence_id':f['evidence_id']} for f in facts]})
  response={'choices':[{'finish_reason':'stop','message':{'content':raw}}]}
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(response).encode())
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
'''
        with tempfile.TemporaryDirectory() as folder:
            service = LocalWording(Path(folder))
            runtime = service.root / 'runtime';runtime.mkdir(parents=True)
            server = runtime / 'llama-server'
            server.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(runtime/'fake.py'))+' "$@"\n')
            server.chmod(0o755)
            (runtime/'fake.py').write_text(program)
            models=service.root/'models';models.mkdir()
            model=models/local_wording.MODEL_FILE;model.write_bytes(b'controlled_only')
            inventory={p.name:local_wording.digest(p) for p in runtime.iterdir()}
            (service.root/'runtime_inventory.json').write_text(json.dumps(inventory))
            try:
                with patch.object(local_wording,'MODEL_SIZE',model.stat().st_size), \
                     patch.object(local_wording,'MODEL_SHA256',local_wording.digest(model)):
                    first=optional_description(service,CASES[0]['input'],'r18')
                    self.assertEqual(first[2]['status'],'accepted_local_wording')
                    process=service.process
                    second=optional_description(service,CASES[2]['input'],'r18')
                    self.assertIs(service.process,process)
                    self.assertEqual(second[2]['startup_s'],0)
            finally:
                service.close()
            self.assertIsNotNone(process.poll())
            self.assertIsNone(service.process)

    def test_generation_schema_requires_fields_and_limits_claim_count(self):
        claims=describe(CASES[45]['input'],'r18')['claims']
        schema=output_schema(claims)
        array=schema['properties']['sentences']
        self.assertEqual(array['minItems'],len(claims))
        self.assertEqual(array['maxItems'],len(claims))
        item=array['items']
        self.assertEqual(set(item['required']),{'text','evidence_id'})
        self.assertFalse(item['additionalProperties'])
        self.assertEqual(item['properties']['evidence_id']['enum'],[c['evidence_id'] for c in claims])
        self.assertEqual(item['properties']['text']['enum'],['The model predicts '+c['phrase']+'.' for c in claims])


if __name__ == '__main__':
    unittest.main(verbosity=2)
