"""Controlled runtime checks. No Qwen weights or real model inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from unittest.mock import patch
import run_trial as trial
import setup

FAKE_SERVER = r'''import argparse, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--port',type=int); args,_=p.parse_known_args()
Path(__file__).with_name('server.pid').write_text(str(os.getpid()))
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ok"}')
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  facts=json.loads(body['messages'][1]['content'])['allowed_facts']
  sentences=[dict(text='The model predicts '+f['phrase']+'.',evidence_id=f['evidence_id']) for f in facts]
  if facts[0]['phrase']=='eyeglasses':sentences[0]['text']='The person is happy.'
  if facts[0]['phrase']=='no visible beard':sentences[0]['evidence_id']='r50:No_Beard'
  response=dict(choices=[dict(message=dict(content=json.dumps(dict(sentences=sentences))),finish_reason='stop')],usage=dict(completion_tokens=12))
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(response).encode())
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
'''


def rejects(fn):
    try:
        fn()
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return
    raise AssertionError('Invalid data accepted')


def main():
    suite = json.loads((trial.APP / 'development_cases.json').read_text())['cases']
    claims = trial.describe(suite[0]['input'], 'r18')['claims']
    good = {'sentences': [{'text': 'The model predicts a smile.', 'evidence_id': claims[0]['evidence_id']}]}
    trial.audit(json.dumps(good), claims)
    rejects(lambda: trial.audit('{', claims))
    rejects(lambda: trial.audit('{"sentences":[]}', claims))
    rejects(lambda: trial.audit(json.dumps({'sentences': good['sentences'] * 2}), claims))
    rejects(lambda: trial.audit('{"sentences":[{"text":"x","evidence_id":"m3:Smiling"}]}', claims))
    # Explicitly document the structural check's semantic limitation.
    wrong_semantics = {'sentences': [{'text': 'The person is happy.', 'evidence_id': claims[0]['evidence_id']}]}
    trial.audit(json.dumps(wrong_semantics), claims)
    with tempfile.TemporaryDirectory(prefix='fvp_trial_check_') as tmp:
        root = Path(tmp)
        data = b'controlled test download'
        original = root / 'original.bin'; original.write_bytes(data)
        target = root / 'copy.bin'
        digest = hashlib.sha256(data).hexdigest()
        setup.download(original.as_uri(), target, len(data), digest)
        setup.download(original.as_uri(), target, len(data), digest)
        rejects(lambda: setup.download(original.as_uri(), root / 'bad.bin', len(data), '0' * 64))
        assert not (root / 'bad.bin').exists() and not (root / 'bad.bin.part').exists()
        archive = root / 'escape.tar'
        with tarfile.open(archive, 'w') as tar:
            item = tarfile.TarInfo('../escape'); item.size = 1
            tar.addfile(item, io.BytesIO(b'x'))
        rejects(lambda: setup.extract(archive, root / 'extract'))
        local = root / 'local'; runtime = local / 'runtime'; runtime.mkdir(parents=True)
        server = runtime / 'llama-server'
        # Shell quoting supports Python paths containing spaces on the user's Mac.
        import shlex
        server.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(runtime / 'fake_server.py')) + ' "$@"\n')
        server.chmod(0o755)
        (runtime / 'fake_server.py').write_text(FAKE_SERVER)
        models = local / 'models'; models.mkdir()
        model = models / 'fake.gguf'; model.write_bytes(b'test_only')
        spec = dict(file=model.name, size=model.stat().st_size, sha256=setup.sha256(model))
        (root / 'model_manifest.json').write_text(json.dumps({'0.6b': spec}))
        (root / 'runtime_manifest.json').write_text('{"controlled_test":true}')
        (local / 'runtime_asset.json').write_text('{"controlled_test":true}')
        (local / 'runtime_inventory.json').write_text(json.dumps({p.name:setup.sha256(p) for p in runtime.iterdir()}))
        inputs = [dict(suite[i], provenance='controlled_test_fixture') for i in (0, 2, 6, 24, 38, 41)]
        with patch.object(trial, 'ROOT', root), patch.object(trial, 'cases', return_value=inputs), \
             patch.object(sys, 'argv', ['run_trial.py', '--models', '0.6b', '--startup-timeout', '5', '--request-timeout', '5']):
            assert trial.main() == 0
        report_path = next((root / 'results').glob('*/0.6b.json'))
        report = json.loads(report_path.read_text())
        assert report['status'] == 'completed'
        assert [r['status'] for r in report['rows']] == [
            'structure_pass_manual_review_pending', 'structure_pass_manual_review_pending',
            'request_or_structure_failed', 'gate_not_called', 'gate_not_called', 'gate_not_called']
        assert report['rows'][1]['manual_semantic_review'] == 'pending'
        assert (report_path.parent / 'summary.csv').is_file() and (report_path.parent / 'review.csv').is_file()
        import os
        try:
            os.kill(int((runtime / 'server.pid').read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('Trial left a server process running')
        # Invalid model must fail before a server is spawned.
        directory = root / 'failed'; directory.mkdir()
        args = argparse.Namespace(threads=1, startup_timeout=2, request_timeout=2)
        with patch.object(trial, 'ROOT', root):
            bad = trial.run_model('0.6b', dict(spec, sha256='0'*64), args, inputs, directory, server)
        assert bad['status'] == 'failed' and bad['rows'] == []
        # An unresponsive startup must time out and terminate its process too.
        hanging = runtime / 'hanging_server.py'
        hanging.write_text('import os,time\nfrom pathlib import Path\nPath(__file__).with_suffix(".pid").write_text(str(os.getpid()))\ntime.sleep(30)\n')
        server.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(hanging)) + ' "$@"\n')
        args.startup_timeout = 0.4
        with patch.object(trial, 'ROOT', root):
            timed = trial.run_model('0.6b', spec, args, inputs, directory, server)
        assert timed['status'] == 'failed' and 'timed out' in timed['error']
        try:
            os.kill(int(hanging.with_suffix('.pid').read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('Timed-out server was left running')
    print('PASS: controlled HTTP/report/cleanup, gate skips, bad evidence rejection and download verification.')
    print('TRIAL TOOL CHECK PASSED. No real Qwen inference or Intel Mac performance measured.')


if __name__ == '__main__':
    main()
