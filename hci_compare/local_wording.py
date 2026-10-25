"""Optional local Qwen wording with a finite phrase policy and template fallback."""
import atexit
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request
from language import describe, validate_payload

MODEL_FILE = 'Qwen3-1.7B-Q8_0.gguf'
MODEL_SHA256 = '061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a'
MODEL_SIZE = 1834426016
POLICY = 'finite_phrases_v1'
SYSTEM = ('Rewrite allowed model predictions in English. Output only a JSON object with one key '
          '"sentences", a list of objects with exactly "text" and "evidence_id". '
          'Use every supplied evidence_id once. For each text use exactly "The model predicts " '
          'followed by its supplied phrase; an optional final period is allowed. '
          'Do not add any other information or explanations. /no_think')


def output_schema(claims):
    """Constrain generation itself; post-generation evidence/phrase checks still apply."""
    return dict(type='object', properties=dict(sentences=dict(type='array',
        minItems=len(claims), maxItems=len(claims), items=dict(type='object',
            properties=dict(text=dict(type='string', enum=[
                'The model predicts ' + c['phrase'] + '.' for c in claims]),
                evidence_id=dict(type='string', enum=[c['evidence_id'] for c in claims])),
            required=['text', 'evidence_id'], additionalProperties=False))),
        required=['sentences'], additionalProperties=False)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_wording(raw, claims):
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {'sentences'} or not isinstance(obj['sentences'], list):
        raise ValueError('invalid_json_structure')
    allowed = {c['evidence_id']: c for c in claims}
    seen = set()
    accepted = {}
    for sentence in obj['sentences']:
        if not isinstance(sentence, dict) or set(sentence) != {'text', 'evidence_id'}:
            raise ValueError('invalid_sentence_fields')
        eid = sentence['evidence_id']
        if not isinstance(eid, str) or eid not in allowed or eid in seen:
            raise ValueError('invalid_evidence_reference')
        text = sentence['text']
        if not isinstance(text, str):
            raise ValueError('invalid_text')
        canonical = 'The model predicts ' + allowed[eid]['phrase']
        variants = {canonical, canonical + '.'}
        # Reviewed equivalent form observed in the Oct31 development trial.
        if allowed[eid]['attribute'] == 'Eyeglasses':
            variants.update({'The model predicts eyeglasses are present in the image',
                             'The model predicts eyeglasses are present in the image.'})
        if text.strip() not in variants:
            raise ValueError('wording_outside_allowed_phrases')
        seen.add(eid)
        accepted[eid] = text.strip().rstrip('.') + '.'
    if seen != set(allowed):
        raise ValueError('missing_evidence_reference')
    return [accepted[c['evidence_id']] for c in claims]


class LocalWording:
    def __init__(self, project_root):
        self.root = Path(project_root) / 'llm_trial' / 'local'
        self.process = None
        self.log = None
        self.base = None
        self.lock = threading.RLock()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
            self.base = None
            if self.log is not None:
                self.log.close()
                self.log = None

    def request(self, route, body=None, timeout=2):
        req = urllib.request.Request(self.base + route,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=timeout) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError('response_too_large')
        return json.loads(data)

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return 0.0
        self.close()
        started = time.perf_counter()
        model = self.root / 'models' / MODEL_FILE
        if not model.is_file() or model.stat().st_size != MODEL_SIZE or digest(model) != MODEL_SHA256:
            raise RuntimeError('local_model_missing_or_invalid')
        runtime = self.root / 'runtime'
        inventory_path = self.root / 'runtime_inventory.json'
        if not inventory_path.is_file():
            raise RuntimeError('runtime_inventory_missing')
        inventory = json.loads(inventory_path.read_text())
        servers = list(runtime.rglob('llama-server'))
        if len(servers) != 1 or not inventory or str(servers[0].relative_to(runtime)) not in inventory:
            raise RuntimeError('local_runtime_missing_or_invalid')
        for relative, expected in inventory.items():
            path = (runtime / relative).resolve()
            if not path.is_relative_to(runtime.resolve()) or not path.is_file() or digest(path) != expected:
                raise RuntimeError('local_runtime_verification_failed')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.base = 'http://127.0.0.1:' + str(port)
        self.log = (self.root / 'hci_wording_server.log').open('a')
        self.process = subprocess.Popen([str(servers[0]), '-m', str(model), '--host', '127.0.0.1',
            '--port', str(port), '-c', '2048', '-t', '4', '-ngl', '0', '--parallel', '1',
            '--jinja', '--chat-template-kwargs', '{"enable_thinking":false}', '--perf'],
            stdout=self.log, stderr=self.log)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('local_server_exited')
            try:
                if self.request('/health').get('status') == 'ok':
                    return time.perf_counter() - started
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        raise TimeoutError('local_server_startup_timeout')

    def generate(self, template):
        with self.lock:
            try:
                startup = self.start()
                facts = [dict(evidence_id=c['evidence_id'], phrase=c['phrase']) for c in template['claims']]
                body = dict(messages=[dict(role='system', content=SYSTEM),
                                      dict(role='user', content=json.dumps({'allowed_facts': facts}))],
                            temperature=0, seed=31, max_tokens=384, stream=False, cache_prompt=False,
                            chat_template_kwargs={'enable_thinking': False},
                            response_format={'type': 'json_object', 'schema': output_schema(template['claims'])})
                started = time.perf_counter()
                response = self.request('/v1/chat/completions', body, timeout=60)
                choice = response['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('incomplete_generation')
                return choice['message']['content'], dict(startup_s=round(startup, 3),
                    generation_s=round(time.perf_counter() - started, 3))
            except Exception:
                self.close()
                raise


def optional_description(service, payload, source):
    if payload is None:
        return '', 'Compare models first to create current evidence.', dict(status='no_current_evidence')
    try:
        validate_payload(payload)
        template = describe(payload, source)
    except Exception:
        return '', 'Description unavailable: evidence validation failed. Compare models again.', dict(status='invalid_evidence')
    metadata = dict(status='gate_not_called', source_model=source, policy=POLICY,
                    final_rgb_sha256=payload['input']['final_rgb_sha256'],
                    evidence_ids=[c['evidence_id'] for c in template['claims']])
    if not template['claims']:
        return template['text'], 'Template retained: no eligible local generation request.', metadata
    try:
        raw, timings = service.generate(template)
        sentences = audit_wording(raw, template['claims'])
        metadata.update(status='accepted_local_wording', **timings)
        text = source.upper() + ' predictions · local wording:\n' + '\n'.join(sentences)
        if template['notes']:
            text += '\n' + '\n'.join(template['notes'])
        return text, 'Local wording passed the allowed-phrase and evidence checks.', metadata
    except Exception as exc:
        print(f'Local wording fallback: {type(exc).__name__}: {exc}', flush=True)
        metadata.update(status='template_fallback', reason=str(exc)[:200])
        return template['text'], 'Local wording unavailable or rejected; showing the grounded template.', metadata
