"""Sequential local Qwen trial. Synthetic development inputs; no accuracy claim."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
APP = ROOT.parent / 'hci_compare'
sys.path.insert(0, str(APP))
from language import describe, validate_payload  # pure stdlib; no model/image import
from setup import sha256

SELECTED = (0, 1, 2, 6, 23, 24, 30, 38, 41, 45, 50, 59)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SYSTEM = ('You rewrite model-predicted facial attributes into short English sentences. '
          'These are uncalibrated classifier predictions, not verified facts. '
          'Use every allowed fact exactly once; add no other facts, emotions, identity, '
          'age, gender, ethnicity, attractiveness, confidence percentages, or explanations. '
          'Do not change a smile into a mood. Output JSON only, with exactly one key '
          '"sentences": a list of {"text": "The model predicts ...", '
          '"evidence_id": "the supplied id"}. Each sentence uses one allowed fact. /no_think')


def request(base, route, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=data, headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as response:
        return json.load(response)


def audit(raw, claims):
    """Structural/reference check ONLY. An id does not prove sentence semantics."""
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {'sentences'} or not isinstance(obj['sentences'], list):
        raise ValueError('Expected exactly sentences list')
    expected = {c['evidence_id'] for c in claims}
    ids = []
    for sentence in obj['sentences']:
        if not isinstance(sentence, dict) or set(sentence) != {'text', 'evidence_id'}:
            raise ValueError('Unexpected sentence fields')
        if not isinstance(sentence['text'], str) or not sentence['text'].strip() or len(sentence['text']) > 500:
            raise ValueError('Invalid sentence text')
        if not isinstance(sentence['evidence_id'], str) or sentence['evidence_id'] not in expected:
            raise ValueError('Unknown/cross-model evidence id')
        ids.append(sentence['evidence_id'])
    if set(ids) != expected or len(ids) != len(expected):
        raise ValueError('Missing or duplicate evidence ids')
    return obj


def cases(args):
    if args.payload:
        payload = json.loads(args.payload.read_text())
        validate_payload(payload)
        return [dict(case_id='exported_input', input=payload, source_model=args.source,
                     provenance='user_export_not_held_out_evaluation')]
    source = APP / 'development_cases.json'
    suite = json.loads(source.read_text())
    if not suite['not_real_images'] or not suite['not_accuracy_evaluation']:
        raise ValueError('Unexpected development suite provenance')
    selected = suite['cases'] if args.all_dev else [suite['cases'][i] for i in SELECTED]
    return [dict(c, provenance='synthetic_development_fixture') for c in selected]


class MemorySampler:
    """Observed RSS of only the server PID, every 250ms; not system or app peak."""
    def __init__(self, pid):
        self.pid = pid
        self.peak_kib = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                value = int(subprocess.check_output(
                    ['ps', '-o', 'rss=', '-p', str(self.pid)], stderr=subprocess.DEVNULL,
                    text=True, timeout=2).strip())
                self.peak_kib = max(value, self.peak_kib or 0)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=3)


def generation(base, template, timeout):
    facts = [dict(evidence_id=c['evidence_id'], phrase=c['phrase']) for c in template['claims']]
    body = dict(messages=[dict(role='system', content=SYSTEM),
                          dict(role='user', content=json.dumps({'allowed_facts': facts}))],
                temperature=0, seed=31, max_tokens=384, stream=False, cache_prompt=False,
                chat_template_kwargs={'enable_thinking': False},
                response_format={'type': 'json_object'})
    started = time.perf_counter()
    response = request(base, '/v1/chat/completions', body, timeout)
    elapsed = time.perf_counter() - started
    choice = response['choices'][0]
    return choice['message']['content'], elapsed, choice.get('finish_reason'), response.get('usage'), response.get('timings')


def run_model(name, spec, args, inputs, directory, server):
    model = ROOT / 'local' / 'models' / spec['file']
    report = dict(model=name, quantization='Q8_0', model_spec=spec, status='starting', rows=[],
                  startup_to_health_s=None, observed_peak_server_rss_mib=None,
                  manual_semantic_review_required=True, threads=args.threads)
    process = None
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = 'http://127.0.0.1:' + str(port)
    try:
        if not model.is_file() or model.stat().st_size != spec['size'] or sha256(model) != spec['sha256']:
            raise ValueError('Missing/invalid GGUF; run setup.py --models ' + name)
        command = [str(server), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
                   '-c', '2048', '-t', str(args.threads), '-ngl', '0', '--parallel', '1',
                   '--jinja', '--chat-template-kwargs', '{"enable_thinking":false}', '--perf']
        report['command'] = command
        with (directory / (name + '_server.log')).open('w') as log:
            started = time.perf_counter()
            process = subprocess.Popen(command, stdout=log, stderr=log)
            with MemorySampler(process.pid) as memory:
                try:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError('llama-server exited; inspect ' + name + '_server.log')
                        if time.perf_counter() - started > args.startup_timeout:
                            raise TimeoutError('Server startup timed out')
                        try:
                            if request(base, '/health', timeout=1).get('status') == 'ok':
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(0.2)
                    report['startup_to_health_s'] = time.perf_counter() - started
                    # Explicit discarded warm-up. Health time may include runtime internal warm-up.
                    first = next((describe(c['input'], c['source_model']) for c in inputs
                                  if describe(c['input'], c['source_model'])['claims']), None)
                    if first is not None:
                        generation(base, first, args.request_timeout)
                    for case in inputs:
                        template = describe(case['input'], case['source_model'])
                        row = dict(case_id=case['case_id'], source_model=case['source_model'],
                                   provenance=case['provenance'], template=template,
                                   payload_sha256=hashlib.sha256(json.dumps(case['input'], sort_keys=True).encode()).hexdigest(),
                                   manual_semantic_review='pending', manual_naturalness_review='pending')
                        if not template['claims']:
                            row.update(status='gate_not_called', reason=template['status'], manual_semantic_review='not_applicable',
                                       manual_naturalness_review='not_applicable')
                        else:
                            try:
                                raw, elapsed, finish, usage, timings = generation(base, template, args.request_timeout)
                                row.update(raw=raw, wall_s=elapsed, finish_reason=finish, usage=usage, timings=timings)
                                if finish != 'stop':
                                    raise ValueError('Incomplete generation: ' + str(finish))
                                row['parsed'] = audit(raw, template['claims'])
                                row['status'] = 'structure_pass_manual_review_pending'
                            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                                row.update(status='request_or_structure_failed', error=str(error))
                        report['rows'].append(row)
                        print(name, row['case_id'], row['status'], flush=True)
                        (directory / (name + '.json')).write_text(json.dumps(report, indent=2))
                    report['status'] = 'completed'
                finally:
                    report['observed_peak_server_rss_mib'] = None if memory.peak_kib is None else memory.peak_kib / 1024
    except Exception as error:
        report.update(status='failed', error=str(error))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        (directory / (name + '.json')).write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('0.6b', '1.7b'), default=['0.6b', '1.7b'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--startup-timeout', type=int, default=180)
    parser.add_argument('--request-timeout', type=int, default=180)
    parser.add_argument('--all-dev', action='store_true')
    parser.add_argument('--payload', type=Path, help='Optional exported live language payload JSON')
    parser.add_argument('--source', choices=('r18', 'r50', 'm3'), default='r18')
    args = parser.parse_args()
    if min(args.threads, args.startup_timeout, args.request_timeout) <= 0:
        parser.error('Thread count and timeouts must be positive')
    inputs = cases(args)
    for case in inputs:
        validate_payload(case['input'])
    servers = list((ROOT / 'local' / 'runtime').rglob('llama-server'))
    if len(servers) != 1:
        parser.error('Run llm_trial/setup.py first; expected one llama-server')
    inventory_path = ROOT / 'local' / 'runtime_inventory.json'
    if not inventory_path.is_file():
        parser.error('Runtime verification inventory missing; run setup.py again')
    for relative, digest in json.loads(inventory_path.read_text()).items():
        path = (ROOT / 'local' / 'runtime' / relative).resolve()
        if not path.is_relative_to((ROOT / 'local' / 'runtime').resolve()) or not path.is_file() or sha256(path) != digest:
            parser.error('Runtime verification failed; run setup.py again')
    directory = ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory.mkdir(parents=True)
    environment = dict(platform=platform.platform(), architecture=platform.machine(), python=sys.version,
                       runtime=json.loads((ROOT / 'runtime_manifest.json').read_text()),
                       runtime_asset=json.loads((ROOT / 'local' / 'runtime_asset.json').read_text()),
                       server_sha256=sha256(servers[0]), arguments=vars(args).copy(),
                       generation_settings=dict(temperature=0, seed=31, max_tokens=384, context=2048,
                                                enable_thinking=False, cache_prompt=False, response_format='json_object'),
                       system_prompt=SYSTEM, runner_sha256=sha256(Path(__file__)),
                       limitations=['Synthetic development fixtures by default, not real-image or held-out evaluation.',
                                    'Reference/JSON checks do not validate sentence semantics.',
                                    'Memory is sampled server PID RSS only, excludes HCI app, OS and other processes.',
                                    'One measured pass after discarded warm-up; no prompt-cache reuse; no cold-disk claim.'])
    environment['arguments']['payload'] = str(args.payload) if args.payload else None
    (directory / 'environment.json').write_text(json.dumps(environment, indent=2))
    models = json.loads((ROOT / 'model_manifest.json').read_text())
    reports = [run_model(name, models[name], args, inputs, directory, servers[0]) for name in dict.fromkeys(args.models)]
    columns = ['model', 'status', 'startup_to_health_s', 'observed_peak_server_rss_mib',
               'generated_requests', 'structure_pass', 'failed_requests', 'gated_requests', 'median_wall_s',
               'manual_semantic_review']
    with (directory / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            rows = report['rows']
            latency = [r['wall_s'] for r in rows if 'wall_s' in r]
            summary = {k: report[k] for k in columns[:4]}
            summary.update(generated_requests=sum(r['status'] != 'gate_not_called' for r in rows),
                           structure_pass=sum(r['status'] == 'structure_pass_manual_review_pending' for r in rows),
                           failed_requests=sum(r['status'] == 'request_or_structure_failed' for r in rows),
                           gated_requests=sum(r['status'] == 'gate_not_called' for r in rows),
                           median_wall_s=statistics.median(latency) if latency else None, manual_semantic_review='pending')
            writer.writerow(summary)
            print(json.dumps(summary), flush=True)
    with (directory / 'review.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'case_id', 'status', 'allowed_facts', 'raw_output', 'faithful_yes_no', 'natural_1_to_5', 'notes'])
        for report in reports:
            for row in report['rows']:
                writer.writerow([report['model'], row['case_id'], row['status'],
                                 json.dumps(row['template']['claims']), row.get('raw', ''), '', '', ''])
    print('TRIAL REPORT:', directory, flush=True)
    print('Manual semantic review remains pending. No app behavior or accuracy claim changed.', flush=True)
    return 1 if any(r['status'] != 'completed' for r in reports) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted. Server terminated; partial per-model report retained.', file=sys.stderr)
        sys.exit(130)
