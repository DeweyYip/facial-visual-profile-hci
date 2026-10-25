"""11/2 language acceptance, fixed post-freeze synthetic inputs; not an accuracy benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from language import validate_payload, describe, audit_description
from local_wording import LocalWording, optional_description

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / 'acceptance_nov02'


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen():
    freeze = json.loads((SUITE / 'freeze.json').read_text())
    if freeze['protocol'] != 'nov02_acceptance_v1':
        raise ValueError('Unsupported acceptance protocol')
    for relative, expected in freeze['policy_files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or checksum(path) != expected:
            raise ValueError('Frozen policy file mismatch: ' + relative + '. Install v5 and v5.1; do not edit the freeze file to bypass this check.')
    case_file = SUITE / 'cases.json'
    if checksum(case_file) != freeze['cases_sha256']:
        raise ValueError('Acceptance input file changed; results cannot use this frozen protocol.')
    suite = json.loads(case_file.read_text())
    if not all(suite[k] is True for k in ('not_real_images','not_accuracy_evaluation','not_preexisting_holdout')):
        raise ValueError('Unexpected input provenance')
    cases = suite['cases']
    if len(cases) != 20 or len({c['case_id'] for c in cases}) != 20:
        raise ValueError('Expected twenty unique cases')
    # Detect accidental reuse of the original development payloads.
    development_file = ROOT / 'development_cases.json'
    def payload_id(payload):
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    old = {payload_id(c['input']) for c in json.loads(development_file.read_text())['cases']}
    current = [payload_id(c['input']) for c in cases]
    if len(set(current)) != 20 or any(key in old for key in current):
        raise ValueError('Acceptance payloads overlap or duplicate development inputs')
    for case in cases:
        if case['synthetic'] is not True:
            raise ValueError('Unexpected case provenance')
        validate_payload(case['input'])
    return freeze, suite


def run_case(case, service=None):
    payload = case['input']; source = case['source_model']
    started = time.perf_counter()
    template = describe(payload, source)
    audit_description(payload, template)
    actual = dict(status=template['status'], claim_attributes=[c['attribute'] for c in template['claims']])
    passed = actual == case['expected']
    row = dict(case_id=case['case_id'], scenario=case['scenario'], source_model=source,
               expected=case['expected'], actual=actual, template_contract_pass=passed,
               allowed_claims=template['claims'], template_text=template['text'])
    if service is None:
        row.update(local_status='not_run', local_contract_pass=None, displayed_text=template['text'])
    else:
        text, message, metadata = optional_description(service, payload, source)
        local_status = metadata['status']
        if not template['claims']:
            local_pass = local_status == 'gate_not_called' and text == template['text']
        elif local_status == 'template_fallback':
            local_pass = text == template['text']
        else:
            # The display renderer uses the finite phrases already checked against each evidence id.
            local_pass = (local_status == 'accepted_local_wording' and metadata['source_model'] == source
                          and metadata['evidence_ids'] == [c['evidence_id'] for c in template['claims']]
                          and all(note in text for note in template['notes']))
        row.update(local_status=local_status, local_contract_pass=local_pass, displayed_text=text,
                   local_message=message, local_audit=metadata)
    row['elapsed_s'] = round(time.perf_counter() - started,3)
    return row


def summarize(rows, local, expected_total=20):
    template_pass = sum(r.get('template_contract_pass') is True for r in rows)
    local_pass = sum(r.get('local_contract_pass') is True for r in rows)
    fallback = sum(r.get('local_status') == 'template_fallback' for r in rows)
    generated = sum(r.get('local_status') == 'accepted_local_wording' for r in rows)
    gated = sum(r.get('local_status') == 'gate_not_called' for r in rows)
    system_pass = len(rows) == expected_total and template_pass == expected_total and (not local or local_pass == expected_total)
    return dict(total_expected=expected_total, completed=len(rows), template_contract_pass=template_pass,
                local_contract_pass=local_pass if local else None, accepted_local_generations=generated,
                template_fallbacks=fallback, gated_without_generation=gated if local else None,
                system_contract_pass=system_pass,
                local_generation_without_fallback=(system_pass and local and fallback == 0),
                real_image_accuracy_evaluated=False, preexisting_held_out_evaluated=False)


def review_markdown(report):
    lines = ['# 11/2 language acceptance review', '',
             'Synthetic post-freeze acceptance inputs. Not real-image accuracy or a pre-existing held-out evaluation.', '',
             'For each row compare the displayed text with the allowed claims. Check only evidence fidelity, omissions and clarity; do not infer photo accuracy.', '']
    for row in report['rows']:
        lines += ['## ' + row['case_id'] + ' · ' + row['scenario'], '',
                  'Source: ' + row['source_model'] + '; local status: ' + row.get('local_status','error'), '',
                  'Allowed attributes: ' + ', '.join(c['attribute'] for c in row.get('allowed_claims',[])), '',
                  '```text', row.get('displayed_text', row.get('error','')), '```', '',
                  'Reviewer: faithful / omission / unsupported claim / clear wording / notes: ______', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true', help='Also run the actual installed Qwen3-1.7B; may take several minutes')
    args = parser.parse_args()
    freeze, suite = load_frozen()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = SUITE / 'results' / stamp
    directory.mkdir(parents=True)
    report = dict(protocol=suite['protocol'], provenance=suite['provenance'], frozen_policy=freeze,
                  runner_sha256=checksum(Path(__file__)), local_requested=args.local, status='running', rows=[],
                  manual_review_status='pending', limitations=[
                      'New synthetic inputs created after policy freeze; no pre-existing held-out split.',
                      'No images, detector or visual model inference are evaluated.',
                      'Finite-phrase contract checks do not measure unrestricted natural language quality.',
                      'A correct template fallback counts as system recovery, not successful local generation.'])
    service = LocalWording(ROOT.parent) if args.local else None
    interrupted = False
    try:
        for case in suite['cases']:
            try:
                row = run_case(case, service)
            except Exception as exc:
                row = dict(case_id=case['case_id'],scenario=case['scenario'],source_model=case['source_model'],
                           template_contract_pass=False,local_contract_pass=False,error=str(exc),local_status='error')
            report['rows'].append(row)
            print(row['case_id'], 'template=' + str(row['template_contract_pass']), 'local=' + row['local_status'],flush=True)
            (directory / 'report.json').write_text(json.dumps(report, indent=2))
        report['status'] = 'completed'
    except KeyboardInterrupt:
        interrupted = True
        report['status'] = 'interrupted'
    finally:
        if service is not None:
            service.close()
        report['summary'] = summarize(report['rows'], args.local)
        (directory / 'report.json').write_text(json.dumps(report,indent=2))
        (directory / 'review.md').write_text(review_markdown(report))
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('REPORT:', directory / 'report.json',flush=True)
    print('Review displayed descriptions in review.md; upload report.json for review.',flush=True)
    if interrupted:
        return 130
    if not report['summary']['system_contract_pass']:
        return 1
    if args.local and report['summary']['template_fallbacks']:
        print('System recovery passed, but local fallbacks occurred. Inspect their reasons before acceptance.',flush=True)
        return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print('Acceptance not started:', error,file=sys.stderr)
        sys.exit(1)
