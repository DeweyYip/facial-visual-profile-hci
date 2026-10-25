"""Deterministic, evidence-linked descriptions. No LLM or network dependency."""
import json
import math
import re
from pathlib import Path

SCHEMA_VERSION = 'fvp_language_v1'
POLICY_VERSION = 'template_v1'
MODELS = ('r18','r50','m3')
ATTRIBUTES = tuple(json.loads((Path(__file__).parent/'attribute_names.json').read_text()))
WORDING_MARGIN = 0.05  # An abstention heuristic; not calibrated uncertainty or a new classifier threshold.
MAX_CLAIMS = 6
PHRASES = dict(zip(ATTRIBUTES, (
    'a smile', 'a slightly open mouth', 'eyeglasses', 'bangs', 'a hat', 'a mustache',
    'no visible beard', 'black hair', 'bushy eyebrows', 'arched eyebrows',
    'high cheekbones', 'under-eye bags', 'full lips', 'a large nose', 'a double chin',
    'an oval face shape', 'a receding hairline', 'blond hair', 'brown hair', 'gray hair',
    'baldness', 'straight hair', 'wavy hair', 'sideburns',
)))
COLOR_GROUP = ('Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair')
TEXTURE_GROUP = ('Straight_Hair','Wavy_Hair')
BLOCK_MESSAGES = {
    'no_face':'No facial description generated: no face was detected. Upload a clear single-person portrait.',
    'multiple_faces':'No facial description generated: multiple faces were detected. Upload one person only.',
    'invalid_keypoints':'No facial description generated: face landmarks are invalid. Try another image or reduce the corruption.',
    'invalid_output':'No facial description generated: the detector returned invalid data.',
    'detector_error':'No facial description generated: face detection failed. Try again.',
    'detector_unavailable':'No facial description generated: the face detector is unavailable. Check component status.',
}

def require(condition,message):
    if not condition: raise ValueError(message)

def build_payload(comparison, loaded_models):
    """Translate current inference only; never persist images or merge model scores."""
    result_models = comparison['results']
    d = comparison.get('detection')
    d = d if isinstance(d,dict) else {}
    geometry = bool(comparison.get('geometry_valid',False))
    face_count = d.get('face_count',0)
    face_count = face_count if type(face_count) is int and face_count>=0 else 0
    reason = comparison.get('geometry_reason')
    if reason is None:
        reason = 'ok' if geometry else d.get('status','invalid_output')
        if not geometry and reason == 'ok':
            reason = 'multiple_faces' if face_count>1 else 'invalid_keypoints'
    if reason not in ('ok',*BLOCK_MESSAGES): reason='invalid_output'
    status_rows = {r[0].lower():r[2] for r in comparison.get('system_rows',[])}
    payload = dict(schema_version=SCHEMA_VERSION, policy_version=POLICY_VERSION,
        score_meaning='uncalibrated_model_score',
        wording_policy=dict(positive_margin=WORDING_MARGIN,max_claims=MAX_CLAIMS,
                            note='Description-only abstention heuristic. Original model decisions and thresholds are unchanged.'),
        input=dict(final_rgb_sha256=comparison['corruption']['output_rgb_sha256'],
                   corruption_kind=comparison['corruption']['kind'],severity=comparison['corruption']['severity']),
        detection=dict(status=reason,face_count=face_count,geometry_valid=geometry),
        models={})
    for name in MODELS:
        result = result_models.get(name)
        model = loaded_models.get(name)
        attributes = []
        for index,attribute in enumerate(ATTRIBUTES):
            score = float(result['scores'][index]) if result is not None else None
            threshold = float(model.thresholds[index]) if model is not None else None
            decision = bool(result['decisions'][index]) if result is not None else None
            attributes.append(dict(evidence_id=name+':'+attribute,attribute=attribute,
                                   score=score,threshold=threshold,decision=decision))
        payload['models'][name] = dict(run_name=name+'_seed1',available=result is not None,
            reason=None if result is not None else status_rows.get(name,'Model result unavailable.'),attributes=attributes)
    validate_payload(payload)
    payload['agreement'] = agreement(payload)
    return payload

def validate_payload(payload):
    require(set(payload).issubset({'schema_version','policy_version','score_meaning','wording_policy','input','detection','models','agreement'}),'Unexpected language input fields')
    require(payload.get('schema_version')==SCHEMA_VERSION,'Unsupported language input version')
    require(payload.get('policy_version')==POLICY_VERSION,'Unsupported language policy')
    require(payload.get('score_meaning')=='uncalibrated_model_score','Unexpected score semantics')
    policy=payload.get('wording_policy',{})
    require(policy.get('positive_margin')==WORDING_MARGIN and policy.get('max_claims')==MAX_CLAIMS,'Unexpected wording policy')
    d=payload['detection']
    require(type(d['geometry_valid']) is bool and type(d['face_count']) is int and d['face_count']>=0,'Invalid detector status')
    if d['geometry_valid']:
        require(d['status']=='ok' and d['face_count']==1,'Invalid description face gate')
    else:
        require(d['status'] in BLOCK_MESSAGES,'Unknown blocked detection status')
    info=payload['input']
    require(isinstance(info['final_rgb_sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',info['final_rgb_sha256']) is not None,'Invalid final image hash')
    require(info['corruption_kind'] in ('blur','brightness','rotation','occlusion','jpeg'),'Invalid corruption family')
    require(type(info['severity']) is int and info['severity'] in range(5),'Invalid corruption severity')
    require(set(payload['models'])==set(MODELS),'Expected exactly three model slots')
    for name in MODELS:
        model=payload['models'][name]
        require(type(model['available']) is bool and model['run_name']==name+'_seed1','Invalid model identity')
        records=model['attributes']
        require([r['attribute'] for r in records]==list(ATTRIBUTES),'Language input attribute order mismatch')
        for record in records:
            require(record['evidence_id']==name+':'+record['attribute'],'Evidence identity mismatch')
            threshold=record['threshold']
            if threshold is not None:
                require(type(threshold) in (int,float) and math.isfinite(threshold) and 0<=threshold<=1,'Invalid threshold')
            if model['available']:
                score=record['score']
                require(type(score) in (int,float) and math.isfinite(score) and 0<=score<=1,'Invalid model score')
                require(threshold is not None and type(record['decision']) is bool,'Missing decision evidence')
                require(record['decision']==(score>=threshold),'Decision disagrees with its own frozen threshold')
            else:
                require(record['score'] is None and record['decision'] is None,'Unavailable model must not contain scores/decisions')
    if 'agreement' in payload:
        require(payload['agreement']==agreement(payload),'Agreement metadata mismatch')
    return True

def agreement(payload):
    output=[]
    available=[payload['models'][m] for m in MODELS if payload['models'][m]['available']]
    for index,attribute in enumerate(ATTRIBUTES):
        decisions=[m['attributes'][index]['decision'] for m in available]
        state=('insufficient_models' if len(decisions)<2 else
               'mixed' if len(set(decisions))>1 else
               'all_positive' if decisions[0] else 'all_negative')
        output.append(dict(attribute=attribute,available_models=len(decisions),state=state))
    return output

def eligible_records(payload,name):
    """Conservative wording policy, independent of the displayed classifier decisions."""
    model=payload['models'][name]
    if not payload['detection']['geometry_valid'] or not model['available']:
        return [],[],[]
    eligible=[];withheld=[];notes=[]
    for record in model['attributes']:
        if not record['decision']:
            withheld.append(dict(attribute=record['attribute'],reason='not_predicted_present'))
        elif record['score']+1e-12 < record['threshold']+WORDING_MARGIN:
            withheld.append(dict(attribute=record['attribute'],reason='near_threshold'))
        else:
            eligible.append(record)
    names={r['attribute'] for r in eligible}
    suppressed={}
    for group,label in ((COLOR_GROUP,'hair color'),(TEXTURE_GROUP,'hair texture')):
        if len(names.intersection(group))>1:
            suppressed.update({a:'overlapping_hair_labels' for a in names.intersection(group)})
            notes.append('Several '+label+' labels passed the wording filter; that detail is omitted.')
    if 'Bald' in names:
        for a in (*COLOR_GROUP,*TEXTURE_GROUP,'Bangs'):
            if a in names: suppressed[a]='overlap_with_baldness'
        if any(reason=='overlap_with_baldness' for reason in suppressed.values()):
            notes.append('Other hair details are omitted when baldness also passes the wording filter.')
    filtered=[]
    for record in eligible:
        name_=record['attribute']
        if name_ in suppressed:
            withheld.append(dict(attribute=name_,reason=suppressed[name_]))
        elif len(filtered)>=MAX_CLAIMS:
            withheld.append(dict(attribute=name_,reason='summary_length_limit'))
        else:
            filtered.append(record)
    near=sum(r['reason']=='near_threshold' for r in withheld)
    if near: notes.append(f'{near} positive label(s) are close to their decision thresholds and are omitted from the description.')
    if any(r['reason']=='summary_length_limit' for r in withheld):
        notes.append('The description shows at most six attributes; the comparison table retains all 24.')
    return filtered,withheld,notes

def describe(payload,name):
    validate_payload(payload)
    require(name in MODELS,'Unknown description source')
    d=payload['detection'];model=payload['models'][name]
    if not d['geometry_valid']:
        return dict(model=name,text=BLOCK_MESSAGES[d['status']],claims=[],withheld=[],notes=[],status='blocked_detection')
    if not model['available']:
        return dict(model=name,text=name.upper()+' description unavailable: this model did not return a valid result. Check component status.',
                    claims=[],withheld=[],notes=[],status='model_unavailable')
    records,withheld,notes=eligible_records(payload,name)
    claims=[dict(evidence_id=r['evidence_id'],attribute=r['attribute'],phrase=PHRASES[r['attribute']],
                 score=r['score'],threshold=r['threshold']) for r in records]
    text=(name.upper()+' predicts these visible attributes: '+ '; '.join(c['phrase'] for c in claims)+'.'
          if claims else name.upper()+': no attributes passed the description filter; see the full score table.')
    mixed=sum(a['state']=='mixed' for a in agreement(payload))
    if mixed:
        notes.append(f'Available models disagree on {mixed} attribute decision(s). This description uses only '+name.upper()+'.')
    if payload['input']['severity']:
        notes.append(f"Input: {payload['input']['corruption_kind']}, severity {payload['input']['severity']}. Description refers to this transformed input.")
    if notes: text+='\n'+'\n'.join(notes)
    return dict(model=name,text=text,claims=claims,withheld=withheld,notes=notes,status='generated' if claims else 'no_eligible_attributes')

def audit_description(payload,description):
    """Reject changed wording, facts, numbers, identity or evidence links."""
    expected=describe(payload,description.get('model'))
    require(description==expected,'Description or evidence differs from the deterministic grounded template')
    return True

def language_outputs(comparison,loaded_models):
    payload=build_payload(comparison,loaded_models)
    descriptions={name:describe(payload,name) for name in MODELS}
    evidence=[]
    for name,description in descriptions.items():
        audit_description(payload,description)
        for claim in description['claims']:
            evidence.append([name.upper(),claim['attribute'],claim['phrase'],claim['score'],claim['threshold'],claim['evidence_id']])
    return tuple(descriptions[name]['text'] for name in MODELS)+(evidence,payload)
