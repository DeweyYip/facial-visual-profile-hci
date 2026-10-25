"""Display-only views over the frozen language payload; no inference or thresholds changed."""
from html import escape
from language import ATTRIBUTES, MODELS, PHRASES, validate_payload, describe, agreement

CATEGORIES = {
    'All': set(ATTRIBUTES),
    'Expression': {'Smiling','Mouth_Slightly_Open'},
    'Hair': {'Bangs','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Bald','Straight_Hair','Wavy_Hair','Receding_Hairline'},
    'Face': {'Bushy_Eyebrows','Arched_Eyebrows','High_Cheekbones','Bags_Under_Eyes','Big_Lips','Big_Nose','Double_Chin','Oval_Face'},
    'Facial hair': {'Mustache','No_Beard','Sideburns'},
    'Accessories': {'Eyeglasses','Wearing_Hat'},
}
LABELS = {a: a.replace('_',' ').capitalize() for a in ATTRIBUTES}
LABELS.update(No_Beard='No visible beard',Mouth_Slightly_Open='Mouth slightly open',Big_Lips='Full lips',Big_Nose='Large nose')


def empty_explorer():
    return '<div class="quiet-empty">Compare a portrait to explore its attributes.</div>'


def summaries(payload):
    cards=[]
    for name in MODELS:
        if payload is None:
            text='Your portrait description will appear here.'
            notes=[]
        else:
            result=describe(payload,name)
            phrases=[c['phrase'] for c in result['claims']]
            if phrases:
                joined=phrases[0] if len(phrases)==1 else ', '.join(phrases[:-1])+' and '+phrases[-1]
                text='The model predicts '+joined+'.'
            else:
                text=result['text'].split('\n')[0]
            notes=[]
            near=sum(c['reason']=='near_threshold' for c in result['withheld'])
            if near: notes.append(str(near)+' near-threshold label'+('s' if near!=1 else '')+' omitted.')
            if any(c['reason'] in ('overlapping_hair_labels','overlap_with_baldness') for c in result['withheld']):
                notes.append('Conflicting hair details omitted.')
            if any(c['reason']=='summary_length_limit' for c in result['withheld']):
                notes.append('Six attributes shown. Explore all predictions below.')
        subtitle={'r18':'ResNet18','r50':'ResNet50','m3':'ResNet18 + landmarks'}[name]
        cards.append('<article class="summary-card"><div class="model-heading"><b>'+name.upper()+
                     '</b><span>'+subtitle+'</span></div><p>'+escape(text)+'</p>'+
                     ''.join('<small>'+escape(note)+'</small>' for note in notes)+'</article>')
    return tuple(cards)


def explorer(payload, category='All', mode='Predicted'):
    if payload is None: return empty_explorer()
    validate_payload(payload)
    if category not in CATEGORIES or mode not in ('Predicted','Differences','All attributes'):
        raise ValueError('Unknown attribute filter')
    rows=[]
    states={r['attribute']:r['state'] for r in agreement(payload)}
    count=0
    for index,attribute in enumerate(ATTRIBUTES):
        if attribute not in CATEGORIES[category]:continue
        cells=[payload['models'][m]['attributes'][index] for m in MODELS]
        if mode=='Predicted' and not any(c['decision'] is True for c in cells):continue
        if mode=='Differences' and states[attribute]!='mixed':continue
        count+=1
        marks=[]
        for cell in cells:
            value=cell['decision']
            state,label=('positive','Predicted') if value is True else ('negative','Not predicted') if value is False else ('unavailable','Unavailable')
            marks.append('<span class="attribute-cell '+state+'" role="cell" aria-label="'+label+'" title="'+label+'"><span class="result-dot" aria-hidden="true"></span><span class="prediction-word">'+label+'</span></span>')
        rows.append('<div class="attribute-row" role="row"><span class="attribute-label" role="rowheader">'+escape(LABELS[attribute])+'</span>'+''.join(marks)+'</div>')
    if not rows:
        return '<div class="quiet-empty">No attributes match this view. Try another filter.</div>'
    caption='RGB results refer to the whole image; no valid single-face description is available.' if not payload['detection']['geometry_valid'] else 'Predictions use each model’s own threshold.'
    return '<div class="attribute-grid" role="table" aria-label="Filtered attribute predictions"><div class="attribute-row attribute-header" role="row"><span>'+str(count)+' attributes</span><b>R18</b><b>R50</b><b>M3</b></div>'+''.join(rows)+'</div><p class="mobile-legend">● Predicted · ○ Not predicted · – Unavailable</p><p class="fine-print">'+caption+'</p>'


def focus(payload, attribute):
    if payload is None:return '<div class="quiet-empty">Select an attribute after comparing.</div>'
    validate_payload(payload)
    if attribute not in ATTRIBUTES:raise ValueError('Unknown attribute')
    i=ATTRIBUTES.index(attribute);views=[]
    for name in MODELS:
        r=payload['models'][name]['attributes'][i]
        if r['score'] is None:
            views.append('<div class="score-item"><b>'+name.upper()+'</b><span>Unavailable</span></div>')
        else:
            score=r['score'];threshold=r['threshold']
            views.append('<div class="score-item"><div><b>'+name.upper()+'</b><span>Score '+f'{score:.3f}'+' · threshold '+f'{threshold:.3f}'+'</span></div>'+
                         '<div class="score-track"><div class="score-fill" style="width:'+f'{score*100:.4f}'+'%"></div><i style="left:'+f'{threshold*100:.4f}'+'%" title="Frozen decision threshold"></i></div></div>')
    return '<div class="score-detail">'+''.join(views)+'<p class="fine-print">Scores are not calibrated probabilities. The vertical mark is the decision threshold.</p></div>'
