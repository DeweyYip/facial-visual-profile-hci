"""Local Gradio comparison of three fixed Seed1 models."""
import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from backend import Comparison, require
from language import language_outputs
from ui_presenter import summaries, explorer, focus, LABELS, CATEGORIES
from language import ATTRIBUTES

ROOT = Path(__file__).resolve().parent

def detail_rows(rows, evidence=False):
    """Format display copies; retain full precision in inference/evidence payloads."""
    formatted=[]
    for original in rows:
        row=list(original)
        label_index=1 if evidence else 0
        row[label_index]=LABELS.get(row[label_index],row[label_index].replace('_',' '))
        for index in ((3,4) if evidence else (1,2,3)):
            row[index]='—' if row[index] is None else f'{float(row[index]):.3f}'
        formatted.append(row)
    return formatted

def analyze_request(engine, path, kind, severity):
    def empty(message):
        return (None,None,None,[],[],message,engine.startup_rows(),'','','',[],None)
    if not path:
        return empty('Please upload a still portrait before comparing models.')
    try:
        with Image.open(path) as image:
            if getattr(image,'n_frames',1) != 1:
                return empty('Animated images are unsupported. Upload a still JPG, PNG or WebP image.')
            image.load()
            result = engine.infer(image,kind=kind,severity=int(severity),sample_id=Path(path).name)
        try:
            language = language_outputs(result,engine.models)
        except Exception as exc:
            print(f'Language input validation error: {type(exc).__name__}: {exc}',flush=True)
            language = ('Description unavailable: language evidence validation failed.',)*3+([],None)
        return (result['clean_canvas'],result['canvas'],result['overlay'],result['rows'],
                result['thresholds'],result['status'],result['system_rows'])+language
    except (UnidentifiedImageError,OSError) as exc:
        print(f'Image read error: {type(exc).__name__}: {exc}',flush=True)
        return empty('This file cannot be read as an image. Upload a valid still JPG, PNG or WebP image.')
    except Image.DecompressionBombError as exc:
        print(f'Image size error: {exc}',flush=True)
        return empty('This image is too large to process. Resize it and upload again.')
    except Exception as exc:
        print(f'Processing error: {type(exc).__name__}: {exc}',flush=True)
        message = ('Image aspect ratio is too extreme. Use a normal portrait.'
                   if 'aspect ratio' in str(exc) else 'Image processing failed. Try another image; see the terminal if it persists.')
        return empty(message)


def build_demo(engine):
    import gradio as gr
    theme = gr.themes.Base(
        font=['-apple-system','BlinkMacSystemFont','Segoe UI','sans-serif'],
        font_mono=['SFMono-Regular','Consolas','monospace'],
        primary_hue='stone',neutral_hue='stone',radius_size='lg').set(
        body_background_fill='#f6f4ee',body_background_fill_dark='#f6f4ee',
        body_text_color='#302f29',body_text_color_dark='#302f29',
        background_fill_primary='#fffefa',background_fill_primary_dark='#fffefa',
        background_fill_secondary='#f1eee6',background_fill_secondary_dark='#f1eee6',
        block_background_fill='#fffefa',block_background_fill_dark='#fffefa',
        border_color_primary='#e3dfd5',border_color_primary_dark='#e3dfd5',
        button_primary_background_fill='#373a32',button_primary_background_fill_dark='#373a32',
        button_primary_text_color='#fffefa',button_primary_text_color_dark='#fffefa')
    css=(ROOT/'styles.css').read_text()
    with gr.Blocks(title='Facial Visual Profile',theme=theme,css=css,analytics_enabled=False,
                   delete_cache=(60,300),elem_id='fvp-app') as demo:
        gr.HTML('<header class="page-header"><div class="wordmark" aria-label="FVP">F<span>V</span>P</div></header>'
                '<section class="page-intro"><h1>Facial Visual Profile</h1><p>One portrait. Three perspectives.</p></section>')
        with gr.Row(elem_id='portrait-workspace',equal_height=True):
            with gr.Column(scale=2,min_width=340,elem_id='portrait-main'):
                gr.HTML('<div class="panel-caption">01 <span>Your portrait</span></div>')
                upload=gr.File(label='Upload your image',file_types=['.jpg','.jpeg','.png','.webp','.bmp'],
                               type='filepath',show_label=False,height=440,elem_id='portrait-upload')
                clean=gr.Image(label='Your portrait',show_label=False,interactive=False,format='png',
                               height=392,visible=False,elem_id='portrait-preview',show_download_button=False,show_fullscreen_button=False)
                reload_image=gr.Button('Reload image',visible=False,variant='secondary',elem_id='reload-image')
            with gr.Column(scale=1,min_width=230,elem_id='portrait-results'):
                gr.HTML('<div class="panel-caption">02 <span>A different view</span></div>',elem_id='variation-caption')
                canvas=gr.Image(label='Transformed image',interactive=False,format='png',height=210,
                                elem_id='transformed-image',show_download_button=False,show_fullscreen_button=False)
                points=gr.Image(label='Landmarks',interactive=False,format='png',height=210,
                                elem_id='landmark-image',show_download_button=False,show_fullscreen_button=False)
        with gr.Row(elem_id='transform-controls'):
            corruption=gr.Dropdown(choices=[('Blur','blur'),('Darkening','brightness'),('Rotation','rotation'),
                                            ('Lower-face occlusion','occlusion'),('JPEG compression','jpeg')],
                                   value='blur',label='Image variation',interactive=True,scale=1,min_width=220)
            severity=gr.Slider(minimum=0,maximum=4,step=1,value=0,label='Intensity · 0 is original',interactive=True,scale=2,min_width=220)
        button=gr.Button('Compare models',variant='primary',elem_id='compare-button')
        gr.HTML('<p class="workspace-note">Each comparison starts from your original image.</p>')
        gr.HTML('<div class="section-heading"><span>03</span><h2>Portrait descriptions</h2></div>')
        initial=summaries(None)
        with gr.Row(elem_id='summary-grid',equal_height=True):
            r18_text=gr.HTML(initial[0],elem_classes=['model-summary'])
            r50_text=gr.HTML(initial[1],elem_classes=['model-summary'])
            m3_text=gr.HTML(initial[2],elem_classes=['model-summary'])
        gr.HTML('<div class="section-heading"><span>04</span><h2>Explore the attributes</h2></div>')
        category=gr.Radio(choices=list(CATEGORIES),value='All',label='Category',show_label=False,interactive=True,elem_id='attribute-categories')
        mode=gr.Radio(choices=['Predicted','Differences','All attributes'],value='Predicted',show_label=False,interactive=True,elem_id='attribute-mode')
        attribute_view=gr.HTML(explorer(None),elem_id='attribute-explorer')
        with gr.Accordion('Look closer at one attribute',open=False,elem_classes=['quiet-accordion']):
            attribute=gr.Dropdown(choices=[(LABELS[a],a) for a in ATTRIBUTES],value='Smiling',label='Attribute',interactive=True)
            attribute_detail=gr.HTML(focus(None,'Smiling'))
        with gr.Accordion('Model details & decision thresholds',open=False,elem_classes=['quiet-accordion']):
            gr.HTML('<div class="detail-overview">'
                    '<section><h3>Models</h3><p>Fixed Seed1 checkpoints, with validation-frozen thresholds for each model.</p></section>'
                    '<section><h3>Face landmarks</h3><p>M3 requires one face and five valid landmarks. R18 and R50 use the full image canvas.</p></section>'
                    '<section><h3>Descriptions</h3><p>Positive labels within 0.05 of their threshold and conflicting hair details are omitted.</p></section></div>')
            gr.HTML('<div class="detail-heading"><h3>Decision thresholds</h3><p>Scores at or above these values count as positive predictions.</p></div>')
            thresholds=gr.Dataframe(headers=['Attribute','R18','R50','M3'],
                                   datatype=['str']*4,interactive=False,label='Decision thresholds',show_label=False,
                                   max_height=280,elem_classes=['detail-table','threshold-table'])
            gr.HTML('<div class="detail-heading"><h3>Description evidence</h3><p>Source scores for the attributes included in each description. Values shown to three decimals.</p></div>')
            wording_evidence=gr.Dataframe(headers=['Model','Attribute','Phrase','Score','Threshold','Evidence ID'],
                datatype=['str']*6,interactive=False,label='Description evidence',show_label=False,
                max_height=280,elem_classes=['detail-table','evidence-table'])
            gr.HTML('<p class="detail-footnote">Landmarks: left eye · right eye · nose · left mouth · right mouth.<br>'
                    'Display markers only. No automatic cropping, alignment or EXIF rotation. Values are rounded for display only.</p>')
        with gr.Accordion('Advanced / Troubleshooting',open=False,elem_classes=['quiet-accordion']):
            system=gr.Dataframe(headers=['Component','Availability','Explanation'],datatype=['str','str','str'],
                                value=engine.startup_rows(),interactive=False,label='Component details',elem_classes=['detail-table'])
            gr.Markdown('After fixing missing model files or a component error, reload and compare again.')
            reload_button=gr.Button('Reload models',variant='secondary')
        gr.HTML('<footer class="page-footer"><span>Facial Visual Profile</span><span>Processed locally · model predictions, not verified facts</span></footer>')
        rows_state=gr.State([])
        status_state=gr.State('')
        payload_state=gr.State(None)
        outputs=[clean,canvas,points,rows_state,thresholds,status_state,system,r18_text,r50_text,m3_text,wording_evidence,payload_state,
                 attribute_view,attribute_detail]
        def cleared():
            return (None,None,None,[],[],'',engine.startup_rows(),*summaries(None),[],None,explorer(None),focus(None,'Smiling'))
        def clear_results():
            result=list(cleared())
            result[0]=gr.skip()
            return tuple(result)
        def upload_changed(path):
            result=list(cleared())
            if not path:
                result[0]=gr.update(value=None,visible=False)
                return tuple(result)+(gr.update(height=440,visible=True),gr.update(visible=False))
            try:
                with Image.open(path) as image:
                    if getattr(image,'n_frames',1)!=1:
                        raise ValueError('Upload a still image, not an animation.')
                    image.load()
                    # Display the original aspect ratio. The backend still receives
                    # the unchanged file and creates its frozen black 224px canvas.
                    preview=image.convert('RGB')
                    preview.thumbnail((1200,1200),Image.Resampling.BILINEAR)
                result[0]=gr.update(value=preview,visible=True)
                return tuple(result)+(gr.update(visible=False),gr.update(visible=True))
            except Exception as exc:
                print(f'Preview error: {type(exc).__name__}: {exc}',flush=True)
                gr.Warning('This image cannot be previewed. Try a still JPG, PNG or WebP.')
                result[0]=gr.update(value=None,visible=False)
                return tuple(result)+(gr.update(height=440,visible=True),gr.update(visible=False))
        # The original upload remains a File rather than a re-encoded Image input.
        # Keep the original filename and bytes for frozen JPEG/rotation behavior.
        upload.change(upload_changed,inputs=[upload],outputs=outputs+[upload,reload_image],queue=False,api_name=False)
        def reset_image():
            result=list(cleared())
            result[0]=gr.update(value=None,visible=False)
            return tuple(result)+(gr.update(value=None,height=440,visible=True),gr.update(visible=False))
        reload_image.click(reset_image,outputs=outputs+[upload,reload_image],queue=False,api_name=False)
        for control in (corruption,severity):
            control.change(clear_results,outputs=outputs,queue=False,api_name=False)
        def analyze(path,kind,level,selected_category,selected_mode,selected_attribute):
            result=list(analyze_request(engine,path,kind,level))
            # Keep the unpadded upload preview; the two result views remain
            # the actual 224px model/detector canvases.
            result[0]=gr.skip()
            payload=result[11]
            result[4]=detail_rows(result[4])
            result[10]=detail_rows(result[10],evidence=True)
            if payload is None:
                gr.Warning(result[5] or 'Unable to compare this image. Try another portrait.')
                result[7:10]=summaries(None)
            else:
                result[7:10]=summaries(payload)
            return tuple(result)+(explorer(payload,selected_category,selected_mode),focus(payload,selected_attribute))
        def render_explorer(payload,selected_category,selected_mode):
            return explorer(payload,selected_category,selected_mode)
        category.change(render_explorer,inputs=[payload_state,category,mode],outputs=attribute_view,queue=False,api_name=False)
        mode.change(render_explorer,inputs=[payload_state,category,mode],outputs=attribute_view,queue=False,api_name=False)
        attribute.change(focus,inputs=[payload_state,attribute],outputs=attribute_detail,queue=False,api_name=False)
        controls=[upload,reload_image,corruption,severity,button,reload_button,category,mode,attribute]
        def lock_controls():return tuple(gr.update(interactive=False) for _ in controls)
        def unlock_controls():return tuple(gr.update(interactive=True) for _ in controls)
        lock=button.click(lock_controls,outputs=controls,queue=False,api_name=False)
        prediction=lock.then(analyze,inputs=[upload,corruption,severity,category,mode,attribute],outputs=outputs,
                             concurrency_limit=1,concurrency_id='engine',api_name=False)
        prediction.then(unlock_controls,outputs=controls,queue=False,api_name=False)
        def reload_models():
            try:
                engine.reload()
                gr.Info('Models reloaded. Compare your portrait again.')
            except Exception as exc:
                print(f'Reload error: {type(exc).__name__}: {exc}',flush=True)
                gr.Warning('Reload failed. Check component details below.')
            return clear_results()
        reload_lock=reload_button.click(lock_controls,outputs=controls,queue=False,api_name=False)
        reloaded=reload_lock.then(reload_models,outputs=outputs,concurrency_limit=1,concurrency_id='engine',api_name=False)
        reloaded.then(unlock_controls,outputs=controls,queue=False,api_name=False)
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,default=ROOT.parent/'weights')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--image',type=Path,help='Optional real single-face image for the smoke check')
    args = parser.parse_args()
    engine = Comparison(args.weights)
    if args.smoke:
        require(len(engine.models)==3 and engine.detector is not None, 'Runtime smoke requires all three models and detector. Launch the page for component-specific errors.')
        blank = engine.infer(Image.new('RGB',(224,224)))
        require(not blank['geometry_valid'] and 'm3' not in blank['results'], 'Blank-image geometry must be unavailable')
        require(len(blank['rows']) == 24, 'Expected 24 output rows')
        print('PASS: real R18/R50 inference; actual blank-image detection; M3 unavailable on failure.')
        # Test M3 wiring separately using synthetic coordinates, never as a UI detection result.
        from backend import prepare_image
        import torch
        _, tensor = prepare_image(Image.new('RGB',(224,224)))
        synthetic = [[70,80],[150,80],[110,120],[80,160],[140,160]]
        result = engine.models['m3'].infer(tensor,synthetic)
        require(np.isfinite(result['scores']).all(), 'M3 forward failed')
        print('PASS: real M3 weights forward with explicitly synthetic points; this does not validate face detection.')
        # Exercise all 21 conditions with actual model inference and detector calls.
        from backend import LEVELS, apply_corruption, rgb_sha256, normalize_canvas
        import hashlib
        probe = Image.fromarray(np.random.default_rng(28).integers(0,256,(218,178,3),dtype=np.uint8))
        clean, _ = prepare_image(probe)
        conditions = [('blur',0)]+[(kind,level) for kind in LEVELS for level in range(1,5)]
        for kind, level in conditions:
            outcome = engine.infer(probe,kind=kind,severity=level,sample_id='day28_smoke.png')
            expected, _, metadata = apply_corruption(clean,kind,level,sample_id='day28_smoke.png',seed=0)
            require(outcome['canvas'].tobytes()==expected.tobytes(), 'Corruption pixel mismatch')
            require(outcome['corruption']==metadata, 'Corruption metadata mismatch')
            require(len(outcome['rows'])==24, 'Comparison row mismatch')
        print('PASS: all 21 conditions; final pixels match frozen transforms; detector rerun for every request.')
        if args.image:
            with Image.open(args.image) as image:
                actual = engine.infer(image,sample_id=args.image.name)
            print(actual['status'])
            require(actual['geometry_valid'] and 'm3' in actual['results'], 'Real-image check requires one detected face')
            print('PASS: actual detected points reached M3; all three models returned 24 scores.')
        print('SMOKE PASSED. Runtime checks only, not an accuracy benchmark.')
    else:
        build_demo(engine).queue(default_concurrency_limit=1).launch(
            server_name='127.0.0.1',server_port=7860,share=True,inbrowser=True)

if __name__ == '__main__':
    main()
