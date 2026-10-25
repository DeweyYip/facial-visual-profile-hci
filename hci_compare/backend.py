"""Verified Seed1 model loading and final-canvas inference."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MODELS = ('r18', 'r50', 'm3')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def verify_vendor():
    for name, digest in json.loads((ROOT/'vendor_hashes.json').read_text()).items():
        require(sha((ROOT/name).read_bytes()) == digest, f'Vendor hash mismatch: {name}')

verify_vendor()
from vendor.resnet18 import ResNet18Attributes
from vendor.resnet50 import ResNet50Attributes
from vendor.resnet18_geometry import ResNet18GeometryAttributes
from vendor.frozen.benchmark_preprocessing import normalize_canvas, normalize_points
from vendor.frozen.corruptions import apply_corruption, rgb_sha256, LEVELS

CONSTRUCTORS = dict(r18=ResNet18Attributes, r50=ResNet50Attributes, m3=ResNet18GeometryAttributes)

def prepare_image(image):
    """Same frozen image letterbox without requiring or fabricating GT points."""
    width, height = image.size
    scale = 224 / max(width, height)
    size = (round(width*scale), round(height*scale))
    require(min(size) >= 1, 'Image aspect ratio is too extreme')
    resized = image.convert('RGB').resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (224,224), (0,0,0))
    canvas.paste(resized, ((224-size[0])//2, (224-size[1])//2))
    return canvas, normalize_canvas(canvas)

def sigmoid(logits):
    x = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1/(1+np.exp(-x[positive]))
    e = np.exp(x[~positive]); out[~positive] = e/(1+e)
    return out

class AttributeModel:
    def __init__(self, name, archive_path):
        self.name = name
        binding = json.loads((ROOT/'execution_binding.json').read_text())['models'][name+'_seed1']
        with zipfile.ZipFile(archive_path) as z:
            require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate archive members')
            prefix = 'runs/'+name+'_seed1/'
            raw_config = z.read(prefix+'config.json')
            require(sha(raw_config) == binding['config_sha256'], name+': config mismatch')
            config = json.loads(raw_config)
            require(config['source_sha256'] == binding['source_sha256'], name+': source binding mismatch')
            require(config['seed'] == 1 and config['run_name'] == name+'_seed1', name+': run mismatch')
            for relative, digest in binding['source_sha256'].items():
                require(sha(z.read(prefix+'source/'+relative)) == digest, name+': archived source mismatch: '+relative)
            best = json.loads(z.read(prefix+'best.json'))
            require(best['epoch'] == binding['selected_epoch'] and best.get('checkpoint',f"epoch_{best['epoch']:02d}.pt") == binding['checkpoint'], name+': selection mismatch')
            raw_threshold = z.read('runs/benchmark_preparation/seed1/'+name+'_seed1.json')
            require(sha(raw_threshold) == binding['threshold_sha256'], name+': thresholds mismatch')
            t = json.loads(raw_threshold)
            require(t['checkpoint_sha256'] == binding['checkpoint_sha256'] and t['fit_split'] == 'valid', name+': threshold binding mismatch')
            self.attributes = config['attribute_names']
            require(len(self.attributes) == 24 and t['attribute_names'] == self.attributes, name+': attribute mismatch')
            self.thresholds = np.asarray(t['thresholds'], dtype=np.float64)
            require(self.thresholds.shape == (24,) and np.isfinite(self.thresholds).all(), name+': invalid thresholds')
            # Stream to a seekable spool to avoid retaining optimizer-state bytes twice.
            import tempfile
            with tempfile.SpooledTemporaryFile(max_size=8*1024*1024) as checkpoint_file:
                digest = hashlib.sha256()
                with z.open(prefix+binding['checkpoint']) as source:
                    for chunk in iter(lambda: source.read(8*1024*1024), b''):
                        digest.update(chunk); checkpoint_file.write(chunk)
                require(digest.hexdigest() == binding['checkpoint_sha256'], name+': checkpoint hash mismatch')
                checkpoint_file.seek(0)
                allowlist = getattr(torch.serialization, 'safe_globals', None)
                if allowlist:
                    from torch.torch_version import TorchVersion
                    context = allowlist([TorchVersion])
                else:
                    context = contextlib.nullcontext()
                try:
                    with context:
                        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
                except Exception as exc:
                    raise RuntimeError(name+': restricted checkpoint loading failed; keep weights_only=True and share this error for a compatible inference export.') from exc
        self.model = CONSTRUCTORS[name](num_attributes=24, pretrained=False).cpu().float()
        self.model.load_state_dict(checkpoint[binding['state_key']], strict=True)
        self.model.eval()
        print(f'Verified {name.upper()} Seed1 {binding["checkpoint"]}; strict loading passed.', flush=True)

    def infer(self, image_tensor, points=None):
        with torch.inference_mode():
            batch = image_tensor.unsqueeze(0)
            if self.name == 'm3':
                require(points is not None, 'M3 requires actual detected landmarks')
                logits = self.model(batch, normalize_points(points).unsqueeze(0), torch.tensor([True]))
            else:
                logits = self.model(batch)
        values = logits.detach().cpu().numpy()[0].astype(np.float64)
        require(values.shape == (24,) and np.isfinite(values).all(), self.name+': invalid logits')
        scores = sigmoid(values)
        return dict(logits=values, scores=scores, decisions=scores >= self.thresholds)

def validate_detection(result):
    if not isinstance(result, dict):
        return False, 'invalid_output'
    status = result.get('status')
    if status != 'ok':
        return False, status if status in ('no_face','detector_error','detector_unavailable') else 'invalid_output'
    count = result.get('face_count')
    if type(count) is not int or count < 0:
        return False, 'invalid_output'
    if count == 0:
        return False, 'no_face'
    if count > 1:
        return False, 'multiple_faces'
    try:
        points = np.asarray(result.get('landmarks'), dtype=np.float64)
    except (TypeError, ValueError):
        return False, 'invalid_keypoints'
    if points.shape != (5,2) or not np.isfinite(points).all() or (points < 0).any() or (points > 223).any():
        return False, 'invalid_keypoints'
    return True, 'ok'

DETECTION_MESSAGES = {
    'ok': 'One face detected; five valid landmarks.',
    'no_face': 'No face detected. Use a clear single-face portrait or reduce the corruption.',
    'multiple_faces': 'Multiple faces detected. Upload a portrait containing only one person.',
    'invalid_keypoints': 'Landmarks are missing, malformed or outside the image. Try another portrait or reduce the corruption.',
    'invalid_output': 'The detector returned invalid data. Try another portrait.',
    'detector_error': 'Face detection failed. Try again; see the terminal if it persists.',
    'detector_unavailable': 'Face detector is unavailable. Restore its dependencies and click Reload models.',
}

def load_error_message(exc):
    text = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return 'Model asset ZIP is missing. Restore it in weights/ and click Reload models.'
    if isinstance(exc, zipfile.BadZipFile):
        return 'Model asset ZIP is unreadable. Restore the original ZIP and click Reload models.'
    if any(word in text for word in ('mismatch','invalid thresholds')):
        return 'Model asset verification failed. Restore the original matching model ZIP.'
    if 'restricted checkpoint' in text:
        return 'Checkpoint is incompatible with this runtime. Share the terminal error; keep restricted loading enabled.'
    return 'Model could not load. See the terminal for details, fix the assets, then click Reload models.'


def overlay(canvas, detection, valid):
    image = canvas.copy()
    if valid:
        draw = ImageDraw.Draw(image)
        colors = ['#ff5252','#44ddff','#ffff55','#66ff66','#ff66dd']
        for number, ((x,y), color) in enumerate(zip(detection['landmarks'], colors), 1):
            draw.ellipse((x-3,y-3,x+3,y+3), fill=color, outline='black')
            draw.text((x+4,y-7), str(number), fill=color, stroke_width=1, stroke_fill='black')
    return image

class Comparison:
    def __init__(self, weights):
        import threading
        self._lock = threading.RLock()
        self.weights = Path(weights)
        self.attributes = json.loads((ROOT/'attribute_names.json').read_text())
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.reload()

    def reload(self):
        with self._lock:
            self.models = {}
            self.load_errors = {}
            self.detector = None
            self.detector_error = None
            r18 = self.weights/'r18_seed1_hci_assets.zip'
            candidates = [self.weights/'r50_m3_seed1_hci_assets.zip', self.weights/'r50_m3_seed1_hci_assets(1).zip']
            extra = next((p for p in candidates if p.is_file()), candidates[0])
            for name in MODELS:
                archive = r18 if name == 'r18' else extra
                try:
                    if not archive.is_file():
                        raise FileNotFoundError(archive)
                    model = AttributeModel(name, archive)
                    require(model.attributes == self.attributes, name+': attribute order mismatch')
                    self.models[name] = model
                except Exception as exc:
                    print(f'{name.upper()} load error: {type(exc).__name__}: {exc}', flush=True)
                    self.load_errors[name] = load_error_message(exc)
            try:
                self.detector = self._load_detector()
            except Exception as exc:
                print(f'Detector load error: {type(exc).__name__}: {exc}', flush=True)
                self.detector_error = 'Detector could not load. Check facenet-pytorch installation and terminal details, then click Reload models.'
            return self.startup_rows()

    def _load_detector(self):
        import facenet_pytorch
        import importlib.metadata
        provenance = json.loads((ROOT/'detector_provenance.json').read_text())
        require(importlib.metadata.version('facenet-pytorch') == provenance['version'], 'Detector dependency version mismatch')
        package_root = Path(facenet_pytorch.__file__).parent
        for relative, digest in provenance['weights_sha256'].items():
            require(sha((package_root/relative).read_bytes()) == digest, 'Detector weight mismatch: '+relative)
        from vendor.mtcnn import MTCNNDetector
        detector = MTCNNDetector(device='cpu')
        print('MTCNN ready: actual detection on each final 224x224 canvas.', flush=True)
        return detector

    def startup_rows(self):
        rows = [['Detector', 'Ready' if self.detector is not None else 'Unavailable',
                 'Ready for actual final-image detection.' if self.detector is not None else self.detector_error]]
        for name in MODELS:
            ready = name in self.models
            message = 'Verified Seed1 weights loaded.' if ready else self.load_errors.get(name,'Model is unavailable.')
            if name == 'm3' and ready and self.detector is None:
                ready = False
                message = 'Weights loaded, but the face detector is unavailable.'
            rows.append([name.upper(), 'Ready' if ready else 'Unavailable', message])
        return rows

    def summary(self):
        return f"{len(self.models)}/3 models loaded. Detector: {'ready' if self.detector is not None else 'unavailable'}."

    def infer(self, image, kind='blur', severity=0, sample_id=None):
        with self._lock:
            return self._infer(image, kind, severity, sample_id)

    def _infer(self, image, kind, severity, sample_id):
        started = time.perf_counter()
        require(kind in LEVELS, 'Unknown corruption')
        require(type(severity) is int and severity in range(5), 'Severity must be integer 0..4')
        clean_canvas, _ = prepare_image(image)
        if sample_id is None:
            sample_id = rgb_sha256(clean_canvas)+'.png'
        canvas, _, corruption = apply_corruption(
            clean_canvas, kind, severity, sample_id=sample_id, seed=0, gt_pixels=None)
        tensor = normalize_canvas(canvas)
        if self.detector is None:
            detection = dict(status='detector_unavailable',face_count=0)
        else:
            try:
                detection = self.detector.detect(canvas)
            except Exception as exc:
                print(f'Detector error: {type(exc).__name__}: {exc}', flush=True)
                detection = dict(status='detector_error',face_count=0)
        valid, reason = validate_detection(detection)
        results = {}
        unavailable = {}
        for name in MODELS:
            if name not in self.models:
                unavailable[name] = self.load_errors.get(name,'Model is not loaded.')
                continue
            if name == 'm3' and not valid:
                unavailable[name] = DETECTION_MESSAGES[reason]
                continue
            try:
                result = self.models[name].infer(tensor,detection['landmarks']) if name == 'm3' else self.models[name].infer(tensor)
                scores = np.asarray(result['scores'])
                decisions = np.asarray(result['decisions'])
                require(scores.shape == (24,) and decisions.shape == (24,) and np.isfinite(scores).all(), name+': invalid prediction output')
                results[name] = result
            except Exception as exc:
                print(f'{name.upper()} inference error: {type(exc).__name__}: {exc}', flush=True)
                unavailable[name] = 'Inference failed for this image. Try again; see the terminal if it persists.'
        rows = []
        for index, attribute in enumerate(self.attributes):
            row = [attribute]
            for name in MODELS:
                result = results.get(name)
                row += ([float(result['scores'][index]), 'Present' if result['decisions'][index] else 'Absent']
                        if result is not None else [None,'Unavailable'])
            rows.append(row)
        thresholds = [[attribute]+[float(self.models[name].thresholds[i]) if name in self.models else None for name in MODELS]
                      for i,attribute in enumerate(self.attributes)]
        count = detection.get('face_count',0) if isinstance(detection,dict) else 0
        status = f"{describe_corruption(corruption)} | {DETECTION_MESSAGES[reason]}"
        status += ' | M3: '+('available' if 'm3' in results else 'unavailable')
        if unavailable:
            status += ' | Unavailable: '+', '.join(name.upper() for name in unavailable)
        if not valid:
            status += ' | RGB outputs, when available, describe the whole canvas.'
        status += f' | CPU total: {time.perf_counter()-started:.3f} s'
        system_rows = [['Detector','Ready' if valid else 'Unavailable for this image',DETECTION_MESSAGES[reason]]]
        for name in MODELS:
            system_rows.append([name.upper(),'Available' if name in results else 'Unavailable',
                                'Fresh inference completed.' if name in results else unavailable[name]])
        return dict(clean_canvas=clean_canvas,canvas=canvas,corruption=corruption,
                    overlay=overlay(canvas,detection,valid),rows=rows,thresholds=thresholds,status=status,
                    results=results,detection=detection,geometry_valid=valid,geometry_reason=reason,system_rows=system_rows)


def describe_corruption(metadata):
    if metadata['severity'] == 0:
        return 'Clean (severity 0)'
    kind = metadata['kind']; parameters = metadata['parameters']
    if kind == 'blur':
        detail = f"sigma {parameters['sigma_pixels']:g} px"
    elif kind == 'brightness':
        detail = f"brightness x{parameters['factor']:g}"
    elif kind == 'rotation':
        detail = f"rotation {parameters['angle_degrees_ccw']:+g} degrees CCW"
    elif kind == 'occlusion':
        detail = f"lower-face square {parameters['side_pixels']} px"
    else:
        detail = f"JPEG quality {parameters['quality']}"
    return f"{kind} / severity {metadata['severity']} / {detail}"
