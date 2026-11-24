siglip = ROOT/'weights/siglip2-base-patch16-224'
dinov3 = ROOT/'weights/convnext_tiny.dinov3_lvd1689m.safetensors'
# Download only when this experiment is selected; verify pinned files below.
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
if not (siglip / 'model.safetensors').exists():
    snapshot_download('google/siglip2-base-patch16-224', revision=ex.SIGLIP_REV,
                      allow_patterns=['*.json', '*.model', 'model.safetensors'], local_dir=siglip)
if not dinov3.exists():
    weight = hf_hub_download('timm/' + ex.V3_MODEL, 'model.safetensors', revision=ex.V3_REV)
    shutil.copyfile(weight, dinov3)
issues = [ex.verify(siglip/'model.safetensors', ex.SIGLIP_SHA), ex.verify(dinov3, ex.V3_SHA)]
if any(issues):
    raise FileNotFoundError('; '.join(problem for problem in issues if problem))

siglip_images, siglip_bank = ex.siglip_embeddings(
    siglip, rows, source, release_prompts, nola_eval.loader, EXTRACT_BATCH)
siglip_logits = score_clip(siglip_images, siglip_bank, clip_scale)
v3_features, v3_preprocessing = ex.dinov3_features(dinov3, rows, source, nola_eval.loader, EXTRACT_BATCH)

# Match anchor counts across nomination encoders before crossing with DINO features.
nomination = {'clip': release_logits, 'siglip2': siglip_logits}
encoder_cap = max(1, len(rows['train']) // (5*len(names)))
pools = {name: ex.candidates(scores['train'], encoder_cap)
         for name, scores in nomination.items()}
quota = [min(len(pools['clip'][c]), len(pools['siglip2'][c])) for c in range(len(names))]
matched_anchors = {name: ex.pack(pool, quota) for name, pool in pools.items()}

export_geometry(out, key, 'siglip2', siglip_images['val'], truth['val'], rows['val'], names)
export_geometry(out, key, 'dinov3', v3_features['val'], truth['val'], rows['val'], names)
