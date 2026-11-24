def load_data():
    nola_eval.seed_all(SEED)
    splits, source, names, bank_name = nola_eval.load_data(TAG, DATA, seed=SEED)
    rows = {part: splits[part] for part in ('train', 'val')}
    truth = {part: torch.tensor([y for _, y, _ in group]) for part, group in rows.items()}
    return rows, source, names, bank_name, truth


def load_clip():
    weights = torch.jit.load(str(ROOT / 'weights/ViT-B-32.pt'), map_location='cpu')
    model = nola_eval.clip.build_model(weights.state_dict())
    return model.float().cuda().eval().requires_grad_(False)


def make_cde(clip, names, bank_name):
    descriptions, templates = nola_eval.get_prompts(ROOT, bank_name)
    prompt_parts = [(descriptions[name], [t.format(name) for t in templates]) for name in names]
    prompts = [desc + tmpl for desc, tmpl in prompt_parts]
    return nola_eval.build_cde_from_prompts(clip, prompts), prompt_parts, prompts


def encode_clip_images(clip, rows, source):
    return nola_eval.extract_features(nola_eval.clip_scores, clip, rows, source)


def score_clip(features, cde, scale=1.0):
    return nola_eval.class_scores(features, cde, scale)


def select_anchors(train_logits, cap):
    return nola_eval.select_anchors(train_logits, cap)


def extract_dino_features(rows, source):
    model = nola_eval.vit_base(patch_size=16, num_classes=0)
    weights = ROOT / 'weights/dino_vitbase16_pretrain.pth'
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.cuda().eval().requires_grad_(False)
    return nola_eval.extract_features(nola_eval.dino_features, model, rows, source)


def train_dino_head(features, anchors, classes):
    head = nola_eval.train_alignment_stage(features, anchors, classes, HEAD_EPOCHS, SEED)
    return nola_eval.alignment_predictions(head, features)


def fine_tune_clip(clip, cde, rows, source, pseudo_labels, truth, folder):
    folder.mkdir(parents=True, exist_ok=True)
    return nola_eval.train_visual_prompt_stage(
        clip, cde, rows, source, pseudo_labels, truth, folder,
        PROMPT_EPOCHS, BATCH, MICRO, SEED,
    )


def evaluate_full_run(folder, arm, rows, source, truth, names, logits, aligned,
                      tuned, anchors, epoch, curve):
    nola_eval.export_baseline(
        folder, dict(**key, arm=arm), rows, source, truth, names, logits, aligned,
        tuned, *anchors, epoch, curve, HEAD_EPOCHS, MICRO,
    )
