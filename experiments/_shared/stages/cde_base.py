rows, source, names, bank_name, truth = load_data()
out = out_dir
out.mkdir(parents=True, exist_ok=True)
split_json = json.dumps(rows)
split_id = hashlib.sha256(split_json.encode()).hexdigest()
key = dict(run_id=f'{RUN_NAME}/{TAG}', dataset=TAG, seed=SEED, split_id=split_id)
(out/'split.json').write_text(split_json)
(out/'manifest.json').write_text(json.dumps(dict(**key, head_epochs=HEAD_EPOCHS,
    prompt_epochs=PROMPT_EPOCHS, batch=BATCH, micro=MICRO, stage='development_validation',
    sources={name: nola_eval.sha(helpers/name) for name in ('nola_eval.py', 'experiments.py')}), indent=2))

clip = load_clip()
release_cde, prompt_parts, release_prompts = make_cde(clip, names, bank_name)
clip_scale = clip.logit_scale.detach().exp().item()
clip_val = encode_clip_images(clip, {'val': rows['val']}, source)
release_val = score_clip(clip_val, release_cde)
