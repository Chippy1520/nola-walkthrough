clip_train = encode_clip_images(clip, {'train': rows['train']}, source)
clip_features = {**clip_train, **clip_val}
release_cosine = score_clip(clip_features, release_cde)
release_logits = score_clip(clip_features, release_cde, clip_scale)
del clip
release_cap = max(16, int(.2 * len(rows['train']) / len(names)))
