dino_features = extract_dino_features(rows, source)

export_geometry(out, key, 'dino', dino_features['val'], truth['val'], rows['val'], names)
