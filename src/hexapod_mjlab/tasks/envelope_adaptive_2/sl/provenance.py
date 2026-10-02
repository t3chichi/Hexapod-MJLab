"""Record actual frozen collection and environment split alongside a model."""
import hashlib


def training_provenance(cfg, maps, files, dataset, seed):
    config = cfg.to_dict()
    collection = []
    for data, path in zip(maps, files, strict=True):
        collection.append({
            'seed': data.seed,
            'path': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'shape': list(data.obs.shape),
            'metadata': data.meta,
        })
    config['data']['seeds'] = [m.seed for m in maps]
    config['data']['pillar_counts'] = [m.meta.get('pillar_count') for m in maps]
    # A single global field can describe uniform collections only. Per-map
    # metadata remains authoritative when the files have different budgets.
    projected = []
    for field in ['num_envs', 'num_steps', 'lidar_decimation']:
        values = [m.meta.get(field) for m in maps]
        if values and values[0] is not None and all(v == values[0] for v in values):
            config['data'][field] = values[0]
            projected.append(field)
    split = {}
    for name, indices in [('train', dataset.train_idx), ('val', dataset.val_idx)]:
        split[name] = {
            'environment_ids': dataset.env_id[indices].unique().tolist(),
            'window_indices': indices.tolist(),
        }
    return {
        'teacher': dataset.teacher,
        'config': config,
        'seed': seed,
        'seeds': config['data']['seeds'],
        'data_files': {entry['path']: entry['sha256'] for entry in collection},
        'actual_collection': collection,
        'projected_collection_fields': projected,
        'dataset_summary': dataset.summary(),
        'split': split,
    }
