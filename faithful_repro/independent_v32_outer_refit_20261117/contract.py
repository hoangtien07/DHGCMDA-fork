def make_cfg(parent, seed, arm):
    if seed not in (1234,42,2025) or arm not in ('dynamic','static'):
        raise ValueError('unsupported seed or policy')
    cfg=dict(parent)
    if cfg.get('max_epochs')!=650 or cfg.get('dynamic_enabled') is not True:
        raise ValueError('unexpected parent horizon or policy')
    cfg.update(stage='U1_REFIT_NOVAL',refit_noval=True,seed=seed,inter_seed=seed,dynamic_enabled=(arm=='dynamic'))
    return cfg

def validate_refit_keys(data):
    required={'views','train_mask','train_pairs','y_train'}
    if not required.issubset(data):raise ValueError('missing train fields')
    for key in data:
        if key.startswith(('val_','test_','y_val','y_test')) or 'outer' in key:
            raise ValueError('held-out field is forbidden: '+key)
