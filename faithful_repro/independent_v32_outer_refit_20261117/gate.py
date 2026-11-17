def required_keys():
    return [f'fold{fold}_seed{seed}_{arm}' for fold in range(5) for seed in (1234,42,2025) for arm in ('dynamic','static')]

def check_keys(keys):
    expected=required_keys()
    if len(keys)!=len(expected) or len(set(keys))!=len(keys) or set(keys)!=set(expected):
        raise ValueError('expected exactly 30 distinct fold/seed/policy runs')
