def hit_rate(hit, total):
    """命中缓存的 token 占输入 token 的比例。比如 hit=1152, total=1349 → 0.854"""
    if total > 0:
        return round(hit*100 / total , 2)
    return None

