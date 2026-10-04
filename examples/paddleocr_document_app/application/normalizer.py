from result_adapter import normalize_v2, normalize_v3


def normalize_legacy(raw):
    return normalize_v2(raw)


def normalize_target(raw):
    return normalize_v3(raw)
