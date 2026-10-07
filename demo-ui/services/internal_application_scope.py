"""Dependency versions belong to the POC application, not its own release ID."""
CURRENT_DEPENDENCY = 'v2.9.1'
TARGET_DEPENDENCY = 'v3.0.0'


def dependency_scope(versions, mode='current', selected=None):
    if mode not in ('current', 'target', 'manual'):
        raise ValueError('未知查询范围')
    version = CURRENT_DEPENDENCY if mode == 'current' else TARGET_DEPENDENCY if mode == 'target' else selected
    if version not in versions and not (mode == 'manual' and version == 'all' and versions):
        raise ValueError('应用所需依赖版本未收录，请补齐资料或明确选择已有版本；不会静默切换版本。')
    return version
