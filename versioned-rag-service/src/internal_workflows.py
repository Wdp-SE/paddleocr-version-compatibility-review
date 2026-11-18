"""Supported maintenance workflows and their dependency axes, owned by the server."""

WORKFLOWS = {
    'upgrade_review': ('v2.9.1', 'v3.0.0'),
    'configuration_change': ('v2.9.1', 'v2.9.1'),
}


def workflow_versions(request):
    # Existing persisted upgrade requests predate the explicit discriminator.
    kind = request.get('workflow_type', 'upgrade_review')
    if kind not in WORKFLOWS:
        raise ValueError('Unsupported maintenance workflow')
    versions = WORKFLOWS[kind]
    if any(request.get(key, expected) != expected
           for key, expected in zip(('source_version', 'target_version'), versions)):
        raise ValueError('Workflow dependency versions do not match')
    return kind, *versions
