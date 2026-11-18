"""Only evidence gaps with remaining machine budget can be resumed automatically."""
def can_resume(investigation):
    return (isinstance(investigation,dict) and investigation.get('elapsed_seconds',240)<240
        and investigation.get('search_calls',24)<24
        and any(r.get('reason') in ('required_source_not_verified','content_requirements_not_covered')
                for r in investigation.get('unresolved',[])))
