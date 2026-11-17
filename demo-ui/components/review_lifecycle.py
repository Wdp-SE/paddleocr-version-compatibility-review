"""Action-first summary common to both change-review branches."""
def render_lifecycle(result):
    from components.review_overview import render_overview
    return render_overview(result)
