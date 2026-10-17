from scripts.select_persona_adapter import choose


def row(name, degeneration, changed, style, accuracy=.5, loss=.7):
    return {"name": name, "degeneration_rate": degeneration,
            "changed_vs_base_rate": changed, "style_marker_rate": style,
            "held_out_reward_accuracy": accuracy, "held_out_eval_loss": loss}


def test_selects_highest_style_rate_among_eligible():
    rows = [row("a", .04, .5, .4), row("b", .05, .5, .7)]
    assert choose(rows, .02)[0] == "b"


def test_no_eligible_selects_lowest_degeneration_and_discloses_reason():
    rows = [row("a", .2, .1, .8), row("b", .1, .1, .2)]
    selected, reason = choose(rows, .0)
    assert selected == "b" and "no adapter met eligibility" in reason


def test_ties_use_accuracy_then_lower_loss():
    rows = [row("a", .01, .5, .5, .8, .6), row("b", .01, .5, .5, .9, .8)]
    assert choose(rows, .0)[0] == "b"
    rows = [row("a", .01, .5, .5, .9, .5), row("b", .01, .5, .5, .9, .6)]
    assert choose(rows, .0)[0] == "a"
