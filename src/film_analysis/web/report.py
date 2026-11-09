"""A scouting report from the play chart, once a team is labeled."""

from collections.abc import Sequence

from fpdf import FPDF

from film_analysis.storage.catalog import FilmTeams, PlayRecord

_CALLS = {
    "run": "Run",
    "pass": "Pass",
    "kneel": "Kneel",
    "spike": "Spike",
    "two_point": "Two-point",
    "kickoff": "Kickoff",
    "punt": "Punt",
    "field_goal": "Field goal",
    "extra_point": "Extra point",
}


def untagged_plays(plays: Sequence[PlayRecord]) -> int:
    """How many plays still need an offense or defense tag.

    Special-teams plays are already typed, so they do not wait on that tag.
    """

    return sum(1 for play in plays if _needs_unit(play))


def scouting_blocks(
    film_name: str,
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
) -> list[tuple[str, str]]:
    """The report as titled sections and lines, before it becomes a PDF."""

    yours, opponent, owned = _names(teams)
    offense = [play for play in plays if play.unit == "offense" and not _special(play)]
    defense = [play for play in plays if play.unit == "defense" and not _special(play)]
    special = [play for play in plays if _special(play)]
    intro = f"{yours} against {opponent}." if owned else _both_teams(teams)
    blocks: list[tuple[str, str]] = [
        ("title", f"Scouting report: {film_name}"),
        ("body", intro),
        ("body", _chart_summary(plays)),
        ("heading", "Plays"),
    ]
    blocks.extend(_play_lines(plays))
    if offense or defense:
        offense_name = yours if owned else "Offense"
        defense_name = yours if owned else "Defense"
        blocks.append(("heading", f"{offense_name} offense" if owned else "Offense"))
        blocks.append(("body", _side_copy(offense, offense_name, "offense")))
        blocks.extend(_play_lines(offense))
        blocks.append(("heading", f"{defense_name} defense" if owned else "Defense"))
        blocks.append(("body", _side_copy(defense, defense_name, "defense")))
        blocks.extend(_play_lines(defense))
    blocks.append(("heading", "Special teams"))
    blocks.append(("body", _special_copy(special)))
    if special:
        blocks.extend(_play_lines(special))
    return blocks


def scouting_pdf(
    film_name: str,
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
) -> bytes:
    """Render the scouting report as a PDF."""

    pdf = FPDF(format="letter")
    pdf.set_compression(False)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    for kind, text in scouting_blocks(film_name, teams, plays):
        safe = text.encode("latin-1", "replace").decode("latin-1")
        if kind == "title":
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 9, safe, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif kind == "heading":
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, safe, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, safe, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def _needs_unit(play: PlayRecord) -> bool:
    return not _special(play) and play.unit not in {"offense", "defense"}


def _special(play: PlayRecord) -> bool:
    return play.family == "special_teams"


def _names(teams: FilmTeams | None) -> tuple[str, str, bool]:
    if teams is None:
        return "Your team", "the opponent", False
    if teams.mine == "right":
        return teams.right_name, teams.left_name, True
    if teams.mine == "left":
        return teams.left_name, teams.right_name, True
    return teams.left_name, teams.right_name, False


def _both_teams(teams: FilmTeams | None) -> str:
    if teams is None:
        return "Both teams."
    return f"{teams.left_name} and {teams.right_name}."


def _chart_summary(plays: Sequence[PlayRecord]) -> str:
    if not plays:
        return "No plays are in this film."
    counts: dict[str, int] = {}
    for play in plays:
        name = _call(play)
        counts[name] = counts.get(name, 0) + 1
    pieces = [f"{count} {name.lower()}" for name, count in counts.items()]
    noun = "play" if len(plays) == 1 else "plays"
    return f"{len(plays)} {noun}: {', '.join(pieces)}."


def _side_copy(plays: Sequence[PlayRecord], team: str, unit: str) -> str:
    if not plays:
        side = "offensive" if unit == "offense" else "defensive"
        return f"{team} has no {side} snaps in this film."
    snaps = len(plays)
    runs = sum(1 for play in plays if play.play_type == "run")
    passes = sum(1 for play in plays if play.play_type == "pass")
    charted = runs + passes
    average = sum(max(0.0, play.end_seconds - play.start_seconds) for play in plays) / snaps
    snap_word = "snap" if snaps == 1 else "snaps"
    if unit == "offense":
        lead = f"{team} ran {snaps} offensive {snap_word}."
    else:
        lead = (
            f"{team} defended {snaps} {snap_word}. "
            "The call on each snap is what the offense ran."
        )
    if charted == 0:
        mix = " None of these snaps are charted as a run or a pass."
    else:
        mix = (
            f" {_calls(runs, 'run', 'runs')} ({_percent(runs, charted)}) and "
            f"{_calls(passes, 'pass', 'passes')} ({_percent(passes, charted)}) "
            f"among the {charted} {_noun(charted, 'snap', 'snaps')} charted as a run or a pass."
        )
    order = ", ".join(_call(play) for play in plays)
    return f"{lead}{mix} The snaps average {_seconds(average)}. In order: {order}."


def _special_copy(plays: Sequence[PlayRecord]) -> str:
    if not plays:
        return "No special-teams plays are tagged."
    counts: dict[str, int] = {}
    for play in plays:
        name = _call(play)
        counts[name] = counts.get(name, 0) + 1
    pieces = [f"{count} {name.lower()}" for name, count in counts.items()]
    noun = "play" if len(plays) == 1 else "plays"
    return f"{len(plays)} special-teams {noun}: {', '.join(pieces)}."


def _play_lines(plays: Sequence[PlayRecord]) -> list[tuple[str, str]]:
    return [("row", _play_line(play)) for play in plays]


def _play_line(play: PlayRecord) -> str:
    span = max(0.0, play.end_seconds - play.start_seconds)
    return (
        f"Play {play.play_number}  {_clock(play.start_seconds)}-{_clock(play.end_seconds)}"
        f"  {_call(play)}  {_seconds(span)}"
    )


def _call(play: PlayRecord) -> str:
    if play.play_type in _CALLS:
        return _CALLS[play.play_type]
    if _special(play):
        return "Special teams"
    return "Scrimmage"


def _calls(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _noun(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _percent(count: int, total: int) -> str:
    return f"{round(100 * count / total)}%"


def _seconds(value: float) -> str:
    whole = max(0, int(round(value)))
    return f"{whole}s"


def _clock(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"
