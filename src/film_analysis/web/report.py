"""A scouting report read from the data stored on each play."""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class PlayRead:
    """The chart fields one play contributes to the report."""

    number: int
    start_seconds: float
    end_seconds: float
    seconds: float
    call: str
    call_key: str | None
    family: str
    side: str


def play_reads(plays: Sequence[PlayRecord]) -> tuple[PlayRead, ...]:
    """Read the stored chart on each play. The report is built only from these."""

    return tuple(_read_play(play) for play in plays)


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

    reads = play_reads(plays)
    yours, opponent, owned = _names(teams)
    offense = [play for play in reads if play.side == "offense"]
    defense = [play for play in reads if play.side == "defense"]
    special = [play for play in reads if play.side == "special teams"]
    intro = f"{yours} against {opponent}." if owned else _both_teams(teams)
    blocks: list[tuple[str, str]] = [
        ("title", f"Scouting report: {film_name}"),
        ("body", intro),
        ("body", _chart_summary(reads)),
    ]
    tendencies = tendency_lines(teams, plays)
    if tendencies:
        blocks.append(("heading", "Tendencies"))
        blocks.extend(("body", line) for line in tendencies)
    blocks.append(("heading", "Each play"))
    blocks.extend(_play_lines(reads))
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


def scouting_csv(
    film_name: str,
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
) -> str:
    """One row per play, for a spreadsheet."""

    yours, opponent, _owned = _names(teams)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "film",
            "your_team",
            "opponent",
            "play",
            "start_seconds",
            "end_seconds",
            "seconds",
            "call",
            "family",
            "side",
        ]
    )
    for play in play_reads(plays):
        writer.writerow(
            [
                film_name,
                yours,
                opponent,
                play.number,
                _csv_number(play.start_seconds),
                _csv_number(play.end_seconds),
                _csv_number(play.seconds),
                play.call,
                play.family,
                play.side,
            ]
        )
    return buffer.getvalue()


def _read_play(play: PlayRecord) -> PlayRead:
    return PlayRead(
        number=play.play_number,
        start_seconds=play.start_seconds,
        end_seconds=play.end_seconds,
        seconds=max(0.0, play.end_seconds - play.start_seconds),
        call=_call(play),
        call_key=play.play_type,
        family=play.family,
        side=_side(play),
    )


def _needs_unit(play: PlayRecord) -> bool:
    return not _special(play) and play.unit not in {"offense", "defense"}


def _special(play: PlayRecord) -> bool:
    return play.family == "special_teams"


def _side(play: PlayRecord) -> str:
    if _special(play):
        return "special teams"
    if play.unit in {"offense", "defense"}:
        return play.unit
    return ""


def _csv_number(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


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


def tendency_lines(
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
) -> list[str]:
    """Run and pass rates for the user's offense and for the plays they defended."""

    reads = play_reads(plays)
    yours, _opponent, owned = _names(teams)
    lines: list[str] = []
    for unit, label in (("offense", "offense"), ("defense", "defense")):
        side = [play for play in reads if play.side == unit]
        runs = sum(1 for play in side if play.call_key == "run")
        passes = sum(1 for play in side if play.call_key == "pass")
        charted = runs + passes
        if charted == 0:
            continue
        team = yours if owned else label.capitalize()
        mix = (
            f"{_calls(runs, 'run', 'runs')} ({_percent(runs, charted)}) and "
            f"{_calls(passes, 'pass', 'passes')} ({_percent(passes, charted)})"
        )
        if unit == "offense":
            lines.append(f"{team} offense: {mix}.")
        else:
            lines.append(f"{team} defense, from the plays the offense ran: {mix}.")
    return lines


def _chart_summary(plays: Sequence[PlayRead]) -> str:
    if not plays:
        return "Read from no plays."
    counts: dict[str, int] = {}
    for play in plays:
        counts[play.call] = counts.get(play.call, 0) + 1
    pieces = [f"{count} {name.lower()}" for name, count in counts.items()]
    noun = "play" if len(plays) == 1 else "plays"
    return f"Read from {len(plays)} {noun}: {', '.join(pieces)}."


def _side_copy(plays: Sequence[PlayRead], team: str, unit: str) -> str:
    if not plays:
        side = "offensive" if unit == "offense" else "defensive"
        return f"{team} has no {side} snaps in this film."
    snaps = len(plays)
    runs = sum(1 for play in plays if play.call_key == "run")
    passes = sum(1 for play in plays if play.call_key == "pass")
    charted = runs + passes
    average = sum(play.seconds for play in plays) / snaps
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
    order = ", ".join(play.call for play in plays)
    return f"{lead}{mix} The snaps average {_seconds(average)}. In order: {order}."


def _special_copy(plays: Sequence[PlayRead]) -> str:
    if not plays:
        return "No special-teams plays are tagged."
    counts: dict[str, int] = {}
    for play in plays:
        counts[play.call] = counts.get(play.call, 0) + 1
    pieces = [f"{count} {name.lower()}" for name, count in counts.items()]
    noun = "play" if len(plays) == 1 else "plays"
    return f"{len(plays)} special-teams {noun}: {', '.join(pieces)}."


def _play_lines(plays: Sequence[PlayRead]) -> list[tuple[str, str]]:
    return [("row", _play_line(play)) for play in plays]


def _play_line(play: PlayRead) -> str:
    side = f"  {play.side.capitalize()}" if play.side else ""
    return (
        f"Play {play.number}  {_clock(play.start_seconds)}-{_clock(play.end_seconds)}"
        f"  {play.call}  {_seconds(play.seconds)}{side}"
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
