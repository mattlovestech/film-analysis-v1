"""A scouting report read from the data stored on each play."""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass

from fpdf import FPDF

from film_analysis.chart.formation import formation_label
from film_analysis.chart.formation_features import (
    NEEDS_LOOK,
    FormationFeatures,
    features_from_dict,
    lineup_label,
)
from film_analysis.storage.catalog import FilmTeams, PlayRecord

_CALLS = {
    "run": "Run",
    "pass": "Pass",
    "kneel": "Kneel",
    "spike": "Spike",
    "two_point": "Two-point",
    "kickoff": "Kickoff",
    "kick_return": "Kick return",
    "punt": "Punt",
    "punt_return": "Punt return",
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
    formation: str = ""


REPORT_SECTIONS = frozenset({"calls", "special", "tendencies", "plays", "formations"})
_DEFAULT_SECTIONS = frozenset({"calls", "special", "tendencies", "plays", "formations"})


def report_sections(names: Sequence[str] | None) -> frozenset[str]:
    """Every section when the caller did not choose. Unknown names are dropped."""

    if names is None:
        return _DEFAULT_SECTIONS
    chosen = frozenset(name for name in names if name in REPORT_SECTIONS)
    if not chosen:
        raise ValueError("Choose what the report should include.")
    return chosen


def chart_kinds(sections: frozenset[str]) -> frozenset[str]:
    """The play kinds that have to be read before these sections can be written."""

    kinds: set[str] = set()
    if sections & {"calls", "tendencies", "plays", "formations"}:
        kinds.add("scrimmage")
    if sections & {"special", "plays"}:
        kinds.add("special_teams")
    return frozenset(kinds)


def live_snapshot(
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
    sections: Sequence[str] | None = None,
) -> dict[str, object]:
    """The rows and run-pass counts available so far, while plays are still being read."""

    chosen = report_sections(sections)
    label, snaps = _opponent_offense(play_reads(plays), teams)
    runs = sum(1 for play in snaps if play.call_key == "run")
    passes = sum(1 for play in snaps if play.call_key == "pass")
    rows: list[dict[str, object]] = []
    for play in plays:
        if play.discarded or _kind(play) not in chart_kinds(chosen):
            continue
        read = _read_play(play)
        rows.append(
            {
                "play": read.number,
                "side": _shown_side(read.side, teams),
                "call": read.call if read.call_key else "",
                "formation": read.formation,
                "reading": _reading(play, chosen),
            }
        )
    return {
        "live": True,
        "label": f"{label} on offense.",
        "counts": {"run": runs, "pass": passes, "offense": len(snaps)},
        "rows": rows,
        "tendency": "tendencies" in chosen,
        "formation": "formations" in chosen,
    }


def play_reads(plays: Sequence[PlayRecord]) -> tuple[PlayRead, ...]:
    """Read the stored chart on each play. The report is built only from these."""

    return tuple(_read_play(play) for play in plays if not play.discarded)


def untagged_plays(plays: Sequence[PlayRecord]) -> int:
    """How many plays still need an offense or defense tag.

    Special-teams plays are already typed, so they do not wait on that tag.
    """

    return sum(1 for play in plays if not play.discarded and _needs_unit(play))


def scouting_blocks(
    film_name: str,
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
    sections: Sequence[str] | None = None,
    layouts: dict[int, dict[str, object]] | None = None,
) -> list[tuple[str, str]]:
    """The report as titled sections and lines, before it becomes a PDF."""

    chosen = report_sections(sections)
    reads = play_reads(plays)
    yours, opponent, owned = _names(teams)
    offense = [play for play in reads if play.side == "offense"]
    defense = [play for play in reads if play.side == "defense"]
    special = [play for play in reads if play.side == "special teams"]
    intro = f"{yours} against {opponent}." if owned else _both_teams(teams)
    blocks: list[tuple[str, str]] = [
        ("title", f"Scouting report: {film_name}"),
        ("body", intro),
    ]
    if "tendencies" in chosen:
        label, snaps = _opponent_offense(reads, teams)
        blocks.append(("heading", "Play tendency"))
        blocks.append(("body", f"{label} on offense."))
        blocks.append(("table", _tendency_table(snaps)))
        blocks.extend(_call_play_index(snaps))
    visible = _reads_for(reads, chosen)
    if visible and chosen & {"calls", "special", "plays"}:
        blocks.append(("body", _chart_summary(visible, teams)))
    if "plays" in chosen:
        blocks.append(("heading", "Each play"))
        blocks.extend(_play_lines(reads, teams))
    if "calls" in chosen and (offense or defense):
        if owned:
            blocks.append(("heading", f"{opponent} offense"))
            blocks.append(("body", _side_copy(defense, opponent, "offense")))
            blocks.extend(_call_play_index(defense))
            blocks.extend(_play_lines(defense, teams))
            blocks.append(("heading", f"{opponent} defense"))
            blocks.append(("body", _side_copy(offense, opponent, "defense", against_you=True)))
            blocks.extend(_call_play_index(offense))
            blocks.extend(_play_lines(offense, teams))
        else:
            blocks.append(("heading", "Offense"))
            blocks.append(("body", _side_copy(offense, "Offense", "offense")))
            blocks.extend(_call_play_index(offense))
            blocks.extend(_play_lines(offense, teams))
            blocks.append(("heading", "Defense"))
            blocks.append(("body", _side_copy(defense, "Defense", "defense")))
            blocks.extend(_call_play_index(defense))
            blocks.extend(_play_lines(defense, teams))
    if "special" in chosen:
        blocks.append(("heading", "Special teams"))
        blocks.append(("body", _special_copy(special)))
        if special:
            blocks.extend(_call_play_index(special))
            blocks.extend(_play_lines(special, teams))
    if "formations" in chosen:
        for breakdown in team_breakdowns(teams, plays, layouts):
            blocks.extend(_breakdown_blocks(breakdown))
    return blocks


def scouting_pdf(
    film_name: str,
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
    sections: Sequence[str] | None = None,
    layouts: dict[int, dict[str, object]] | None = None,
) -> bytes:
    """Render the scouting report as a PDF."""

    pdf = FPDF(format="letter")
    pdf.set_compression(False)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    for kind, text in scouting_blocks(film_name, teams, plays, sections, layouts):
        safe = text.encode("latin-1", "replace").decode("latin-1")
        if kind == "title":
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 9, safe, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif kind == "heading":
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, safe, new_x="LMARGIN", new_y="NEXT")
        elif kind == "table":
            _pdf_table(pdf, text)
        elif kind == "grouped":
            _pdf_table(pdf, text.split("\n", 1)[1] if "\n" in text else "")
        elif kind == "clips":
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, _clip_line(safe), new_x="LMARGIN", new_y="NEXT")
        elif kind == "plot":
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, f"Lineup plot: Play {safe}", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, safe, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def scouting_csv(
    film_name: str,
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
    sections: Sequence[str] | None = None,
    layouts: dict[int, dict[str, object]] | None = None,
) -> str:
    """One row per play, for a spreadsheet."""

    yours, opponent, _owned = _names(teams)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    chosen = report_sections(sections)
    show_formation = "formations" in chosen
    header = [
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
    if show_formation:
        header.append("formation")
    writer.writerow(header)
    for play in _reads_for(play_reads(plays), chosen):
        row = [
            film_name,
            yours,
            opponent,
            play.number,
            _csv_number(play.start_seconds),
            _csv_number(play.end_seconds),
            _csv_number(play.seconds),
            play.call,
            play.family,
            _shown_side(play.side, teams),
        ]
        if show_formation:
            row.append("" if play.side == "special teams" else _csv_formation(play, layouts))
        writer.writerow(row)
    return buffer.getvalue()


def _reads_for(reads: Sequence[PlayRead], sections: frozenset[str]) -> list[PlayRead]:
    """Plays that belong in the sections the user asked for."""

    kept: list[PlayRead] = []
    specials, calls = _covered(sections)
    for play in reads:
        special = play.side == "special teams"
        if (special and specials) or (not special and calls):
            kept.append(play)
    return kept


def _covered(sections: frozenset[str]) -> tuple[bool, bool]:
    """Whether special teams, then scrimmage, belong in this report."""

    specials = "special" in sections
    calls = "calls" in sections or "formations" in sections
    if not specials and not calls and sections & {"tendencies", "plays"}:
        return True, True
    return specials, calls


def _opponent_offense(
    reads: Sequence[PlayRead],
    teams: FilmTeams | None,
) -> tuple[str, list[PlayRead]]:
    """The other team's offensive snaps. Those are the plays this side defended."""

    _yours, opponent, owned = _names(teams)
    if owned:
        return opponent, [play for play in reads if play.side == "defense"]
    return "Offense", [play for play in reads if play.side == "offense"]


def _tendency_table(snaps: Sequence[PlayRead]) -> str:
    """Runs, passes, and the offensive play count, as tab-separated rows."""

    runs = sum(1 for play in snaps if play.call_key == "run")
    passes = sum(1 for play in snaps if play.call_key == "pass")
    charted = runs + passes
    rows = [
        "Call\tPlays\tShare",
        f"Run\t{runs}\t{_share(runs, charted)}",
        f"Pass\t{passes}\t{_share(passes, charted)}",
        f"Offense\t{len(snaps)}\t",
    ]
    return "\n".join(rows)


def _share(count: int, total: int) -> str:
    if total == 0:
        return "0%"
    return _percent(count, total)


def _pdf_table(pdf: FPDF, text: str) -> None:
    rows = [line.split("\t") for line in text.splitlines() if line]
    if not rows:
        return
    columns = max(len(row) for row in rows)
    first = 54 if columns > 3 else 42
    rest = min(42.0, (pdf.epw - first) / max(1, columns - 1))
    for index, row in enumerate(rows):
        pdf.set_font("Helvetica", "B" if index == 0 else "", 11)
        for position, cell in enumerate(row):
            safe = cell.encode("latin-1", "replace").decode("latin-1")
            pdf.cell(first if position == 0 else rest, 7, safe, border=1)
        pdf.ln(7)


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
        formation=formation_label(play.formation),
    )


def _needs_unit(play: PlayRecord) -> bool:
    return not _special(play) and play.unit not in {"offense", "defense"}


def _special(play: PlayRecord) -> bool:
    return play.family == "special_teams"


def _kind(play: PlayRecord) -> str:
    return "special_teams" if _special(play) else "scrimmage"


def _side_label(side: str) -> str:
    return {"offense": "Offense", "defense": "Defense", "special teams": "Special teams"}.get(
        side, ""
    )


def _reading(play: PlayRecord, chosen: frozenset[str]) -> list[str]:
    """Fields this play still needs before its row is finished."""

    fields: list[str] = []
    if _kind(play) == "scrimmage":
        if play.unit not in {"offense", "defense"}:
            fields.append("side")
        if play.play_type not in {"run", "pass"}:
            fields.append("call")
        if "formations" in chosen and not play.formation:
            fields.append("formation")
        return fields
    specials = {
        "kickoff",
        "kick_return",
        "punt",
        "punt_return",
        "field_goal",
        "extra_point",
    }
    if play.play_type not in specials:
        fields.append("call")
    return fields


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
    """Run and pass rates for the opponent's offense and the opponent's defense."""

    reads = play_reads(plays)
    yours, opponent, owned = _names(teams)
    lines: list[str] = []
    groups = (
        (("defense", "offense"), ("offense", "defense"))
        if owned
        else (
            ("offense", "offense"),
            ("defense", "defense"),
        )
    )
    for stored, spoken in groups:
        side = [play for play in reads if play.side == stored]
        runs = sum(1 for play in side if play.call_key == "run")
        passes = sum(1 for play in side if play.call_key == "pass")
        charted = runs + passes
        if charted == 0:
            continue
        team = opponent if owned else spoken.capitalize()
        mix = (
            f"{_calls(runs, 'run', 'runs')} ({_percent(runs, charted)}) and "
            f"{_calls(passes, 'pass', 'passes')} ({_percent(passes, charted)})"
        )
        lines.append(f"{team} {spoken}: {mix}.")
    special = [play for play in reads if play.side == "special teams" and play.call_key]
    if special:
        team = yours if owned else "Special teams"
        counts: dict[str, int] = {}
        for play in special:
            counts[play.call] = counts.get(play.call, 0) + 1
        pieces = [f"{count} {name.lower()}" for name, count in counts.items()]
        lines.append(f"{team} special teams: {', '.join(pieces)}.")
    return lines


def _chart_summary(plays: Sequence[PlayRead], teams: FilmTeams | None) -> str:
    if not plays:
        return "Read from no plays."
    yours, opponent, owned = _names(teams)
    counts: dict[str, int] = {}
    for play in plays:
        label = _ball_label(play.side, yours, opponent, owned)
        counts[label] = counts.get(label, 0) + 1
    pieces = [f"{count} {name}" for name, count in counts.items()]
    noun = "play" if len(plays) == 1 else "plays"
    calls: dict[str, int] = {}
    for play in plays:
        if play.call_key and play.call:
            calls[play.call.lower()] = calls.get(play.call.lower(), 0) + 1
    call_text = ", ".join(f"{count} {name}" for name, count in calls.items())
    summary = f"Read from {len(plays)} {noun}: {', '.join(pieces)}."
    if call_text:
        summary = f"{summary} {call_text}."
    return summary


def _ball_label(side: str, yours: str, opponent: str, owned: bool) -> str:
    """Whose offense or defense this snap is, using the labeled teams."""

    del yours
    if side == "special teams":
        return "special teams"
    if owned and side == "defense":
        return f"{opponent} offense"
    if owned and side == "offense":
        return f"{opponent} defense"
    if side == "offense":
        return "offense"
    if side == "defense":
        return "defense"
    return "not read"


def _shown_side(side: str, teams: FilmTeams | None) -> str:
    """The side to show, named for the opponent when that team is known."""

    yours, opponent, owned = _names(teams)
    label = _ball_label(side, yours, opponent, owned)
    if label == "not read":
        return ""
    if label == "special teams":
        return "Special teams"
    return label[0].upper() + label[1:] if label else ""


def _side_copy(
    plays: Sequence[PlayRead], team: str, unit: str, *, against_you: bool = False
) -> str:
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
        detail = (
            "The call on each snap is what your offense ran against them."
            if against_you
            else "The call on each snap is what the offense ran."
        )
        lead = f"{team} defended {snaps} {snap_word}. {detail}"
    if charted == 0:
        mix = " None of these snaps are charted as a run or a pass."
    else:
        mix = (
            f" {_calls(runs, 'run', 'runs')} ({_percent(runs, charted)}) and "
            f"{_calls(passes, 'pass', 'passes')} ({_percent(passes, charted)}) "
            f"among the {charted} {_noun(charted, 'snap', 'snaps')} charted as a run or a pass."
        )
    order = ", ".join(play.call for play in plays if play.call)
    sequence = f" In order: {order}." if order else ""
    return f"{lead}{mix} The snaps average {_seconds(average)}.{sequence}"


def _special_copy(plays: Sequence[PlayRead]) -> str:
    if not plays:
        return "No special-teams plays are tagged."
    counts: dict[str, int] = {}
    for play in plays:
        counts[play.call] = counts.get(play.call, 0) + 1
    pieces = [f"{count} {name.lower()}" for name, count in counts.items()]
    noun = "play" if len(plays) == 1 else "plays"
    return f"{len(plays)} special-teams {noun}: {', '.join(pieces)}."


def _call_play_index(plays: Sequence[PlayRead]) -> list[tuple[str, str]]:
    """Which play numbers were tagged as each run, pass, or special-teams call."""

    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for play in plays:
        if not play.call:
            continue
        if play.call not in groups:
            order.append(play.call)
            groups[play.call] = []
        groups[play.call].append(play.number)
    return [
        ("clips", "\t".join([name, *(str(number) for number in groups[name])])) for name in order
    ]


def _clip_line(text: str) -> str:
    """A clip list as one readable line for the PDF."""

    parts = [part for part in text.split("\t") if part]
    if not parts:
        return ""
    plays = ", ".join(f"Play {number}" for number in parts[1:])
    return f"{parts[0]}: {plays}" if plays else parts[0]


def _play_lines(plays: Sequence[PlayRead], teams: FilmTeams | None = None) -> list[tuple[str, str]]:
    return [("row", _play_line(play, teams)) for play in plays]


def _play_line(play: PlayRead, teams: FilmTeams | None = None) -> str:
    shown = _shown_side(play.side, teams) or "Not read"
    call = f"  {play.call}" if play.call_key and play.call else ""
    formed = f"  {play.formation}" if play.side == "offense" and play.formation else ""
    return (
        f"Play {play.number}  {_clock(play.start_seconds)}-{_clock(play.end_seconds)}"
        f"  {shown}{call}  {_seconds(play.seconds)}{formed}"
    )


@dataclass(frozen=True)
class FormationGroup:
    """Snaps where the scouted team lined up the same way."""

    formation: str
    plays: tuple[int, ...]
    runs: int
    passes: int
    plot: int = 0


@dataclass(frozen=True)
class UnitBreakdown:
    """The scouted team on one side of the ball, grouped by their own lineup."""

    team: str
    unit: str
    opponent: str
    snaps: int
    runs: int
    passes: int
    groups: tuple[FormationGroup, ...]

    @property
    def read(self) -> int:
        return sum(len(group.plays) for group in self.groups if group.formation != NEEDS_LOOK)

    @property
    def scope(self) -> str:
        return f"{self.team} {self.unit}"


def team_breakdowns(
    teams: FilmTeams | None,
    plays: Sequence[PlayRecord],
    layouts: dict[int, dict[str, object]] | None = None,
) -> list[UnitBreakdown]:
    """The scouted team's offense and defense, grouped by how they lined up.

    A stored offense tag means that team had the ball. On defense the group is
    their own alignment, not the formation the other offense came out in.
    """

    reads = play_reads(plays)
    yours, opponent, _owned = _names(teams)
    offense = [play for play in reads if play.side == "offense"]
    defense = [play for play in reads if play.side == "defense"]
    return [
        _unit_breakdown(yours, "offense", opponent, offense, layouts),
        _unit_breakdown(yours, "defense", opponent, defense, layouts),
    ]


def _unit_breakdown(
    team: str,
    unit: str,
    opponent: str,
    snaps: Sequence[PlayRead],
    layouts: dict[int, dict[str, object]] | None = None,
) -> UnitBreakdown:
    grouped: dict[str, list[PlayRead]] = {}
    for play in snaps:
        grouped.setdefault(_lineup_name(play, unit, layouts), []).append(play)
    ranked = sorted(
        grouped.items(),
        key=lambda item: (item[0] == NEEDS_LOOK, -len(item[1]), item[1][0].number),
    )
    groups = tuple(
        FormationGroup(
            name,
            tuple(play.number for play in members),
            sum(1 for play in members if play.call_key == "run"),
            sum(1 for play in members if play.call_key == "pass"),
            _plot_play(members, layouts),
        )
        for name, members in ranked
    )
    return UnitBreakdown(
        team,
        unit,
        opponent,
        len(snaps),
        sum(1 for play in snaps if play.call_key == "run"),
        sum(1 for play in snaps if play.call_key == "pass"),
        groups,
    )


def _breakdown_blocks(breakdown: UnitBreakdown) -> list[tuple[str, str]]:
    """Heading, summary, formation table, and one play grid per formation."""

    blocks: list[tuple[str, str]] = [("heading", f"{breakdown.team} {breakdown.unit}")]
    blocks.append(("body", _breakdown_copy(breakdown)))
    if not breakdown.snaps:
        return blocks
    rows = ["Formation\tPlays\tShare\tRun\tPass"]
    for group in breakdown.groups:
        share = "" if group.formation == NEEDS_LOOK else _share(len(group.plays), breakdown.read)
        rows.append(f"{group.formation}\t{len(group.plays)}\t{share}\t{group.runs}\t{group.passes}")
    blocks.append(("grouped", "\n".join([breakdown.scope, *rows])))
    for group in breakdown.groups:
        if group.plot:
            blocks.append(("plot", str(group.plot)))
        name = f"{breakdown.scope} · {group.formation}"
        blocks.append(("clips", "\t".join([name, *(str(number) for number in group.plays)])))
    return blocks


def _breakdown_copy(breakdown: UnitBreakdown) -> str:
    team, opponent, snaps = breakdown.team, breakdown.opponent, breakdown.snaps
    if not snaps:
        side = "offensive" if breakdown.unit == "offense" else "defensive"
        return f"{team} has no {side} snaps charted in this film."
    snap_word = _noun(snaps, "snap", "snaps")
    charted = breakdown.runs + breakdown.passes
    runs = f"{_calls(breakdown.runs, 'run', 'runs')} ({_percent(breakdown.runs, charted or 1)})"
    passes = (
        f"{_calls(breakdown.passes, 'pass', 'passes')} ({_percent(breakdown.passes, charted or 1)})"
    )
    if breakdown.unit == "offense":
        lead = f"{team} had the ball on {snaps} {snap_word}"
        mix = f": {runs} and {passes}."
    else:
        lead = f"{team} defended {snaps} {snap_word}"
        mix = f" and faced {runs} and {passes} from {opponent}."
    if not charted:
        mix = ". None are charted as a run or a pass."
    read = breakdown.read
    if not read:
        formed = " No lineup was plotted on these snaps yet."
        return f"{lead}{mix}{formed}"
    named = [group for group in breakdown.groups if group.formation != NEEDS_LOOK]
    top = named[0]
    most = (
        f"Most used: {top.formation}, {len(top.plays)} of {read} ({_percent(len(top.plays), read)})"
    )
    lean = _formation_lean(top, breakdown.unit)
    coverage = f" Lineup read on {read} of {snaps} {snap_word}."
    if read < snaps:
        coverage += f" {snaps - read} still need a look."
    return f"{lead}{mix} {most}{lean}.{coverage}"


def _formation_lean(group: FormationGroup, unit: str) -> str:
    """Say when a lineup gives the call away."""

    charted = group.runs + group.passes
    if charted < 3:
        return ""
    verb = "the offense ran" if unit == "defense" else "run"
    passed = "the offense passed" if unit == "defense" else "pass"
    if group.runs / charted >= 0.7:
        return f", {verb} {_percent(group.runs, charted)} of the time"
    if group.passes / charted >= 0.7:
        return f", {passed} {_percent(group.passes, charted)} of the time"
    return ""


def _lineup_name(
    play: PlayRead, unit: str, layouts: dict[int, dict[str, object]] | None
) -> str:
    fallback = play.formation if unit == "offense" else ""
    return lineup_label(_layout_features(play.number, layouts), unit, fallback)


def _layout_features(
    play_number: int, layouts: dict[int, dict[str, object]] | None
) -> FormationFeatures | None:
    if not layouts:
        return None
    raw = layouts.get(play_number)
    if not isinstance(raw, dict):
        return None
    payload = raw.get("features") if isinstance(raw.get("features"), dict) else raw
    if not isinstance(payload, dict):
        return None
    return features_from_dict(payload)


def _plot_play(members: Sequence[PlayRead], layouts: dict[int, dict[str, object]] | None) -> int:
    for play in members:
        if _layout_features(play.number, layouts) is not None:
            return play.number
    return 0


def _csv_formation(play: PlayRead, layouts: dict[int, dict[str, object]] | None) -> str:
    """How the scouted team lined up. Special teams and unread snaps stay blank."""

    if play.side not in {"offense", "defense"}:
        return ""
    label = _lineup_name(play, play.side, layouts)
    if label == NEEDS_LOOK:
        return ""
    return label


def _call(play: PlayRecord) -> str:
    if play.play_type in _CALLS:
        return _CALLS[play.play_type]
    if _special(play):
        return "Special teams"
    return ""


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
