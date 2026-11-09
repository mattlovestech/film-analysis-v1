"""List price of a model read, from the tokens the API reported."""

from collections.abc import Sequence

# Dollars per million tokens, input then output, from the published list price.
_RATES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
}


def read_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollars for one read. An unknown model costs nothing we can price."""

    rates = _rates(model)
    if rates is None or input_tokens < 0 or output_tokens < 0:
        return 0.0
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


def _rates(model: str) -> tuple[float, float] | None:
    name = model.strip().lower()
    if name in _RATES:
        return _RATES[name]
    match: tuple[str, tuple[float, float]] | None = None
    for key, rates in _RATES.items():
        if name.startswith(key) and (match is None or len(key) > len(match[0])):
            match = (key, rates)
    return None if match is None else match[1]


def summarize_reads(
    rows: Sequence[tuple[int, float, int, int, str]],
) -> dict[int, tuple[float, int, int, str]]:
    """Total dollars, tokens, and model names for each film's reads."""

    grouped: dict[int, list[tuple[float, int, int, str]]] = {}
    for project_id, cost, incoming, outgoing, model in rows:
        grouped.setdefault(project_id, []).append((cost, incoming, outgoing, model))
    summary: dict[int, tuple[float, int, int, str]] = {}
    for project_id, notes in grouped.items():
        cost = sum(item[0] for item in notes)
        incoming = sum(item[1] for item in notes)
        outgoing = sum(item[2] for item in notes)
        models: list[str] = []
        for _cost, _incoming, _outgoing, model in notes:
            name = model.strip()
            if name and name not in models:
                models.append(name)
        if cost <= 0 and incoming <= 0 and outgoing <= 0 and not models:
            continue
        summary[project_id] = (cost, incoming, outgoing, ", ".join(models))
    return summary


def cost_per_play(
    reads: Sequence[tuple[float, float, float]],
    plays: Sequence[tuple[int, float, float]],
) -> list[tuple[int, float]]:
    """Share each read across the plays it covered, in play order.

    `reads` are `(start, end, dollars)`. `plays` are `(number, start, end)`.
    A read that covers no play is left out. Two plays in one read split it
    by how long each one overlaps that read.
    """

    totals = {number: 0.0 for number, _start, _end in plays}
    for start, end, dollars in reads:
        if dollars <= 0 or end <= start:
            continue
        covered = [
            (number, overlap)
            for number, play_start, play_end in plays
            if (overlap := min(end, play_end) - max(start, play_start)) > 0
        ]
        span = sum(overlap for _number, overlap in covered)
        if span <= 0:
            continue
        for number, overlap in covered:
            totals[number] += dollars * (overlap / span)
    return [(number, totals[number]) for number, _start, _end in plays if totals[number] > 0]
