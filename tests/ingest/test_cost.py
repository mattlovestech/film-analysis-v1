"""Model-read prices and how a read is shared across the plays it covered."""

from film_analysis.ingest.cost import cost_per_play, read_cost


def test_sonnet_list_price_uses_input_and_output_tokens() -> None:
    assert read_cost("claude-sonnet-5-5", 1_000, 100) == 0.003
    assert read_cost("claude-haiku-5-5", 1_000_000, 0) == 0.10


def test_an_unknown_model_is_not_priced() -> None:
    assert read_cost("some-other-model", 1_000, 100) == 0.0


def test_a_dated_model_id_uses_the_same_price() -> None:
    assert read_cost("claude-sonnet-5-5-20260615", 1_000, 100) == 0.003


def test_one_read_is_split_across_the_plays_it_covers() -> None:
    shares = cost_per_play(
        [(0.0, 50.0, 0.006)],
        [(1, 10.0, 20.0), (2, 30.0, 40.0)],
    )

    assert shares == [(1, 0.003), (2, 0.003)]
