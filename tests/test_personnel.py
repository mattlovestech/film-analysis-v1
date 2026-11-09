import pytest

from film_analysis.domain.personnel import personnel_from_label


def test_eleven_personnel_is_one_back_one_tight_end_three_wide() -> None:
    personnel = personnel_from_label("11")

    assert personnel.running_backs == 1
    assert personnel.tight_ends == 1
    assert personnel.wide_receivers == 3


def test_twenty_two_personnel() -> None:
    personnel = personnel_from_label(" 22 ")

    assert personnel.label == "22"
    assert (personnel.running_backs, personnel.tight_ends, personnel.wide_receivers) == (2, 2, 1)


@pytest.mark.parametrize("label", ["1", "111", "ab", "33", ""])
def test_invalid_personnel_labels(label: str) -> None:
    with pytest.raises(ValueError):
        personnel_from_label(label)
