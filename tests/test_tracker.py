from src.server.tracker import Tracker


def feed(tracker: Tracker, answers: str) -> list[bool]:
    """'ttff' -> updates; returns which ones fired."""
    return [tracker.update(c == "t") for c in answers]


def test_baseline_never_fires():
    assert feed(Tracker("rising"), "tt") == [False, False]
    assert feed(Tracker("falling"), "ff") == [False, False]


def test_rising_fires_once_on_confirmed_flip():
    t = Tracker("rising")
    assert feed(t, "ff" + "tt" + "tttt") == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert t.state is True


def test_falling_fires_on_true_to_false():
    assert feed(Tracker("falling"), "tt" + "ff") == [False, False, False, True]


def test_rising_ignores_true_to_false():
    assert feed(Tracker("rising"), "tt" + "ff") == [False, False, False, False]


def test_flicker_does_not_flip():
    t = Tracker("rising")
    assert feed(t, "ff" + "t" + "f" + "t" + "f") == [False] * 6
    assert t.state is False


def test_fires_again_after_going_back_down():
    fired = feed(Tracker("rising"), "ff" + "tt" + "ff" + "tt")
    assert fired == [False, False, False, True, False, False, False, True]


def test_persist_one_flips_immediately():
    assert feed(Tracker("rising", persist=1), "ft") == [False, True]
