from src.server.tracker import Tracker


def feed(tracker: Tracker, answers: str) -> list[bool]:
    """'ttff' -> updates; returns which ones fired."""
    return [tracker.update(c == "t") for c in answers]


def test_first_answer_fires_if_it_already_matches():
    assert feed(Tracker("rising"), "t" + "t") == [True, False]
    assert feed(Tracker("falling"), "f" + "f") == [True, False]
    assert feed(Tracker("rising"), "f") == [False]


def test_rising_fires_once_on_first_flip():
    t = Tracker("rising")
    assert feed(t, "f" + "t" + "ttt") == [False, True, False, False, False]
    assert t.state is True


def test_falling_fires_on_true_to_false():
    assert feed(Tracker("falling"), "t" + "f") == [False, True]


def test_rising_ignores_true_to_false():
    assert feed(Tracker("rising"), "t" + "f") == [True, False]  # fires on t, not on f


def test_fires_again_after_going_back_down():
    assert feed(Tracker("rising"), "f" + "t" + "f" + "t") == [False, True, False, True]


def test_persist_two_needs_confirmation():
    t = Tracker("rising", persist=2)
    assert feed(t, "ff" + "t" + "f" + "t" + "f") == [False] * 6  # flicker ignored
    assert feed(t, "tt") == [False, True]
