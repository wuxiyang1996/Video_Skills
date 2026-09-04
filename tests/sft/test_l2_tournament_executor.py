from dataset_clip_wrapper.training.evaluate_l2_tournament_executor import knockout_winner


def test_knockout_winner_handles_odd_bracket() -> None:
    calls = []

    def choose(left: int, right: int, salt: str) -> int:
        calls.append((left, right, salt))
        return max(left, right)

    assert knockout_winner([1, 4, 2, 5, 3], choose, "e") == 5
    assert len(calls) == 4
