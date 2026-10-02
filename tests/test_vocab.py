from aftertrace.world import vocab


def test_vocab_is_unique_and_under_limit() -> None:
    assert len(vocab.TOKENS) == len(set(vocab.TOKENS))
    assert len(vocab.TOKENS) < 1000
    assert set(vocab.TOKEN_TO_ID) == set(vocab.TOKENS)
    assert all(vocab.ID_TO_TOKEN[index] == token for token, index in vocab.TOKEN_TO_ID.items())

