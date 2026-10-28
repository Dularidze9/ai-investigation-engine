import pytest

from app.ingestion.chunking import chunk_a, chunk_b, split_sentences


def test_a_one_chunk_per_paragraph_with_title():
    chunks = chunk_a("Ed Wood", "First paragraph.\n\nSecond paragraph.")
    assert chunks == ["Ed Wood\n\nFirst paragraph.", "Ed Wood\n\nSecond paragraph."]


def test_a_long_paragraph_splits_at_sentence_boundaries():
    sentence = "This sentence has exactly seven words."
    para = " ".join([sentence] * 10)  # 70 words
    chunks = chunk_a("T", para, max_words=30)
    assert len(chunks) > 1
    for c in chunks:
        body = c.split("\n\n", 1)[1]
        assert len(body.split()) <= 30
        assert body.endswith(".")  # never cut mid-sentence


def test_a_single_huge_sentence_is_hard_cut():
    chunks = chunk_a("T", " ".join(["word"] * 100), max_words=40)
    assert [len(c.split("\n\n", 1)[1].split()) for c in chunks] == [40, 40, 20]


def test_a_markdown_headings_attach_to_the_next_paragraph():
    text = "# Outage\n\n## Root cause\n\nThe model was slow.\n\nRolled back."
    assert chunk_a("Outage", text) == ["Outage\n\nRoot cause\nThe model was slow.", "Outage\n\nRolled back."]


def test_a_hash_without_space_is_not_a_heading():
    assert chunk_a("Song", "#9 Dream is a song.") == ["Song\n\n#9 Dream is a song."]


def test_a_empty_text_gives_no_chunks():
    assert chunk_a("T", "  \n\n  ") == []


def test_b_windows_and_overlap():
    words = [f"w{i}" for i in range(200)]
    chunks = chunk_b(" ".join(words), window=80, overlap=20)
    assert [len(c.split()) for c in chunks] == [80, 80, 80]  # starts 0, 60, 120; last window reaches the end
    assert chunks[-1].split()[-1] == "w199"
    # consecutive windows share exactly `overlap` words
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_b_short_text_single_chunk_and_no_title():
    assert chunk_b("just a few words") == ["just a few words"]


def test_b_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_b("a b c", window=5, overlap=5)


def test_split_sentences():
    assert split_sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]
