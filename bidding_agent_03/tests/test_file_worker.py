from backend.workers.file_worker import split_chunks


def test_split_chunks_has_overlap_and_rejects_empty():
    assert split_chunks(" \n \n") == []
    chunks = split_chunks("甲" * 1500, size=1000, overlap=100)
    assert len(chunks) == 2
    assert chunks[0][-100:] == chunks[1][:100]
