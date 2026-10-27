import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from src.chunking.hierarchical_chunker import (
    ChildChunk,
    ParentChunk,
    build_child_chunks,
    build_parent_chunks,
    hierarchical_chunk_text,
)

ARABIC_TEXT = """النحو العربي علم يدرس أحكام الكلمة من حيث الإعراب والبناء.
الإعراب هو تغير أواخر الكلم بتغير العوامل الداخلة عليها.
البناء هو ثبات أواخر الكلم على حال واحدة.
العوامل في الإعراب: الرفع، والنصب، والجر، والجزم.
المرفوعات: المبتدأ، والخبر، والفاعل، ونائبه.
المنصوبات: المفعول به، والمفعول المطلق، والمفعول فيه، والمفعول له، والمفعول معه، والحال، والتمييز، والمستثنى، والمنادى.
المجرورات: ما بعد حروف الجر، وما بعد الإضافة.
المجزومات: الفعل المضارع بعد أدوات الجزم."""


def test_hierarchical_chunking_produces_parents_and_children():
    parents, children = hierarchical_chunk_text(
        ARABIC_TEXT,
        max_parent_chars=500,
        max_child_chars=200,
        parent_overlap_sentences=1,
        child_overlap_sentences=0,
    )
    
    assert len(parents) >= 1
    assert len(children) >= 1
    assert all(isinstance(p, ParentChunk) for p in parents)
    assert all(isinstance(c, ChildChunk) for c in children)
    
    # Each parent should have child_ids populated
    for parent in parents:
        assert parent.child_ids
        assert len(parent.child_ids) >= 1
    
    # Each child should reference a valid parent
    parent_ids = {p.parent_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids


def test_parent_child_ids_are_linked():
    """Children should have parent_id matching their parent's parent_id."""
    parents, children = hierarchical_chunk_text(
        ARABIC_TEXT,
        max_parent_chars=500,
        max_child_chars=200,
    )
    
    parent_ids = {p.parent_id for p in parents}
    child_parent_ids = {c.parent_id for c in children}
    
    assert child_parent_ids.issubset(parent_ids)


def test_child_chunks_are_smaller_than_parents():
    """Child chunks should generally be smaller than parent chunks."""
    parents, children = hierarchical_chunk_text(
        ARABIC_TEXT,
        max_parent_chars=500,
        max_child_chars=200,
    )
    
    for child in children:
        # Child text should be reasonably small
        assert len(child.text) <= 300  # Some buffer for sentence boundaries
    
    for parent in parents:
        # Parent text should be larger
        assert len(parent.text) >= 100


def test_hierarchical_chunking_empty_text():
    parents, children = hierarchical_chunk_text("")
    assert parents == []
    assert children == []


def test_hierarchical_chunking_respects_max_parent_chars():
    """Parent chunks should respect the max_parent_chars limit (soft)."""
    parents, _ = hierarchical_chunk_text(
        ARABIC_TEXT,
        max_parent_chars=300,
        max_child_chars=150,
    )
    
    # Most parents should be under the limit (soft limit, so some may exceed slightly)
    oversized = [p for p in parents if len(p.text) > 400]
    # Allow a few oversized due to sentence boundary respect
    assert len(oversized) <= 1


def test_build_parent_chunks_standalone():
    """Test parent chunk building independently."""
    parents = build_parent_chunks(ARABIC_TEXT, max_parent_chars=400, overlap_sentences=1)
    
    assert len(parents) >= 1
    for p in parents:
        assert p.parent_id.startswith("parent_")
        assert p.text
        assert p.start_sentence_idx <= p.end_sentence_idx


def test_build_child_chunks_standalone():
    """Test child chunk building from a parent."""
    from src.chunking.hierarchical_chunker import ParentChunk
    
    parent = ParentChunk(
        text="الجملة الأولى. الجملة الثانية. الجملة الثالثة. الجملة الرابعة.",
        parent_id="parent_0",
        start_sentence_idx=0,
        end_sentence_idx=3,
    )
    
    children = build_child_chunks(parent, max_child_chars=50, overlap_sentences=0)
    
    assert len(children) >= 1
    for c in children:
        assert c.child_id.startswith("parent_0_child_")
        assert c.parent_id == "parent_0"
        assert c.text


def test_hierarchical_chunk_text_backward_compat():
    """Test that chunk_text (backward compat) returns parent chunks."""
    from src.chunking.hierarchical_chunker import chunk_text
    
    chunks = chunk_text(ARABIC_TEXT, max_chunk_chars=400, overlap_sentences=1)
    
    assert len(chunks) >= 1
    for c in chunks:
        assert hasattr(c, 'text')
        assert hasattr(c, 'start_sentence_idx')
        assert hasattr(c, 'end_sentence_idx')


def test_long_unpunctuated_text_is_split():
    """A punctuation-free poem must not yield one giant parent/child chunk."""
    text = " ".join(["كلمة"] * 500)
    parents, children = hierarchical_chunk_text(
        text,
        max_parent_chars=400,
        max_child_chars=150,
        parent_overlap_sentences=1,
        child_overlap_sentences=0,
    )

    assert parents
    assert children
    assert max(len(p.text) for p in parents) <= 850
    assert max(len(c.text) for c in children) <= 300
    for child in children:
        assert set(child.text.split()) == {"كلمة"}
