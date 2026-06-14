from langchain_core.documents import Document

from app.rag.ingest import html_to_markdown
from app.rag.pipeline import _dedup_sources, _format_context


def test_code_block_becomes_fenced():
    md = html_to_markdown("<p>do this</p><pre><code>x = [1, 2]\nx[::-1]</code></pre>")
    assert "```" in md
    assert "x[::-1]" in md


def test_inline_code_and_entities():
    md = html_to_markdown("<p>call <code>read_csv</code> &amp; go</p>")
    assert "`read_csv`" in md
    assert "&" in md and "&amp;" not in md


def test_format_context_is_numbered_with_urls():
    docs = [
        Document(page_content="body one", metadata={"title": "T1", "url": "http://a"}),
        Document(page_content="body two", metadata={"title": "T2", "url": "http://b"}),
    ]
    ctx = _format_context(docs)
    assert "[1]" in ctx and "[2]" in ctx
    assert "http://a" in ctx and "http://b" in ctx


def test_dedup_sources_drops_repeats():
    docs = [
        Document(page_content="a", metadata={"title": "T", "url": "http://u"}),
        Document(page_content="b", metadata={"title": "T", "url": "http://u"}),
    ]
    assert len(_dedup_sources(docs)) == 1
