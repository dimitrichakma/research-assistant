import arxiv


def search_arxiv(query, max_results=10, candidate_pool=30):
    """Find recent, on-topic papers for a free-text query like "latest
    papers on human-computer interaction".

    No LLM/TypeSafe call here - this is a lookup, not a judgment or
    generation task.

    Neither of arXiv's own sort modes fits "latest on X" alone (checked
    live against real queries): SubmittedDate ignores topic match
    entirely and returns near-random recent papers; Relevance finds
    genuinely on-topic papers but mixes in decade-old ones ahead of
    recent ones. Fixed by pulling a relevance-ranked candidate pool,
    then re-sorting that pool by publish date - recent AND on-topic,
    since every candidate already cleared the relevance bar.

    Returns a list of dicts: title, authors, published (date), summary,
    url (abstract page), pdf_url.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=candidate_pool,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    candidates = list(client.results(search))
    candidates.sort(key=lambda r: r.published, reverse=True)
    return [
        {
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "published": r.published.date(),
            "summary": r.summary,
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
        }
        for r in candidates[:max_results]
    ]


if __name__ == "__main__":
    query = input("Search arXiv for: ")
    for paper in search_arxiv(query):
        print(f"\n{paper['published']} - {paper['title']}")
        print(f"  {', '.join(paper['authors'])}")
        print(f"  {paper['url']}")
