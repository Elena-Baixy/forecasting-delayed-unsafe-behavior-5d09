import arxiv, sys, json
client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
queries = sys.argv[1:]
seen = {}
for q in queries:
    try:
        for r in client.results(arxiv.Search(query=q, max_results=12, sort_by=arxiv.SortCriterion.Relevance)):
            aid = r.get_short_id().split('v')[0]
            if aid in seen: continue
            seen[aid] = 1
            print(f"{aid} | {r.published.year} | {r.title} | {r.summary[:220].replace(chr(10),' ')}")
    except Exception as e:
        print("ERR", q, e)
