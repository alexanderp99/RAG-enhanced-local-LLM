import json


class SearchResult:
    def __init__(self, snippet="", title="", link=""):
        self.snippet = snippet
        self.title = title
        self.link = link

    def __repr__(self):
        return f"SearchResult(snippet='{self.snippet}', title='{self.title}', link='{self.link}')"

    def to_json(self):
        return json.dumps({
            'snippet': self.snippet,
            'title': self.title,
            'link': self.link
        }, indent=4)
