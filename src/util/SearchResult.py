import json


class SearchResult:
    def __init__(self, snippet: str = "", title: str = "", link: str = "") -> None:
        self.snippet: str = snippet
        self.title: str = title
        self.link: str = link

    def __repr__(self) -> str:
        return f"SearchResult(snippet='{self.snippet}', title='{self.title}', link='{self.link}')"

    def to_json(self) -> str:
        return json.dumps({
            'snippet': self.snippet,
            'title': self.title,
            'link': self.link
        }, indent=4)
