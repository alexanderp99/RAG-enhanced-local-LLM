from langchain_core.documents.base import Document

class DocumentWrapper(Document):
    def __init__(self, document: Document) -> None:
        super().__init__(
            page_content=document.page_content,
            metadata=document.metadata,
            id=document.id,
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, DocumentWrapper):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        if self.id is None:
            raise ValueError("Cannot hash a Document with no `id`.")
        return hash(self.id)

    def __repr__(self) -> str:
        return f"DocumentWrapper(id={self.id}, page_content={self.page_content[:50]}...)"
