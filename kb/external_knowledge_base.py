import wikipedia

from kb.models import Entity, ExternalKnowledgeBase


class WikipediaKnowledgeBase(ExternalKnowledgeBase):
    def get_entity(self, entity_candidate: str) -> Entity | None:
        try:
            page = wikipedia.page(entity_candidate, auto_suggest=False)
            # return Entity(title=page.title, url=page.url, summary=page.summary)
            return Entity(title=page.title, url=page.url, summary="")
        except:
            return None


default_external_knowledge_bases = [WikipediaKnowledgeBase()]
