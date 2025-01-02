from typing import Optional, List
from pydantic import Field
from just_semantic_search.document import ArticleDocument
import patito as pt


SCHOLAR_ALL_COLUMNS = [
    'corpusid', 
    'updated', 
    'content_source_oainfo_license', 
    'content_source_oainfo_openaccessurl', 
    'content_source_oainfo_status', 
    'content_source_pdfsha', 
    'content_source_pdfurls', 
    'externalids_acl', 
    'externalids_arxiv', 
    'externalids_dblp', 
    'externalids_doi', 
    'externalids_mag', 
    'externalids_pubmed', 
    'externalids_pubmedcentral', 
    'content_text', 
    'annotations_abstract', 
    'annotations_author', 
    'annotations_authoraffiliation', 
    'annotations_authorfirstname', 
    'annotations_authorlastname', 
    'annotations_bibauthor', 
    'annotations_bibauthorfirstname', 
    'annotations_bibauthorlastname', 
    'annotations_bibentry', 
    'annotations_bibref', 
    'annotations_bibtitle', 
    'annotations_bibvenue', 
    'annotations_figure', 
    'annotations_figurecaption', 
    'annotations_figureref', 
    'annotations_formula', 
    'annotations_paragraph', 
    'annotations_publisher', 
    'annotations_sectionheader', 
    'annotations_table', 
    'annotations_tableref', 
    'annotations_title', 
    'annotations_venue'
]



SCHOLAR_MAIN_COLUMNS = [
    'corpusid',
    'content_source_oainfo_openaccessurl',
    'updated',
    'externalids_doi',
    'externalids_pubmed',
    'annotations_abstract',
    'annotations_author',
    'annotations_title',
    'annotations_paragraph',
    "annotations_bibref", 
    "annotations_bibentry"
    #'content_text'
]

class Paper(pt.Model):
    corpusid: Optional[int] = Field(default=None)
    content_source_oainfo_openaccessurl: Optional[str] = Field(default=None)
    updated: Optional[str] = Field(default=None)
    externalids_doi: Optional[str] = Field(default=None)
    externalids_pubmed: Optional[str] = Field(default=None)
    annotations_abstract: Optional[List[str]] = Field(default=None)
    annotations_author: Optional[List[str]] = Field(default=None)
    annotations_title: Optional[List[str]] = Field(default=None)
    annotations_paragraph: Optional[List[str]] = Field(default=None)
    annotations_bibref: Optional[List[str]] = Field(default=None)
    annotations_bibentry: Optional[List[str]] = Field(default=None)

    @property
    def title(self) -> str:
        return self.annotations_title[0] if self.annotations_title else None
    
    @property
    def abstract(self) -> str:
        return self.annotations_abstract[0] if self.annotations_abstract else None
    
    @property
    def references(self) -> Optional[str]:
        if self.annotations_bibentry is None or self.annotations_bibref is None:
            return None
        result_list = [f"{ref} {entry}" for ref, entry in zip(self.annotations_bibref, self.annotations_bibentry)]
        return '\n'.join(result_list)
    
    @property
    def references_entries(self) -> List[str]:
        return self.annotations_bibentry if self.annotations_bibentry else None
    

    def to_article_document(self) -> ArticleDocument:
        # Create metadata dictionary with all available fields
        metadata = {
            'corpusid': self.corpusid,
            'content_source_oainfo_openaccessurl': self.content_source_oainfo_openaccessurl,
            'updated': self.updated,
            'externalids_doi': self.externalids_doi,
            'externalids_pubmed': self.externalids_pubmed,
            'annotations_author': self.annotations_author,
            'annotations_bibref': self.annotations_bibref,
            'annotations_bibentry': self.annotations_bibentry
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return ArticleDocument(
            title=self.annotations_title,
            abstract=self.annotations_abstract,
            fragment_num=0,
            total_fragments=1,
            text=self.annotations_paragraph[0] if self.annotations_paragraph else None,
            metadata=metadata
        )


