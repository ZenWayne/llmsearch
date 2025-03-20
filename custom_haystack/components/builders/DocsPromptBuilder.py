from haystack import Document, component, logging
from haystack.components.builders import PromptBuilder
from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment
import asyncio
from typing import List, Optional, Dict, Any, Union, Literal

@component
class DocsPromptBuilder(PromptBuilder):
    """
    搭配DocumentSplitter输出Document中的doc.meta['source_id']
    将切分的文档去重并输出索引列表

    使用示例：
    ```python
    from custom_haystack.components.fetcher.url_to_markdown import URLMarkdownFetcher
    datas = [
        {
            "content": "This is a test document.",
            "meta": {
                "url": "https://example.com",
                "title": "Test Document",
                "source_id": "safweaggwe2"
            }
        },
        {
            "content": "This is another test document.",
            "meta": {
                "url": "https://example.com",
                "title": "Another Test Document",
                "source_id": "acsddwfq1"
            }
        }
    ]

    DocList= [Document(content=data["content"], meta=data["meta"]) for data in datas]
    fetcher = DocsPromptBuilder()
    results = fetcher.run(documents=DocList)
    documents = results["documents"]
    ```
    """
    def __init__(self,
        template: str
    ):
        """
        Constructs a PromptBuilder component.

        :param template:
            A prompt template that uses Jinja2 syntax to add variables. For example:
            `"Summarize this document: {{ documents[0].content }}\\nSummary:"`
            It's used to render the prompt.
            The variables in the default template are input for PromptBuilder and are all optional,
            unless explicitly specified.
            If an optional variable is not provided, it's replaced with an empty string in the rendered prompt.

        """
        required_variables = ["contents", "references"]
        variables = ["contents", "references"]
        self._template_string = template
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []

        self._env = SandboxedEnvironment()

        self.template = self._env.from_string(template)
        if not variables:
            # infer variables from template
            ast = self._env.parse(template)
            template_variables = meta.find_undeclared_variables(ast)
            variables = list(template_variables)
        variables = variables or []
        self.variables = variables

        # setup inputs
        for var in self.variables:
            if self.required_variables == "*" or var in self.required_variables:
                component.set_input_type(self, var, Any)
            else:
                component.set_input_type(self, var, Any, "")
    
    @component.output_types(prompt=str)
    def run(self, template: Optional[str] = None, documents: List[Document] = None, **kwargs):
        """
        Run the InMemoryEmbeddingRetriever on the given input data.

        :param documents:
            A list of Document objects.
        :returns:
            One Giant Document objects combined with the prompt template.

        :raises ValueError:
            If the specified DocumentStore is not found or is not an InMemoryDocumentStore instance.
        """
        kwargs = kwargs or {}
        template_variables = {"contents": "", "references": ""}
        source_ids_map = {}
        index = 0
        for doc in documents:
            source_id = doc.meta["source_id"]
            if source_id not in source_ids_map:
                source_ids_map[source_id] = {"docs": [doc], "index": index}
                index += 1
            else:
                source_ids_map[source_id]["docs"].append(doc)

        template_variables["contents"] = "\n".join([f"Document <{source_ids_map[doc.meta['source_id']]['index']}>:\n{doc.content}" for doc in documents])
        template_variables["references"] = "\n".join([f"Document <{v['index']}>[{v['docs'][0].meta['title']}]({v['docs'][0].meta['url']})" for k, v in source_ids_map.items()])
        
        self._validate_variables(set(template_variables.keys()))

        compiled_template = self.template
        if template is not None:
            compiled_template = self._env.from_string(template)

        result = compiled_template.render(template_variables)
        return result


if __name__ == "__main__":
    from custom_haystack.components.fetcher.url_to_markdown import URLMarkdownFetcher
    datas = [
        {
            "content": "This is a test document.",
            "meta": {
                "url": "https://example.com",
                "title": "Test Document",
                "source_id": "safweaggwe2"
            }
        },
        {
            "content": "This is second test document.",
            "meta": {
                "url": "https://example.com",
                "title": "Test Document",
                "source_id": "safweaggwe2"
            }
        },
        {
            "content": "This is another test document.",
            "meta": {
                "url": "https://example.com",
                "title": "Another Test Document",
                "source_id": "acsddwfq1"
            }
        }
    ]

    template = """
## Input Data

### 【Web Page】
{{contents}}

### 【References】
{{references}}
"""
    DocList= [Document(content=data["content"], meta=data["meta"]) for data in datas]
    prompt_builder = DocsPromptBuilder(template=template)
    results = prompt_builder.run(documents=DocList)
    print(results)
