import time
import asyncio as aio
from queue import Queue
import langchain as lc
import pandas as pd
import numpy as np
from requests import request
import os
from datetime import datetime, timedelta
from typing import List, Optional
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from cohere.errors.bad_request_error import BadRequestError
# from langchain_cohere import Cohere  # ERROR !!!!
from langchain_community.llms import Cohere
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
import re
import json
from pypdf import PdfReader, PdfWriter
import pickle as pkl

# Constants
URL = "https://group.intesasanpaolo.com/content/dam/portalgroup/repository-documenti/research/it/economia-e-finanza-dei-distretti/2023/Rapporto%20Economia%20e%20finanza%20dei%20distretti%20industriali%20nr%2015_Bis.pdf"
DATA_PATH = os.path.join(os.environ.get("DATA_PATH"))
API_KEY_COHERE = os.environ.get("API_KEY_COHERE")
API_KEY_ANTHROPIC = os.environ.get("API_KEY_ANTHROPIC")
PDF_FILE_PATH = os.path.join(DATA_PATH, "pdf.pdf")
PDF_FILE_SHORT_PATH = os.path.join(DATA_PATH, "short.pdf")
FROM_PAGE = int(os.environ.get("FROM_PAGE"))
TO_PAGE = int(os.environ.get("TO_PAGE"))
SELECT_COHERE = True if os.environ.get("COHERE") == "True" else False
MODEL_OUTPUT_FILE_NAME = (
    f"{'COHERE' if SELECT_COHERE else 'ANTHROPIC'}_model_output.csv"
)
MODEL_OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "_".join([datetime.now().strftime("%Y%m%d-%H%M%S"), MODEL_OUTPUT_FILE_NAME]),
)
TEMPERATURE = float(os.environ.get("TEMPERATURE"))


def get_document(from_page, to_page):
    """Get PDF and save relevant pages

    Args:
        from_page (int): beginning page
        to_page (int): ending page
    """
    if not os.path.isfile(PDF_FILE_PATH):
        res = request(method="GET", url=URL)
        if res.status_code == 200:
            print("request successful")
            with open(PDF_FILE_PATH, "wb") as file:
                file.write(res.content)
            pdf_writer = PdfWriter()
            reader = PdfReader(PDF_FILE_PATH)
            print(f"Saving pdf from page {FROM_PAGE} to page {TO_PAGE}")
            pages = reader.pages[from_page:to_page]
            for page in pages:
                pdf_writer.add_page(page)
            with open(PDF_FILE_SHORT_PATH, "wb") as file:
                pdf_writer.write(file)
        else:
            print("Couldn't find the document from the link provided")
    else:
        print("file already downloaded")


class Chapter(BaseModel):
    """
    Information about a chapter inside a document.
    The document is structured in chapters and paragraphs.
    The ID of the chapter is identified by a number, for example 2 .
    The sub-chapter is identified by the parent chapter and a number,
    for example, the parent chapter is 2 and the subchapter is 2.1.

    Each chapter and paragraph must have a title, a summary corresponding to the section,
    a positive, neutral or negative sentiment about the section and a list of topics associated the the section.
    The sentiment can have three values: 'Negativo', 'Neutro', 'Positivo'.

    All extracted title, summary, sentiment and topic must be in the Italian Language. Internally
    translate and output italian text.
    """

    chapter_id: str = Field(
        ...,
        description="""
            The progressive numeric ID of the chapter or paragraph,
            with progressive numeration 2 -> 2.1, 2.2. If it's the
            appendix then add .x after the main chapter number, for
            example 2.x""",
    )
    title: str = Field(
        ...,
        description="The Title of the chapter or paragraph",
    )
    summary: str = Field(
        ...,
        description="The short summary of the chapter or paragraph identifying the key concepts",
    )
    sentiment: str = Field(
        ...,
        description="The sentiment of the chapter, the only values allowed are 'Negativo', 'Neutro', 'Positivo'.",
    )
    topics: List[str] = Field(
        ...,
        description="""
                    A list of main topics from the chapter separated.
                    Each topic must be a single word or concept.
                    Each topic must come from a economics topic, for example revenues, increase demand, manufacturing sector etc...\n
                    Each section can have a different number of topics. 
                    """,
    )


class Chapters(BaseModel):
    """Identifying information about all chapters in a text."""

    chapters: List[Chapter]


async def get_chain_reponse(chain, query, queue: aio.Queue) -> BaseModel:
    """Invoke chain request for a single block of text

    Args:
        chain (langchain.chain): chain producing the analysis
        query (str): string of text to analyze
        queue (aio.Queue): async queue

    Returns:
        output (BaseModel): structured output from PydanticOutputParser
    """
    try:
        res = chain.invoke({"query": query})
        res = res.chapters
        return res
    except BadRequestError as br:
        print(f"Bad Request from Cohere {br}")
        await queue.put(query)
    except Exception as e:
        print(e)
        await queue.put(query)


async def gather_tasks(chain, req_queue: aio.Queue, max_req: int) -> List[BaseModel]:
    """Gather all tasks to asynchronously get responses from the model.

    Args:
        chain (langchain.chain): chain producing the analysis
        req_queue (aio.Queue):  async queue
        max_req (int): maximum number of requests to gather

    Returns:
        List[BaseModel]: return a list of the output from the PydanticOutputParser
    """
    print(f"Responses to go {req_queue.qsize()}")
    async_queries = []
    for _ in range(max_req):
        if not req_queue.empty():
            async_queries.append(
                get_chain_reponse(chain, await req_queue.get(), queue=req_queue)
            )
        else:
            break
    res = await aio.gather(*async_queries)
    return res


async def populate_queue(splits: List[str], queue: aio.Queue):
    """Populated the asynchronous queue with all the splits from the document parser

    Args:
        splits (List[str]): splits from document parser
        queue (aio.Queue): asynchronous queue to populate

    Returns:
        aio.Queue: returns the queue
    """
    for split in splits:
        await req_queue.put(split)
    return queue


if __name__ == "__main__":
    if SELECT_COHERE:
        model = Cohere(
            model="command",
            temperature=0,
            cohere_api_key=API_KEY_COHERE,
        )
        MAX_TOKENS = 3000
        MAX_REQUEST_NUMBER = 10
        TIME_LIMIT = 60
    else:
        model = ChatAnthropic(
            model_name="claude-3-sonnet-20240229",
            temperature=0,
            api_key=API_KEY_ANTHROPIC,
        )
        MAX_TOKENS = 20000 / 5
        MAX_REQUEST_NUMBER = 5
        TIME_LIMIT = 60

    # save pdf document on shared volume
    get_document(FROM_PAGE, TO_PAGE)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS,
        chunk_overlap=60,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    )

    # Load and split pdf pages into text chunks
    pdf = PyPDFLoader(PDF_FILE_SHORT_PATH)
    splits = pdf.load_and_split(text_splitter=splitter)

    parser = PydanticOutputParser(pydantic_object=Chapters)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user query. Make absolutely sure to wrap the output in `json` tags\n{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt_template | model | parser
    # prompt_query = prompt_template.format_prompt(query=splits).to_string(); print(prompt_query)
    print("Getting async response from model...")
    res = []
    req_queue = aio.Queue()
    req_queue = aio.run(populate_queue(splits, req_queue))

    # run the async gathering process to deal with api limits
    while not req_queue.empty():
        async_tasks = gather_tasks(
            chain=chain, max_req=MAX_REQUEST_NUMBER, req_queue=req_queue
        )
        start_time = datetime.now()
        results = aio.run(async_tasks)
        sub_res = []
        for result in results:
            sub_res.extend(result)
        res.extend(sub_res)
        now = datetime.now()
        delta = (now - start_time).seconds
        if delta < TIME_LIMIT:
            print(f"Waiting for {TIME_LIMIT - delta} seconds")
            time.sleep(TIME_LIMIT - delta)

    # res = chain.invoke({"query": splits}).chapters
    print("Converting output to pandas DataFrame")
    res_df = pd.DataFrame(
        [
            {
                "ID": str(item.chapter_id),
                "Title": item.title,
                "Summary": item.summary,
                "Sentiment": item.sentiment,
                "Topics": ",".join(item.topics),
            }
            for item in res
        ]
    )
    res_df.sort_values(by="ID", inplace=True)
    print(f"Saving output to {MODEL_OUTPUT_FILE}")
    res_df.to_csv(MODEL_OUTPUT_FILE)
    print(f"Output Saved!")
    