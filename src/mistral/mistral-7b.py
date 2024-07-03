import os
os.environ["HUGGINGFACE_TOKEN"] = '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'

from huggingface_hub import login
login(token="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")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
    pipeline
)

import fitz  # PyMuPDF
import pandas as pd
import nest_asyncio
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

from operator import itemgetter



#################################################################
# Raw data from HP
#################################################################
file_path = r'c:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Summer Project\1038-0610-0614-day.xlsx'
telemetry_data = pd.read_excel(file_path)
print(telemetry_data)

# Custom Document class to hold text content and metadata
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

#################################################################
# Tokenizer
#################################################################

model_name='mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
mistral_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(mistral_model))

standalone_query_generation_pipeline = pipeline(
 model=mistral_model,
 tokenizer=tokenizer,
 task="text-generation",
 temperature=0.0,
 repetition_penalty=1.1,
 return_full_text=True,
 max_new_tokens=1000,
)
standalone_query_generation_llm = HuggingFacePipeline(pipeline=standalone_query_generation_pipeline)

response_generation_pipeline = pipeline(
 model=mistral_model,
 tokenizer=tokenizer,
 task="text-generation",
 temperature=0.2,
 repetition_penalty=1.1,
 return_full_text=True,
 max_new_tokens=1000,
)
response_generation_llm = HuggingFacePipeline(pipeline=response_generation_pipeline)


import nest_asyncio
nest_asyncio.apply()

# Articles to index
articles = [r"file://C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Summer Project\HPSusSys\data\z2 mini G9.pdf",
            r"file://C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Summer Project\HPSusSys\data\ZCentral 4R.pdf"
            # "https://www.fantasypros.com/2023/11/nfl-week-10-sleeper-picks-player-predictions-2023/",
            # "https://www.fantasypros.com/2023/11/nfl-dfs-week-10-stacking-advice-picks-2023-fantasy-football/",
            # "https://www.fantasypros.com/2023/11/players-to-buy-low-sell-high-trade-advice-2023-fantasy-football/"
            ]


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_articles(articles):
    pdf_texts = []
    web_articles = []
    for article in articles:
        if article.startswith("file://"):
            pdf_path = article[7:]  # Remove 'file://' prefix
            pdf_texts.append(extract_text_from_pdf(pdf_path))
        else:
            web_articles.append(article)
    return pdf_texts, web_articles

pdf_texts, web_articles = extract_text_from_articles(articles)

# Process web articles with AsyncChromiumLoader
loader = AsyncChromiumLoader(web_articles)
docs = loader.load()

# Convert HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Add the PDF texts to the transformed docs
for text in pdf_texts:
    docs_transformed.append(Document(page_content=text, metadata={"source": "pdf"}))

# Verify documents
print("Number of documents:", len(docs_transformed))
for doc in docs_transformed:
    print(f"Source: {doc.metadata['source']}, Content: {doc.page_content[:200]}...")

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=800, 
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

retriever = db.as_retriever(k = 1)

_template = """
[INST] 
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index. This query will be used to retrieve documents with additional context. 

Let me share a couple examples that will be important. 

If you do not see any chat history, you MUST return the "Follow Up Input" as is:

```
Chat History:

Follow Up Input: Is the z2 mini or the ZCentral more environemntally frirendly?
Standalone Question:
Is the z2 mini or the ZCentral more environemntally frirendly?
```

If this is the second question onwards, you should properly rephrase the question like this:

```
Chat History:
Human: Is the z2 mini or the ZCentral more environemntally frirendly?
AI: 
The z2 mini is more environmentally friendly.

Follow Up Input: Why is it?
Standalone Question:
Why the z2 mini or the ZCentral more environemntally frirendly?
```

Now, with those examples, here is the actual chat history and input question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:
[your response here]
[/INST] 
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """
[INST] 
Answer the question based only on the following context:
{context}

Question: {question}
[/INST] 
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Instantiate ConversationBufferMemory
memory = ConversationBufferMemory(
 return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | standalone_query_generation_llm,
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | response_generation_llm,
    "question": itemgetter("question"),
    "context": final_inputs["context"]
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer


#################################################################
# Conversation Function
#################################################################

def call_conversational_rag(question, chain, memory):
    """
    Function to call a conversational RAG (Retrieval-Augmented Generation) model to generate an answer.

    Parameters:
    - question (str): The question to be answered.
    - chain (LangChain object): The LangChain instance encapsulating the RAG model and its workflow.
    - memory (Memory object): Object used for storing conversation context.

    Returns:
    - dict: Generated answer from the RAG model.
    """
    inputs = {"question": question}
    result = chain.invoke(inputs)
    memory.save_context(inputs, {"answer": result["answer"]})
    return result

#################################################################
# Main Execution
#################################################################

import sys

if len(sys.argv) > 1:
    question = sys.argv[1]
else:
    question = input("Enter your question: ")

# Call the conversational RAG model with the input question
result = call_conversational_rag(question, final_chain, memory)

# Print the generated answer
print("Generated Answer:", result["answer"])