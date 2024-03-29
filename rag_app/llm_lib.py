import os
import threading
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import IO, Any, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)
from transformers.generation.streamers import BaseStreamer


def construct_rag_store(pdfs: List[IO], embedding_model: Embeddings) -> VectorStore:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    pages = [PyPDFLoader(f"{pdf}").load_and_split(text_splitter) for pdf in pdfs]

    faiss_index = FAISS.from_documents(pages[0], embedding_model)

    for page in pages[1:]:
        faiss_index.merge_from(FAISS.from_documents(page, embedding_model))

    return faiss_index


def construct_bge_model(bge_type: str = "BAAI/bge-large-en-v1.5") -> HuggingFaceBgeEmbeddings:
    model_name = bge_type
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this sentence for searching relevant passages: ",
    )

    return embedding_model


def construct_gen_model(
    model_name: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
) -> Tuple[BaseLLM, BaseStreamer]:
    # Load model definition config
    model_config = AutoConfig.from_pretrained(model_name)
    if "gemma" in model_name:
        # If using gemma patch the approx gelu
        model_config.hidden_act = "gelu_pytorch_tanh"
        model_config.hidden_activation = "gelu_pytorch_tanh"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )

    streamer = TextIteratorStreamer(tokenizer)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.5,
        top_k=500,
        do_sample=True,
        streamer=streamer,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm, streamer


def construct_ggml_model(model_path: str, context_size: int = 32768, stream: bool = False) -> Llama:

    llm = Llama(
        model_path,  #'./qwen-2-4b/ggml-model-Q4_K_M.gguf',
        n_gpu_layers=-1,
        n_ctx=context_size,
        chat_format="chatml",
        verbose=True,
        stream=stream,
    )
    return llm


def context_retrieval(query_input: Any, vector_store: VectorStore = None):
    docs = vector_store.max_marginal_relevance_search(
        query_input["prompt"], k=7, fetch_k=20, lambda_mult=0.4
    )

    contexts = []
    for doc in docs:
        context_str = ""
        if "source" in doc.metadata:
            context_str += f'Source: {Path(doc.metadata["source"]).name} - '
        if "page" in doc.metadata:
            context_str += f'Data: {doc.metadata["page"]}: {doc.page_content}'
        contexts.append(context_str)

    return "\n".join(contexts)


def construct_chain(
    prompt: BasePromptTemplate, llm: BaseLLM, vector_store: VectorStore = None
) -> RunnableSerializable:
    context_dict = {}
    if vector_store is not None:
        f_context_retrieval = partial(context_retrieval, vector_store=vector_store)
        context_dict = {"context": f_context_retrieval}

    return {**context_dict, "prompt": itemgetter("prompt")} | prompt | llm


if __name__ == "__main__":
    hf = False
    load_dotenv(Path(__file__).parent / ".env")
    login(os.environ["HF_TOKEN"])
    pdfs = list(Path("notebooks/data").absolute().iterdir())

    if not hf:
        # llm = construct_ggml_model("./notebooks/qwen-2-4b/ggml-model-Q4_K_M.gguf")

        pdfs = ["notebooks\\data\\Sophia+CV+2021.pdf"]
        embedder = construct_bge_model()
        vector_store = construct_rag_store(pdfs, embedder)

        query = "What future roles should I apply for?"

        context = context_retrieval({"prompt": query}, vector_store)

        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a passionate career adviser and life coach helping users to plan their future.",
                },
                {"role": "user", "content": f"{query}\n\nCONTEXT\n{context}"},
            ],
            temperature=0.5,
        )
        print(output["choices"][0])
    else:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        llm, streamer = construct_gen_model("google/gemma-2b-it", nf4_config)
        embedder = construct_bge_model()
        vector_store = construct_rag_store(pdfs, embedder)
        context_retrieval({"prompt": "What is the meaning of 42"}, vector_store)

        prompt = ChatPromptTemplate.from_template(
            "<bos><start_of_turn>user\n"
            "I will provide you with a question, signified by PROMPT followed by a new line and some context "
            "signified by CONTEXT and a new line. "
            "Using the context try to answer the question to the best of your ability. "
            "If you cannot answer based on the context alone, please still do attempt to resolve the "
            "question but be explicit in letting the user know that you have used your own knowledge.\n\n"
            # "Using either your own expert knowledge (no tag), enriched with the following contextual data (denoted by CONTEXT), "
            # "please answer my prompt (denoted by PROMPT)\nIf there you cannot answer from the context alone, please let "
            # "the user know, but still answer with your expert knowledge.\n\n"
            "CONTEXT\n{context}\n\nPROMPT\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        )

        chain = construct_chain(prompt=prompt, llm=llm, vector_store=vector_store)

        producer = threading.Thread(target=chain.invoke, args=({"prompt": "What is the meaning of 42"},))
        producer.start()

        for part in streamer:
            print(part, end="")
