import itertools
import os
import threading
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import (
    IO,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextGenerationPipeline,
    TextIteratorStreamer,
    pipeline,
)
from transformers.generation.streamers import BaseStreamer

try:
    import flash_attn  # noqa

    HAS_FLASH_ATTENTION = True
except ModuleNotFoundError:
    HAS_FLASH_ATTENTION = False


CHAT_ROLE = Literal["assistant"] | Literal["user"]

GENERATION_THREADS: List[threading.Thread] = []


class ChatMessage(BaseModel):
    role: CHAT_ROLE
    content: str


class ChatHistory(BaseModel):
    history: List[ChatMessage]


def load_hf_model(
    model_path: str,
    load_options: dict,
    generation_params: Optional[dict] = None,
) -> Tuple[TextGenerationPipeline, TextIteratorStreamer]:
    model_config = AutoConfig.from_pretrained(model_path)
    if "gemma" in model_path:
        # If using gemma patch the approx gelu
        model_config.hidden_act = "gelu_pytorch_tanh"
        model_config.hidden_activation = "gelu_pytorch_tanh"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "attn_implementation" not in load_options:
        load_options["attn_implementation"] = "flash_attention_2" if HAS_FLASH_ATTENTION else None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_options,
        config=model_config,
    )

    streamer = TextIteratorStreamer(tokenizer)

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        **(generation_params if generation_params else {}),
    )

    return (pipe, streamer)


def load_ggml_model(model_path: str, load_options: dict) -> Llama:
    llm = Llama(
        model_path=model_path,
        stream=True,
        **load_options,
        chat_format="chatml",
    )

    return llm


def generate_hf_model(
    model: TextGenerationPipeline,
    streamer: TextIteratorStreamer,
    chat_history: ChatHistory,
) -> Generator[str, None, None]:
    global GENERATION_THREADS

    chat_dict = dump_history(chat_history)
    inputs = model.tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=True)
    # TODO: fix issue here with threading
    # producer = threading.Thread(group=None, target=model.__call__, args=[inputs])
    # producer.start()  # TODO: maybe issues here if the stream is not fully terminated
    # return streamer

    # Define a thread complete flag
    thread_complete = threading.Event()

    # Define a wrapper function for the model call to signal completion
    def model_call_wrapper():
        try:
            model.__call__(inputs)
        finally:
            # Signal that the thread's work is done
            thread_complete.set()

    # Start the producer thread
    producer = threading.Thread(target=model_call_wrapper)
    producer.start()

    # Generator function to yield from streamer and ensure thread cleanup
    def generator_wrapper():
        # Yield the results from the streamer as they become available
        for item in streamer:
            yield item

        # Wait for the producer thread to complete before exiting the generator
        producer.join()

    GENERATION_THREADS.append(producer)
    return generator_wrapper()


def ggml_generator(ggml_output_stream: Any) -> Generator[str, None, None]:
    next(ggml_output_stream)  # skip assistant line

    text = ""
    for tok in ggml_output_stream:
        if not tok["choices"]:
            break

        choice = tok["choices"][0]
        if not choice["delta"]:
            break

        text += choice["delta"]["content"]
        yield text


def generate_ggml_model(
    model: Llama, generation_params: dict, chat_history: ChatHistory
) -> Generator[str, None, None]:
    chat_dict = dump_history(chat_history)
    completion = model.create_chat_completion(
        messages=chat_dict,
        stream=True,
        **generation_params,
    )
    return ggml_generator(completion)


class StreamingLLM:
    def __init__(
        self,
        model_base: Literal["ggml", "hf"],
        model_weights: str,
        load_params: dict,
        generate_params: dict,
        system_prompt: Optional[str] = None,
    ):
        self.model_base = model_base
        self.model_weights = model_weights
        self.load_params = load_params
        self.generate_params = generate_params
        self.chat_history = ChatHistory.model_validate(
            {"history": [{"role": "system", "content": system_prompt}] if system_prompt else []}
        )
        # add_message(self.chat_history, "user", "what is the capital of france")

        if model_base == "hf":
            self.llm, self.streamer = load_hf_model(model_weights, load_params, generate_params)
            self._generate = partial(
                generate_hf_model, model=self.llm, streamer=self.streamer, chat_history=self.chat_history
            )
            self.eos_token = self.llm.tokenizer.eos_token
        elif model_base == "ggml":
            self.llm = load_ggml_model(model_weights, load_params)
            self._generate = partial(
                generate_ggml_model,
                model=self.llm,
                generation_params=generate_params,
                chat_history=self.chat_history,
            )
            self.eos_token = self.llm.tokenizer_.detokenize([self.llm.token_eos()])[0]

    def generate(self, prompt):
        # 1. do some shit with chat history
        add_message(self.chat_history, role="user", message=prompt)

        # 2. pass it to generate
        generation = self._generate()

        # 3. yield all the parts
        text = ""

        next(generation)  # skips the first 'assistant' token
        for part in generation:
            if part == self.eos_token:
                break
            elif part.endswith(self.eos_token):
                text += part[: part.rindex(self.eos_token)]
            else:
                text += part
            yield text

        # 4. after end of generator parse the output and add it back to chat history
        add_message(
            self.chat_history, role="assistant", message=text
        )  # TODO: handle not adding the EOS token


def load_history(history_data: dict) -> ChatHistory:
    return ChatHistory.model_validate(history_data)


def dump_history(chat_history: ChatHistory) -> List[Any]:
    return chat_history.model_dump()["history"]


def add_message(chat_history: ChatHistory, role: CHAT_ROLE, message: str):
    chat_history.history.append(ChatMessage.model_validate({"role": role, "content": message}))


def make_chat(chat_history: ChatHistory, model: StreamingLLM):
    pass


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
