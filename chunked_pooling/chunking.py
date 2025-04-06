import bisect
import logging
import torch
from typing import Dict, List, Optional, Tuple, Union
from chonkie import SDPMChunker, SemanticChunker, SentenceTransformerEmbeddings

from llama_index.core.node_parser import SemanticSplitterNodeParser, SemanticDoubleMergingSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

CHUNKING_STRATEGIES = ['semantic', 'fixed', 'sentences', 'semantic_double_merging']


class Chunker:
    def __init__(
        self,
        chunking_strategy: str,
    ):
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError("Unsupported chunking strategy: ", chunking_strategy)
        self.chunking_strategy = chunking_strategy
        self.embed_model = None
        self.embedding_model_name = None

    def _setup_semantic_chunking(self, embedding_model_name):
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name

        # self.embed_model = HuggingFaceEmbedding(
        #     model_name=self.embedding_model_name,
        #     trust_remote_code=True,
        #     embed_batch_size=1,
        #     device='cuda' if torch.cuda.is_available() else 'cpu',
        # )
        # self.splitter = SemanticSplitterNodeParser(
        #     embed_model=self.embed_model,
        #     show_progress=True,
        # )
        self.embed_model = SentenceTransformer(
            self.embedding_model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float32},  # or torch.float32
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        model = SentenceTransformerEmbeddings(self.embed_model)
        self.splitter = SemanticChunker(
            embedding_model=model,
            threshold=0.5,
            chunk_size=256,
            min_sentences=1
        )
    
    def _setup_semantic_double_merging_chunking(self, embedding_model_name):
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name

        # self.embed_model = HuggingFaceEmbedding(
        #     model_name=self.embedding_model_name,
        #     trust_remote_code=True,
        #     embed_batch_size=1,
        #     device='cuda' if torch.cuda.is_available() else 'cpu',
        # )
        # self.splitter = SemanticDoubleMergingSplitterNodeParser(
        #     embed_model=self.embed_model,
        #     show_progress=True,
        # )
        self.embed_model = SentenceTransformer(
            self.embedding_model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float32},  # or torch.float32
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        model = SentenceTransformerEmbeddings(self.embed_model)
        self.splitter = SDPMChunker(
            embedding_model=model,
            threshold=0.5,
            chunk_size=256,
            min_sentences=1,
            skip_window=1
        )

    def chunk_semantically(
        self,
        text: str,
        tokenizer: 'AutoTokenizer',
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        if self.embed_model is None:
            self._setup_semantic_chunking(embedding_model_name)
        """
        # Get semantic nodes
        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=True
            )
        ]

        # Tokenize the entire text
        tokens = tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []

        for char_start, char_end in nodes:
            # Convert char indices to token indices
            start_chunk_index = bisect.bisect_left(
                [offset[0] for offset in token_offsets], char_start
            )
            end_chunk_index = bisect.bisect_right(
                [offset[1] for offset in token_offsets], char_end
            )

            # Add the chunk span if it's within the tokenized text
            if start_chunk_index < len(token_offsets) and end_chunk_index <= len(
                token_offsets
            ):
                chunk_spans.append((start_chunk_index, end_chunk_index))
            else:
                break
        """
        chunk_spans = []
        token_offsets = 0
        # Get semantic nodes
        chunks = self.splitter.chunk(text)
        for chunk in chunks:
            tokens = tokenizer.encode_plus(chunk.text, return_offsets_mapping=True, add_special_tokens=False)
            l = len(tokens.encodings[0])
            chunk_spans.append((token_offsets, token_offsets + l))
            token_offsets += l
        return chunk_spans

    def chunk_by_semantic_double_merging(
        self,
        text: str,
        tokenizer: 'AutoTokenizer',
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        if self.embed_model is None:
            self._setup_semantic_double_merging_chunking(embedding_model_name)

        chunk_spans = []
        token_offsets = 0
        # Get semantic nodes
        chunks = self.splitter.chunk(text)
        for chunk in chunks:
            tokens = tokenizer.encode_plus(chunk.text, return_offsets_mapping=True, add_special_tokens=False)
            l = len(tokens.encodings[0])
            chunk_spans.append((token_offsets, token_offsets + l))
            token_offsets += l
        return chunk_spans
    
    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: int,
        tokenizer: 'AutoTokenizer',
    ) -> List[Tuple[int, int, int]]:
        tokens = tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []
        for i in range(0, len(token_offsets), chunk_size):
            chunk_end = min(i + chunk_size, len(token_offsets))
            if chunk_end - i > 0:
                chunk_spans.append((i, chunk_end))

        return chunk_spans

    def chunk_by_sentences(
        self,
        text: str,
        n_sentences: int,
        tokenizer: 'AutoTokenizer',
    ) -> List[Tuple[int, int, int]]:
        tokens = tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []
        chunk_start = 0
        count_chunks = 0
        for i in range(0, len(token_offsets)):
            if tokens.tokens(0)[i] in ('.', '!', '?') and (
                (len(tokens.tokens(0)) == i + 1)
                or (tokens.token_to_chars(i).end != tokens.token_to_chars(i + 1).start)
                or True
            ):
                count_chunks += 1
                if count_chunks == n_sentences:
                    chunk_spans.append((chunk_start, i + 1))
                    chunk_start = i + 1
                    count_chunks = 0
        if len(tokens.tokens(0)) - chunk_start > 1:
            chunk_spans.append((chunk_start, len(tokens.tokens(0))))
        return chunk_spans

    def chunk(
        self,
        text: str,
        tokenizer: 'AutoTokenizer',
        chunking_strategy: str = None,
        chunk_size: Optional[int] = None,
        n_sentences: Optional[int] = None,
        embedding_model_name: Optional[str] = None,
    ):
        chunking_strategy = chunking_strategy or self.chunking_strategy
        if chunking_strategy == "semantic":
            return self.chunk_semantically(
                text,
                embedding_model_name=embedding_model_name,
                tokenizer=tokenizer,
            )
        elif chunking_strategy == "fixed":
            if chunk_size < 4:
                raise ValueError("Chunk size must be >= 4.")
            return self.chunk_by_tokens(text, chunk_size, tokenizer)
        elif chunking_strategy == "sentences":
            return self.chunk_by_sentences(text, n_sentences, tokenizer)
        elif chunking_strategy == "semantic_double_merging":
            return self.chunk_by_semantic_double_merging(
                text,
                embedding_model_name=embedding_model_name,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError("Unsupported chunking strategy")
