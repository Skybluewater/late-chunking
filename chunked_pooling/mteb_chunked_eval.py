import logging
from typing import Any, Optional

import numpy as np
import torch
from mteb.abstasks import AbsTask
from mteb.evaluation.evaluators import RetrievalEvaluator
from mteb.load_results.mteb_results import ScoresDict
from mteb.tasks import Retrieval
from tqdm import tqdm

from chunked_pooling import chunked_pooling
from chunked_pooling.chunking import Chunker

logger = logging.getLogger(__name__)


class AbsTaskChunkedRetrieval(AbsTask):
    def __init__(
        self,
        chunking_strategy: str = None,
        chunked_pooling_enabled: bool = False,
        tokenizer: Optional[Any] = None,
        prune_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        n_sentences: Optional[int] = None,
        model_has_instructions: bool = False,
        embedding_model_name: Optional[str] = None,  # for semantic chunking
        truncate_max_length: Optional[int] = 8192,
        long_late_chunking_embed_size: Optional[int] = 0,
        long_late_chunking_overlap_size: Optional[int] = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            self.retrieval_task = getattr(
                Retrieval,
                self.metadata_dict['dataset'].get('name', None)
                or self.metadata_dict.get('name'),
            )()
        except:
            logger.warning('Could not initialize retrieval_task')
        self.chunking_strategy = chunking_strategy
        self.chunker = Chunker(self.chunking_strategy)
        self.chunked_pooling_enabled = chunked_pooling_enabled
        self.tokenizer = tokenizer
        self.prune_size = prune_size
        self.model_has_instructions = model_has_instructions
        self.chunking_args = {
            'chunk_size': chunk_size,
            'n_sentences': n_sentences,
            'embedding_model_name': embedding_model_name,
        }
        self.truncate_max_length = (
            truncate_max_length if truncate_max_length is not None and truncate_max_length > 0 else None
        )

        self.long_late_chunking_embed_size = long_late_chunking_embed_size
        self.long_late_chunking_overlap_size = long_late_chunking_overlap_size

    def load_data(self, **kwargs):
        self.retrieval_task.load_data(**kwargs)
        self.corpus = self.retrieval_task.corpus
        self.queries = self.retrieval_task.queries
        self.relevant_docs = self.retrieval_task.relevant_docs
        # prune dataset
        if self.prune_size:
            self.queries, self.corpus, self.relevant_docs = self._prune(
                self.queries, self.corpus, self.relevant_docs, self.prune_size
            )

    def calculate_metadata_metrics(self):
        self.retrieval_task.calculate_metadata_metrics()

    def evaluate(
        self, model, split: str = "test", encode_kwargs: dict[str, Any] = {}, **kwargs
    ) -> dict[str, ScoresDict]:
        scores: dict[str, ScoresDict] = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )

            scores[hf_subset] = self._evaluate_monolingual(
                model,
                corpus,
                queries,
                relevant_docs,
                hf_subset,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )

        return scores

    def _truncate_documents(self, corpus):
        for k, v in corpus.items():
            title_tokens = 0
            if 'title' in v:
                tokens = self.tokenizer(
                    v['title'] + ' ',
                    return_offsets_mapping=True,
                    max_length=self.truncate_max_length,
                )
                title_tokens = len(tokens.input_ids)
            tokens = self.tokenizer(
                v['text'],
                return_offsets_mapping=True,
                max_length=self.truncate_max_length - title_tokens,
            )
            last_token_span = tokens.offset_mapping[-2]
            v['text'] = v['text'][: last_token_span[1]]
        return corpus

    def _embed_with_overlap(self, model, model_inputs):
        len_tokens = len(model_inputs["input_ids"][0])

        if len_tokens > self.long_late_chunking_embed_size:
            indices = []
            for i in range(
                0,
                len_tokens,
                self.long_late_chunking_embed_size
                - self.long_late_chunking_overlap_size,
            ):
                start = i
                end = min(i + self.long_late_chunking_embed_size, len_tokens)
                indices.append((start, end))
        else:
            indices = [(0, len_tokens)]

        outputs = []
        for start, end in indices:
            batch_inputs = {k: v[:, start:end] for k, v in model_inputs.items()}

            with torch.no_grad():
                model_output = model(**batch_inputs)

            if start > 0:
                outputs.append(
                    model_output[0][:, self.long_late_chunking_overlap_size :]
                )
            else:
                outputs.append(model_output[0])

        return torch.cat(outputs, dim=1).to(model.device)

    def _dynamic_chunking(self, output_embs, chunk_annotations, threshold=0.5):
        """Dynamic chunking based on the similarity of the embeddings.
        This method is not implemented yet.
        """
        class Span:
            def __init__(self, start, end, prev=None, next=None):
                self.start = start
                self.end = end
                self.prev = prev
                self.next = next
        
        def cal_similarity(embeddings: np.ndarray) -> np.ndarray:
            # Normalize embeddings to unit length to compute cosine similarity.
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm_embeddings = embeddings / (norms + 1e-8)
            sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
            n = sim_matrix.shape[0]
            indices = np.arange(n)
            # Compute the squared difference matrix: element (i,j) = (|i-j|)^2
            denom = (np.abs(indices.reshape(-1, 1) - indices.reshape(1, -1)))**2
            # denom = (np.abs(indices.reshape(-1, 1) - indices.reshape(1, -1)))
            # Avoid division by zero on the diagonal (set divisor to 1 for diagonal elements)
            denom[denom == 0] = 1
            return sim_matrix / denom
        
        def cal_inside_sim(start, mid, end):
            # span from [start, mid - 1], [mid, end]
            # Calculate similarity sum for indices [start, mid] (upper triangle, excluding diagonal)
            sim_start_mid = np.triu(sim_matrix[start:mid, start:mid], k=0).sum()
            # Calculate similarity sum for indices [mid+1, end] (upper triangle, excluding diagonal)
            sim_mid_end = np.triu(sim_matrix[mid:end+1, mid:end+1], k=0).sum()
            return sim_start_mid, sim_mid_end
        
        def cal_inside_coefficient(start, mid, end):
            cnt = (mid - start - 1) * (mid - start) / 2
            cnt += (end - mid) * (end - mid + 1) / 2
            return cnt
        
        def cal_outside_sim(start, mid, end):
            if start == mid:
                return 0
            sim_outside = sim_matrix[start:mid, mid:end+1].sum()
            return sim_outside
        
        def cal_outside_coefficient(start, mid, end):
            if start == mid:
                return 1
            return (mid - start) * (end - mid + 1)
        
        def recursive_merge(span1, span2):
            nonlocal tail
            start1 = span1.start
            end1 = span1.end
            start2 = span2.start
            end2 = span2.end
            ls = []
            if start1 == end1 and start2 == end2:
                if sim_matrix[start1, start2] > threshold:
                    span1.start = start1
                    span1.end = end2
                    span1.next = span2.next
                    if span2.next is not None:
                        span2.next.prev = span1
                    span2.next = None
                    span2.prev = None
                    del span2
                    if span1.next is None:
                        tail = span1
                    if span1.prev is not root:
                        recursive_merge(span1.prev, span1)
                return
            
            for mid in range(min(start1, start2), max(end1, end2) + 1):
                sim_inside = sum(cal_inside_sim(start1, mid, end2)) / cal_inside_coefficient(start1, mid, end2)
                sim_outside = cal_outside_sim(start1, mid, end2) / cal_outside_coefficient(start1, mid, end2)
                ls.append([sim_inside - sim_outside, mid, sim_inside, sim_outside])
            
            ls = sorted(ls, key=lambda x: x[0], reverse=True)
            best_mid = ls[0][1]
            if best_mid == start2:
                return
            else:
                if best_mid == start1:
                    span1.start = start1
                    span1.end = end2
                    span1.next = span2.next
                    if span2.next is not None:
                        span2.next.prev = span1
                    span2.prev = None
                    span2.next = None
                    del span2
                    # update tail again for the original tail may be deleted
                    if span1.next is None:
                        tail = span1
                    if span1.prev is not root:
                        recursive_merge(span1.prev, span1)
                    return
                
                span1.end = best_mid - 1
                span2.start = best_mid
                if span1.prev is not root:
                    recursive_merge(span1.prev, span1)
            
        sim_matrix = cal_similarity(output_embs)
        np.fill_diagonal(sim_matrix, 0)
        root = Span(-1, -1)
        tail = root
        for idx, chunk in enumerate(output_embs):
            if idx == 0:
                root.next = Span(0, 0, root, None)
                tail = root.next
            else:
                new_span = Span(idx, idx, tail, None)
                tail.next = new_span
                tail = new_span
                recursive_merge(tail.prev, tail)
        
        # Collect the merged spans
        pooled_embeddings = []
        span = root.next
        while span is not None:
            print(f"Span: {span.start} - {span.end}")
            # print(f"text is: {", ".join(chunks[span.start:span.end + 1])}")
            # Compute mean pooled embedding for this span (inclusive of both start and end indices)
            pooled_embed = np.mean(output_embs[span.start:span.end + 1], axis=0)
            pooled_embeddings.append(pooled_embed)
            span = span.next
        return pooled_embeddings
    
    def _use_chunking(
            self,
            model,
            corpus,
            queries,
            relevant_docs,
            lang=None,
            batch_size=1,
            encode_kwargs=None,
            **kwargs,
        ):
        corpus = self._apply_chunking(corpus, self.tokenizer)
        # {document_id: [{text: text_chunk}, ...]}
        query_ids = list(queries.keys())
        query_texts = [queries[k] for k in query_ids]
        if hasattr(model, 'encode_queries'):
            query_embs = model.encode_queries(query_texts)
        else:
            query_embs = model.encode(query_texts)

        corpus_ids = list(corpus.keys())
        corpus = [corpus[k] for k in corpus_ids]
        corpus_embs = []
        # corpus = self._flatten_chunks(corpus)
        with torch.no_grad():
            # Assume that corpus_ids is defined earlier as:
            # corpus_ids = list(sorted(corpus.keys()))
            corpus_ids_list = corpus_ids[:]  # make a copy of the corpus_ids list
            doc_counter = 0
            for inputs in tqdm(
                self._batch_inputs(
                    corpus,
                    batch_size=batch_size,
                ),
                total=(len(corpus) // batch_size),
            ):
                # Process each document in the batch along with its corpus_id.
                for input, doc_id in zip(inputs, corpus_ids_list[doc_counter : doc_counter + len(inputs)]):
                    print("Processing Corpus ID:", doc_id)
                    # Process the current document's chunks.
                    if self.model_has_instructions:
                        instr = model.get_instructions()[1]
                        instr_tokens = self.tokenizer(instr, add_special_tokens=False)
                        n_instruction_tokens = len(instr_tokens[0])
                    else:
                        instr = ''
                        n_instruction_tokens = 0

                    chunk_spans = []
                    for text in input:
                        text = text['text']
                        tokens = self.tokenizer.encode_plus(
                            text, return_offsets_mapping=True, add_special_tokens=False
                        )
                        token_offsets = tokens.offset_mapping
                        chunk_spans.append([0, len(token_offsets)])

                    chunk_spans = [self._extend_special_tokens(
                        [chunk_span],
                        n_instruction_tokens=n_instruction_tokens,
                    ) for chunk_span in chunk_spans]

                    text_inputs = [instr + x['text'] for x in input]

                    tokenized_texts = [
                        self.tokenizer(
                            text,
                            return_tensors='pt',
                            padding=True,
                            truncation=self.truncate_max_length is not None,
                            max_length=self.truncate_max_length,
                        )
                        for text in text_inputs
                    ]

                    model_inputs = [
                        {k: v.to(model.device) for k, v in x.items()}
                        for x in tokenized_texts
                    ]

                    if model.device.type == 'cuda':
                        model_inputs = [{
                            k: v.to(model.device) for k, v in inp.items()
                        } for inp in model_inputs]

                    model_outputs = [model(**model_input) for model_input in model_inputs]

                    output_embs = []
                    for output, chunk_span in zip(model_outputs, chunk_spans):
                        token_embeddings = output[0][0]
                        output_emb = token_embeddings[chunk_span[0][0]: chunk_span[0][1]]
                        output_emb = torch.mean(output_emb, dim=0, keepdim=True)
                        output_emb = output_emb.float().detach().cpu().numpy()
                        output_embs.append(output_emb[0])

                    output_embs = self._dynamic_chunking(
                        output_embs, None, 0.5
                    )
                    corpus_embs.extend([output_embs])
                doc_counter += len(inputs)
        
        max_chunks = max([len(x) for x in corpus_embs])
        k_values = self._calculate_k_values(max_chunks)
        # determine the maximum number of documents to consider in a ranking
        max_k = int(max(k_values) / max_chunks)
        (
            chunk_id_list,
            doc_to_chunk,
            flattened_corpus_embs,
        ) = self.flatten_corpus_embs(corpus_embs, corpus_ids)
        similarity_matrix = np.dot(query_embs, flattened_corpus_embs.T)
        results = self.get_results(
            chunk_id_list, k_values, query_ids, similarity_matrix
        )
        return results, max_k, k_values

    
    def _evaluate_monolingual(
        self,
        model,
        corpus,
        queries,
        relevant_docs,
        lang=None,
        batch_size=1,
        encode_kwargs=None,
        **kwargs,
    ):
        if self.truncate_max_length:
            corpus = self._truncate_documents(corpus)
        # split corpus into chunks
        if not self.chunked_pooling_enabled:
            results, max_k, k_values = self._use_chunking(
                model,
                corpus,
                queries,
                relevant_docs,
                lang=lang,
                batch_size=batch_size,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
        else:
            query_ids = list(queries.keys())
            query_texts = [queries[k] for k in query_ids]
            if hasattr(model, 'encode_queries'):
                query_embs = model.encode_queries(query_texts)
            else:
                query_embs = model.encode(query_texts)

            corpus_ids = list(corpus.keys())
            corpus_texts = [
                (
                    f"{corpus[k]['title']} {corpus[k]['text']}"
                    if 'title' in corpus[k]
                    else corpus[k]['text']
                )
                for k in corpus_ids
            ]

            chunk_annotations = self._calculate_annotations(model, corpus_texts)

            corpus_embs = []
            with torch.no_grad():
                for inputs in tqdm(
                    self._batch_inputs(
                        list(zip(corpus_texts, chunk_annotations)),
                        batch_size=batch_size,
                    ),
                    total=(len(corpus_texts) // batch_size),
                ):
                    if self.model_has_instructions:
                        instr = model.get_instructions()[1]
                    else:
                        instr = ''
                    text_inputs = [instr + x[0] for x in inputs]
                    annotations = [x[1] for x in inputs]
                    model_inputs = self.tokenizer(
                        text_inputs,
                        return_tensors='pt',
                        padding=True,
                        truncation=self.truncate_max_length is not None,
                        max_length=self.truncate_max_length,
                    )
                    if model.device.type == 'cuda':
                        model_inputs = {
                            k: v.to(model.device) for k, v in model_inputs.items()
                        }

                    if self.long_late_chunking_embed_size > 0:
                        model_outputs = self._embed_with_overlap(model, model_inputs)
                        output_embs = chunked_pooling(
                            [model_outputs], annotations, max_length=None
                        )
                    else:  # truncation
                        model_outputs = model(**model_inputs)
                        output_embs = chunked_pooling(
                            model_outputs,
                            annotations,
                            max_length=self.truncate_max_length,
                        )
                    output_embs = [self._dynamic_chunking(
                        output_emb, annotations, 0.5
                    ) for output_emb in output_embs]
                    corpus_embs.extend(output_embs)

            max_chunks = max([len(x) for x in corpus_embs])
            k_values = self._calculate_k_values(max_chunks)
            # determine the maximum number of documents to consider in a ranking
            max_k = int(max(k_values) / max_chunks)
            (
                chunk_id_list,
                doc_to_chunk,
                flattened_corpus_embs,
            ) = self.flatten_corpus_embs(corpus_embs, corpus_ids)
            similarity_matrix = np.dot(query_embs, flattened_corpus_embs.T)
            results = self.get_results(
                chunk_id_list, k_values, query_ids, similarity_matrix
            )
        # results 是每一个回答对应的前 1000 条 chunk 的结果
        doc_results = self.get_doc_results(results)

        ndcg, _map, recall, precision, _ = RetrievalEvaluator.evaluate(
            relevant_docs,
            doc_results,
            [k for k in k_values if k <= max_k],
            ignore_identical_ids=kwargs.get('ignore_identical_ids', True),
        )
        mrr, _ = RetrievalEvaluator.evaluate_custom(
            relevant_docs,
            doc_results,
            [k for k in k_values if k <= max_k],
            'mrr',
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def get_results(self, chunk_id_list, k_values, query_ids, similarity_matrix):
        results = {}
        for i, query_id in enumerate(query_ids):
            query_results = {}
            for idx, score in enumerate(similarity_matrix[i]):
                chunk_id = chunk_id_list[idx]
                query_results[chunk_id] = score
            # Sort results by score and only keep the top k scores
            sorted_query_results = dict(
                sorted(query_results.items(), key=lambda item: item[1], reverse=True)[
                    : max(k_values)
                ]
            )
            results[query_id] = sorted_query_results
        return results

    def flatten_corpus_embs(self, corpus_embs, corpus_ids):
        doc_to_chunk = {}
        flattened_corpus_embs = []
        chunk_id_list = []
        for doc_id, emb in zip(corpus_ids, corpus_embs):
            for i, chunk in enumerate(emb):
                flattened_corpus_embs.append(chunk)
                doc_to_chunk[f"{doc_id}~{i}"] = doc_id
                chunk_id_list.append(f"{doc_id}~{i}")
        flattened_corpus_embs = np.vstack(flattened_corpus_embs)
        flattened_corpus_embs = self._normalize(flattened_corpus_embs)
        return chunk_id_list, doc_to_chunk, flattened_corpus_embs

    @staticmethod
    def get_doc_results(results):
        doc_results = dict()
        for q, result_chunks in results.items():
            docs = dict()
            for c_id, score in result_chunks.items():
                d_id = '~'.join(c_id.split('~')[:-1])
                if (d_id not in docs) or (score > docs[d_id]):
                    docs[d_id] = float(score)
            doc_results[q] = docs
        return doc_results

    def _calculate_k_values(self, max_chunks):
        k_values = [1, 3, 5, 10, 20]
        n = 2
        while 10**n < 100 * max_chunks:
            k_values.append(10**n)
            n += 1
        return k_values

    def _apply_chunking(self, corpus, tokenizer):
        chunked_corpus = dict()
        for k, v in corpus.items():
            text = f"{v['title']} {v['text']}" if 'title' in v else v['text']
            current_doc = []
            chunk_annotations = self.chunker.chunk(
                text,
                tokenizer,
                chunking_strategy=self.chunking_strategy,
                **self.chunking_args,
            )
            tokens = tokenizer.encode_plus(text, add_special_tokens=False)
            for start_token_idx, end_token_idx in chunk_annotations:
                text_chunk = tokenizer.decode(
                    tokens.encodings[0].ids[start_token_idx:end_token_idx]
                )
                current_doc.append({'text': text_chunk})
            chunked_corpus[k] = current_doc
        return chunked_corpus

    def _calculate_annotations(self, model, corpus_texts):
        if self.model_has_instructions:
            instr = model.get_instructions()[1]
            instr_tokens = self.tokenizer(instr, add_special_tokens=False)
            n_instruction_tokens = len(instr_tokens[0])
        else:
            n_instruction_tokens = 0
        chunk_annotations = [
            self._extend_special_tokens(
                self.chunker.chunk(
                    text,
                    self.tokenizer,
                    chunking_strategy=self.chunking_strategy,
                    **self.chunking_args,
                ),
                n_instruction_tokens=n_instruction_tokens,
            )
            for text in corpus_texts
        ]
        return chunk_annotations

    @staticmethod
    def _flatten_chunks(chunked_corpus):
        flattened_corpus = dict()
        for k, li in chunked_corpus.items():
            for i, c in enumerate(li):
                flattened_corpus[f'{k}~{i}'] = c

        return flattened_corpus

    @staticmethod
    def _normalize(x):
        return x / np.linalg.norm(x, axis=1)[:, None]

    @staticmethod
    def _batch_inputs(li, batch_size):
        for i in range(0, len(li), batch_size):
            yield li[i : i + batch_size]

    @staticmethod
    def _extend_special_tokens(
        annotations, n_instruction_tokens=0, include_prefix=True, include_sep=True
    ):
        """Extends the spans because of additional special tokens, e.g. the CLS token
        which are not considered by the chunker.
        """
        new_annotations = []
        for i in range(len(annotations)):
            add_left_offset = 1 if (not include_prefix) or int(i > 0) else 0
            left_offset = 1 + n_instruction_tokens
            left = (
                annotations[i][0] + add_left_offset * left_offset
            )  # move everything by one for [CLS]

            add_sep = 1 if include_sep and ((i + 1) == len(annotations)) else 0
            right_offset = left_offset + add_sep
            right = (
                annotations[i][1] + right_offset
            )  # move everything by one for [CLS] and the last one for [SEP]

            new_annotations.append((left, right))
        return new_annotations

    @staticmethod
    def _prune(queries, corpus, relevant_docs, prune_size):
        new_queries = {'test': {}}
        new_corpus = {'test': {}}
        new_relevant_docs = {'test': {}}
        for i, key in enumerate(relevant_docs['test']):
            if i >= prune_size:
                break
            new_relevant_docs['test'][key] = relevant_docs['test'][key]
            for x in relevant_docs['test'][key]:
                new_corpus['test'][x] = corpus['test'][x]
            new_queries['test'][key] = queries['test'][key]
        return new_queries, new_corpus, new_relevant_docs

    def _calculate_metrics_from_split(*args, **kwargs):
        pass

    def _evaluate_subset(*args, **kwargs):
        pass
