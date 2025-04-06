import click
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer
from typing import Any, Optional, Union

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model

DEFAULT_CHUNKING_STRATEGY = 'sentences'
DEFAULT_CHUNK_SIZE = 128
DEFAULT_N_SENTENCES = 1
BATCH_SIZE = 1
DEFAULT_LONG_LATE_CHUNKING_OVERLAP_SIZE = 256
DEFAULT_LONG_LATE_CHUNKING_EMBED_SIZE = 0  # set to 0 to disable long late chunking
DEFAULT_TRUNCATE_MAX_LENGTH = None
DEFAULT_PRUNE_SIZE = -1 # set to -1 to disable dataset pruning
DEFAULT_USE_DYNAMIC = True
DEFAULT_DIS_COMP = 3
DEFAULT_DIS_ALPHA = 1
DEFAULT_MAX_LENGTH = 256


@click.command()
@click.option(
    '--model-name',
    default='jinaai/jina-embeddings-v3',
    help='The name of the model to use.',
)
@click.option(
    '--model-weights',
    default=None,
    help='The path to the model weights to use, e.g. in case of finetuning.',
)
@click.option(
    '--strategy',
    default=DEFAULT_CHUNKING_STRATEGY,
    help='The chunking strategy to be applied.',
)
@click.option(
    '--task-name', default='SciFactChunked', help='The evaluation task to perform.'
)
@click.option(
    '--eval-split', default='test', help='The name of the evaluation split in the task.'
)
@click.option(
    '--chunking-model',
    default='jinaai/jina-embeddings-v3',
    required=False,
    help='The name of the model used for semantic chunking.',
)
@click.option(
    '--truncate-max-length',
    default=DEFAULT_TRUNCATE_MAX_LENGTH,
    type=int,
    help='Maximum number of tokens; by default, truncation to 8192 tokens. If None, Long Late Chunking algorithm should be enabled.',
)
@click.option(
    '--chunk-size',
    default=DEFAULT_CHUNK_SIZE,
    type=int,
    help='Number of tokens per chunk for fixed strategy.',
)
@click.option(
    '--n-sentences',
    default=DEFAULT_N_SENTENCES,
    type=int,
    help='Number of sentences per chunk for sentence strategy.',
)
@click.option(
    '--long-late-chunking-embed-size',
    default=DEFAULT_LONG_LATE_CHUNKING_EMBED_SIZE,
    type=int,
    help='Number of tokens per macro chunk used for long late chunking.',
)
@click.option(
    '--long-late-chunking-overlap-size',
    default=DEFAULT_LONG_LATE_CHUNKING_OVERLAP_SIZE,
    type=int,
    help='Token length of the embeddings that come before/after soft boundaries (i.e. overlapping embeddings). Above zero, overlap is used between neighbouring embeddings.',
)
@click.option(
    '--prune-size',
    default=DEFAULT_PRUNE_SIZE,
    type=int,
    help='Number of sentences per chunk for sentence strategy.',
)
@click.option(
    '--use-dynamic',
    default=DEFAULT_USE_DYNAMIC,
    type=bool,
    help='Number of sentences per chunk for sentence strategy.',
)
@click.option(
    '--dis-comp',
    default=DEFAULT_DIS_COMP,
    type=int,
    help='Number of sentences per chunk for sentence strategy.',
)
@click.option(
    '--dis-alpha',
    default=DEFAULT_DIS_ALPHA,
    type=float,
    help='Number of sentences per chunk for sentence strategy.',
)
@click.option(
    '--max-length',
    default=DEFAULT_MAX_LENGTH,
    type=int,
    help='Number of sentences per chunk for sentence strategy.',
)
def main(
    model_name,
    model_weights,
    strategy,
    task_name,
    eval_split,
    chunking_model,
    truncate_max_length,
    chunk_size,
    n_sentences,
    long_late_chunking_embed_size,
    long_late_chunking_overlap_size,
    prune_size,
    use_dynamic,
    dis_comp,
    dis_alpha,
    max_length,
):
    try:
        task_cls = globals()[task_name]
    except:
        raise ValueError(f'Unknown task name: {task_name}')
    
    if prune_size < 0:
        prune_size = None
    
    if truncate_max_length is not None and (long_late_chunking_embed_size > 0):
        truncate_max_length = None
        print(
            f'Truncation is disabled because Long Late Chunking algorithm is enabled.'
        )

    model, has_instructions = load_model(model_name, model_weights)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device="cuda")
    
    chunking_args = {
        'chunk_size': chunk_size,
        'n_sentences': n_sentences,
        'chunking_strategy': strategy,
        'model_has_instructions': has_instructions,
        'embedding_model_name': chunking_model if chunking_model else model_name,
    }

    chunking_args_semantic = {
        'chunk_size': chunk_size,
        'n_sentences': n_sentences,
        'chunking_strategy': 'semantic',
        'model_has_instructions': has_instructions,
        'embedding_model_name': chunking_model if chunking_model else model_name,
    }
    
    
    chunking_args_sentences = {
        'chunk_size': chunk_size,
        'n_sentences': n_sentences,
        'chunking_strategy': 'sentences',
        'model_has_instructions': has_instructions,
        'embedding_model_name': chunking_model if chunking_model else model_name,
    }

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Evaluate with late chunking
    tasks = [
        task_cls(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=dis_comp,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args,
        )
    ]

    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args,
    )
    evaluation.run(
        model,
        output_folder='BAAI-results-dynamic-0.8',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    tasks = [
        task_cls(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=dis_comp,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args,
        )
    ]

    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args,
    )
    evaluation.run(
        model,
        output_folder='BAAI-results-normal',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    return
    
    
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=use_dynamic,
    #         dis_comp=dis_comp,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-chunked-pooling-dis-0',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=use_dynamic,
    #         dis_comp=1,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-chunked-pooling-dis-1',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=use_dynamic,
    #         dis_comp=2,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-chunked-pooling-dis-2',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    
    # tasks_sci = [
    #     SciFactChunked(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=True,
    #         dis_comp=3,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]
    
    # evaluation = MTEB(
    #     tasks=tasks_sci,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='sci-chunked-dynamic-3',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    tasks = [
        SciFactChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='sci-semantic-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    
    tasks = [
        SciFactChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='sci-semantic-normal-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    tasks = [
        SciFactChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_sentences,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_sentences,
    )
    evaluation.run(
        model,
        output_folder='sci-sentences-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    #*********************************NF_CORPUS*************************************
    
    tasks = [
        NFCorpusChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='nf-semantic-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    
    tasks = [
        NFCorpusChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='nf-semantic-normal-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    tasks = [
        NFCorpusChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_sentences,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_sentences,
    )
    evaluation.run(
        model,
        output_folder='nf-sentences-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    # ***************************************FiQA*************************************
    
    tasks = [
        FiQA2018Chunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='Fi-semantic-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    
    tasks = [
        FiQA2018Chunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='Fi-semantic-normal-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    tasks = [
        FiQA2018Chunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_sentences,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_sentences,
    )
    evaluation.run(
        model,
        output_folder='Fi-sentences-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    # tasks_sci = [
    #     SciFactChunked(
    #         chunked_pooling_enabled=False,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=False,
    #         dis_comp=3,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args_sentences,
    #     )
    # ]
    
    # evaluation = MTEB(
    #     tasks=tasks_sci,
    #     chunked_pooling_enabled=False,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args_sentences,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='sci-sentences-normal-not-dynamic',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    # tasks_sci = [
    #     SciFactChunked(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=False,
    #         dis_comp=3,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]
    
    # evaluation = MTEB(
    #     tasks=tasks_sci,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='sci-chunked-not-dynamic',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    
    
    
    
    # tasks_nf = [
    #     NFCorpusChunked(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=True,
    #         dis_comp=3,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]
    
    # evaluation = MTEB(
    #     tasks=tasks_nf,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='nf-chunked-dynamic-3',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    # tasks_nf = [
    #     NFCorpusChunked(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=False,
    #         dis_comp=3,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]
    
    # evaluation = MTEB(
    #     tasks=tasks_nf,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='nf-chunked-not-dynamic',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    tasks_nf = [
        NFCorpusChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='nf-normal-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    
    tasks_nf = [
        NFCorpusChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='nf-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    
    tasks_nf = [
        FiQA2018Chunked(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='Fi-chunked-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    tasks_nf = [
        FiQA2018Chunked(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='Fi-chunked-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    tasks_nf = [
        FiQA2018Chunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='Fi-normal-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    
    
    tasks_nf = [
        FiQA2018Chunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='Fi-normal-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=True,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=use_dynamic,
    #         dis_comp=3,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=True,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-chunked-pooling-dis-3',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    
    
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=False,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=use_dynamic,
    #         dis_comp=dis_comp,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=False,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-normal-pooling-dis-0',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=False,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=use_dynamic,
    #         dis_comp=1,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=False,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-normal-pooling-dis-1',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=False,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         long_late_chunking_embed_size=long_late_chunking_embed_size,
    #         long_late_chunking_overlap_size=long_late_chunking_overlap_size,
    #         use_dynamic=use_dynamic,
    #         dis_comp=2,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=False,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-normal-pooling-dis-2',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )
    
    
    tasks = [
        task_cls(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=use_dynamic,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]

    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='results-normal-pooling-dis-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # Encode without late chunking
    # tasks = [
    #     task_cls(
    #         chunked_pooling_enabled=False,
    #         tokenizer=tokenizer,
    #         prune_size=prune_size,
    #         truncate_max_length=truncate_max_length,
    #         use_dynamic=use_dynamic,
    #         dis_comp=dis_comp,
    #         alpha=dis_alpha,
    #         max_length=max_length,
    #         **chunking_args,
    #     )
    # ]

    # evaluation = MTEB(
    #     tasks=tasks,
    #     chunked_pooling_enabled=False,
    #     tokenizer=tokenizer,
    #     prune_size=prune_size,
    #     **chunking_args,
    # )
    # evaluation.run(
    #     model,
    #     output_folder='results-normal-pooling',
    #     eval_splits=[eval_split],
    #     overwrite_results=True,
    #     batch_size=BATCH_SIZE,
    #     encode_kwargs={'batch_size': BATCH_SIZE},
    # )

    # === New evaluation implementations for NFCorpusChunked, FiQA2018Chunked, and TRECCOVID ===

    # NFCorpusChunked - Dynamic evaluation
    tasks_nf = [
        NFCorpusChunked(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='nf-chunked-dynamic-3-new',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # NFCorpusChunked - Non-dynamic evaluation
    tasks_nf = [
        NFCorpusChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_nf,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='nf-normal-not-dynamic-new',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # FiQA2018Chunked - Dynamic evaluation
    tasks_fiqa = [
        FiQA2018Chunked(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_fiqa,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='fi-chunked-dynamic-3-new',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # FiQA2018Chunked - Non-dynamic evaluation
    tasks_fiqa = [
        FiQA2018Chunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_fiqa,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='fi-normal-not-dynamic-new',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # TRECCOVID - assuming a new task class TrecCovidChunked exists

    # TRECCOVID - Dynamic evaluation
    tasks_trec = [
        TRECCOVIDChunked(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_trec,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='trec-chunked-dynamic-3',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # TRECCOVID - Non-dynamic evaluation
    tasks_trec = [
        TRECCOVIDChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_trec,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='trec-normal-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # TRECCOVID - Chunked-not-dynamic evaluation
    tasks_trec = [
        TRECCOVIDChunked(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=False,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_trec,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='trec-chunked-not-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # TRECCOVID - Normal-dynamic evaluation
    tasks_trec = [
        TRECCOVIDChunked(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=prune_size,
            truncate_max_length=truncate_max_length,
            long_late_chunking_embed_size=long_late_chunking_embed_size,
            long_late_chunking_overlap_size=long_late_chunking_overlap_size,
            use_dynamic=True,
            dis_comp=3,
            alpha=dis_alpha,
            max_length=max_length,
            **chunking_args_semantic,
        )
    ]
    evaluation = MTEB(
        tasks=tasks_trec,
        chunked_pooling_enabled=False,
        tokenizer=tokenizer,
        prune_size=prune_size,
        **chunking_args_semantic,
    )
    evaluation.run(
        model,
        output_folder='trec-normal-dynamic',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

if __name__ == '__main__':
    main()
