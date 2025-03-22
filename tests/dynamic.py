import numpy as np
import torch
from chonkie import SentenceChunker

def _dynamic_chunking(output_embs, chunk_annotations, threshold=0.5):
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
                    span2.prev = None
                    del span2
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
                    span2.prev = None
                    del span2
                    return
                span1.end = best_mid - 1
                span2.start = best_mid
                if span1.prev.start != -1:
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
                recursive_merge(tail, new_span)
                if tail.next is not None:
                    tail = tail.next
                else:
                    tail = tail
        
        # Collect the merged spans
        pooled_embeddings = []
        span = root.next
        while span is not None:
            print(f"Span: {span.start} - {span.end}")
            print(f"text is: {", ".join(chunks[span.start:span.end + 1])}")
            # Compute mean pooled embedding for this span (inclusive of both start and end indices)
            pooled_embed = np.mean(output_embs[span.start:span.end + 1], axis=0)
            pooled_embeddings.append(pooled_embed)
            span = span.next
        return pooled_embeddings

from transformers import AutoModel
from transformers import AutoTokenizer

from chunked_pooling import chunked_pooling, chunk_by_sentences

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load model and tokenizer
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True, device_map=device)
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True)

# input_text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."

input_text = """Adapted from "The Colors of Animals" by Sir John Lubbock in A Book of Natural History (1902, ed. David Starr Jordan)
The color of animals is by no means a matter of chance; it depends on many considerations, but in the majority of cases tends to protect the animal from danger by rendering it less conspicuous. Perhaps it may be said that if coloring is mainly protective, there ought to be but few brightly colored animals. There are, however, not a few cases in which vivid colors are themselves protective. The kingfisher itself, though so brightly colored, is by no means easy to see. The blue harmonizes with the water, and the bird as it darts along the stream looks almost like a flash of sunlight.
Desert animals are generally the color of the desert. Thus, for instance, the lion, the antelope, and the wild donkey are all sand-colored. “Indeed,” says Canon Tristram, “in the desert, where neither trees, brushwood, nor even undulation of the surface afford the slightest protection to its foes, a modification of color assimilated to that of the surrounding country is absolutely necessary. Hence, without exception, the upper plumage of every bird, and also the fur of all the smaller mammals and the skin of all the snakes and lizards, is of one uniform sand color.”
The next point is the color of the mature caterpillars, some of which are brown. This probably makes the caterpillar even more conspicuous among the green leaves than would otherwise be the case. Let us see, then, whether the habits of the insect will throw any light upon the riddle. What would you do if you were a big caterpillar? Why, like most other defenseless creatures, you would feed by night, and lie concealed by day. So do these caterpillars. When the morning light comes, they creep down the stem of the food plant, and lie concealed among the thick herbage and dry sticks and leaves, near the ground, and it is obvious that under such circumstances the brown color really becomes a protection. It might indeed be argued that the caterpillars, having become brown, concealed themselves on the ground, and that we were reversing the state of things. But this is not so, because, while we may say as a general rule that large caterpillars feed by night and lie concealed by day, it is by no means always the case that they are brown; some of them still retaining the green color. We may then conclude that the habit of concealing themselves by day came first, and that the brown color is a later adaptation."""

# determine chunks
chunker = SentenceChunker(chunk_size=16, delim=[".", "!", "?", "\n", ","])
chunks = chunker.chunk(input_text)
chunks = [chunk.text for chunk in chunks]
print('Chunks:\n" ' + '\n"'.join(chunks) + '"')
# encode chunks
embeddings_traditional_chunking = model.encode(chunks)
print(_dynamic_chunking(embeddings_traditional_chunking, chunks))