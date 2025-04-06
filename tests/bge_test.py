from transformers import AutoModel
from sentence_transformers import SentenceTransformer

model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")
model = model.eval()
ls = model.encode(["Hello World"])
print(ls)


from transformers import AutoTokenizer, AutoModel
import torch

model_name = "BAAI/bge-m3"  # 假设模型名正确
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Hello, world!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)

# 获取最后一层的 token embeddings
last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
token_embeddings = last_hidden_state[0]  # 取出第一个句子的所有 token 向量
mean_embedding = token_embeddings.mean(dim=0)
# 或者获取所有层的 hidden states
all_hidden_states = outputs.hidden_states  # 包含各层的 hidden states