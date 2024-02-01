# print_gguf

print_gguf.py is a simple utility to parse the `header` & `tensor_infos` of GGUF file.

### How to use?

```python
python print_gguf.py {.gguf_file}
```

### Output Example)

```python
python print_gguf_info.py llama-v1-7b-q2k.gguf
magic                                   =  0x46554747
version                                 =  3
tensor_count                            =  291
metadata_kv_count                       =  16
general.architecture                    =  llama
general.name                            =  LLaMA
llama.context_length                    =  2048
llama.embedding_length                  =  4096
llama.block_count                       =  32
llama.feed_forward_length               =  11008
llama.rope.dimension_count              =  128
llama.attention.head_count              =  32
llama.attention.head_count_kv           =  32
llama.attention.layer_norm_rms_epsilon  =  9.999999974752427e-07
general.file_type                       =  10
tokenizer.ggml.model                    =  llama
tokenizer.ggml.tokens                   =  ['<unk>', '<s>', '</s>', '<0x00>', '<0x01>', '<...
tokenizer.ggml.scores                   =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0...
tokenizer.ggml.token_type               =  [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6...
general.quantization_version            =  2
name          =  token_embd.weight
n_dimensions  =  2
shape         =  [4096, 32000]
ggml_type     =  GGML_TYPE_Q2_K
offset        =  0
==================================================
name          =  output_norm.weight
n_dimensions  =  1
shape         =  [4096]
ggml_type     =  GGML_TYPE_FP32
offset        =  43008000
==================================================
name          =  output.weight
n_dimensions  =  2
shape         =  [4096, 32000]
ggml_type     =  GGML_TYPE_Q6_K
offset        =  43024384
==================================================
name          =  blk.0.attn_q.weight
n_dimensions  =  2
shape         =  [4096, 4096]
ggml_type     =  GGML_TYPE_Q2_K
offset        =  150544384
==================================================
```

*I used `llama-v1-7b-q2k.gguf` which is generated by  @[ikawrakow](https://huggingface.co/ikawrakow)
