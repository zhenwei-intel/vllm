# XPU - Intel® GPUs

## Validated Hardware

| Hardware |
| -------- |
| [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html) |

## Current Gaps on Intel XPU

The following items are currently limited or unsupported on Intel XPU:

### Feature Gaps

- **CUDA graph mode** is not supported on Intel XPU yet ([tracking issue](https://github.com/vllm-project/vllm/issues/26970)).
- **Flash Attention with `float32`** falls back to Triton Attention on XPU.
- **`bfloat16` on Intel Arc A770** is blocked due to known accuracy issues (use `float16` instead).
- **XPU graph capture** has additional limits in multi-GPU communication scenarios.

### Quantization Gaps on Intel GPU

From the quantization hardware matrix in
[`docs/features/quantization/README.md`](../../features/quantization/README.md),
the following are not supported on Intel GPU:

- Marlin (GPTQ/AWQ/FP8/FP4)
- INT8 (W8A8)
- FP8 (W8A8)
- bitsandbytes
- DeepSpeedFP
- GGUF

## Model Support Scope

vLLM currently publishes a **validated model list** for Intel XPU (below), but
does not maintain an exhaustive "unsupported model" deny list.

For Intel XPU, treat a model as **not supported / not yet validated** when any of the following is true:

- The model architecture or checkpoint is **not listed** in the validated tables below.
- The model depends on a quantization method listed above as unsupported on Intel GPU.
- The model only works with unsupported XPU feature combinations.

## Recommended Models

### Text-only Language Models

| Model                                     | Architecture                                         | FP16 | Dynamic FP8 | MXFP4 |
| ----------------------------------------- | ---------------------------------------------------- | ---- | ----------- | ----- |
| openai/gpt-oss-20b                        | GPTForCausalLM                                       |      |             | ✅    |
| openai/gpt-oss-120b                       | GPTForCausalLM                                       |      |             | ✅    |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B  | LlamaForCausalLM                                     | ✅   | ✅          |       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  | QwenForCausalLM                                      | ✅   | ✅          |       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | QwenForCausalLM                                      | ✅   | ✅          |       |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | LlamaForCausalLM                                     | ✅   | ✅          |       |
| Qwen/Qwen2.5-72B-Instruct                 | Qwen2ForCausalLM                                     | ✅   | ✅          |       |
| Qwen/Qwen3-14B                            | Qwen3ForCausalLM                                     | ✅   | ✅          |       |
| Qwen/Qwen3-32B                            | Qwen3ForCausalLM                                     | ✅   | ✅          |       |
| Qwen/Qwen3-30B-A3B                        | Qwen3ForCausalLM                                     | ✅   | ✅          |       |
| Qwen/Qwen3-30B-A3B-GPTQ-Int4              | Qwen3ForCausalLM                                     | ✅   | ✅          |       |
| Qwen/Qwen3-coder-30B-A3B-Instruct         | Qwen3ForCausalLM                                     | ✅   | ✅          |       |
| Qwen/QwQ-32B                              | QwenForCausalLM                                      | ✅   | ✅          |       |
| deepseek-ai/DeepSeek-V2-Lite              | DeepSeekForCausalLM                                  | ✅   | ✅          |       |
| meta-llama/Llama-3.1-8B-Instruct          | LlamaForCausalLM                                     | ✅   | ✅          |       |
| baichuan-inc/Baichuan2-13B-Chat           | BaichuanForCausalLM                                  | ✅   | ✅          |       |
| THUDM/GLM-4-9B-chat                       | GLMForCausalLM                                       | ✅   | ✅          |       |
| THUDM/CodeGeex4-All-9B                    | CodeGeexForCausalLM                                  | ✅   | ✅          |       |
| chuhac/TeleChat2-35B                      | LlamaForCausalLM (TeleChat2 based on Llama arch)     | ✅   | ✅          |       |
| 01-ai/Yi1.5-34B-Chat                      | YiForCausalLM                                        | ✅   | ✅          |       |
| THUDM/CodeGeex4-All-9B                    | CodeGeexForCausalLM                                  | ✅   | ✅          |       |
| deepseek-ai/DeepSeek-Coder-33B-base       | DeepSeekCoderForCausalLM                             | ✅   | ✅          |       |
| baichuan-inc/Baichuan2-13B-Chat           | BaichuanForCausalLM                                  | ✅   | ✅          |       |
| meta-llama/Llama-2-13b-chat-hf            | LlamaForCausalLM                                     | ✅   | ✅          |       |
| THUDM/CodeGeex4-All-9B                    | CodeGeexForCausalLM                                  | ✅   | ✅          |       |
| Qwen/Qwen1.5-14B-Chat                     | QwenForCausalLM                                      | ✅   | ✅          |       |
| Qwen/Qwen1.5-32B-Chat                     | QwenForCausalLM                                      | ✅   | ✅          |       |

### Multimodal Language Models

| Model                        | Architecture                     | FP16 | Dynamic FP8 | MXFP4 |
| ---------------------------- | -------------------------------- | ---- | ----------- | ----- |
| OpenGVLab/InternVL3_5-8B     | InternVLForConditionalGeneration | ✅   | ✅          |       |
| OpenGVLab/InternVL3_5-14B    | InternVLForConditionalGeneration | ✅   | ✅          |       |
| OpenGVLab/InternVL3_5-38B    | InternVLForConditionalGeneration | ✅   | ✅          |       |
| Qwen/Qwen2-VL-7B-Instruct    | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| Qwen/Qwen2.5-VL-72B-Instruct | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| Qwen/Qwen2.5-VL-32B-Instruct | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| THUDM/GLM-4v-9B              | GLM4vForConditionalGeneration    | ✅   | ✅          |       |
| openbmb/MiniCPM-V-4          | MiniCPMVForConditionalGeneration | ✅   | ✅          |       |

### Embedding and Reranker Language Models

| Model                   | Architecture                   | FP16 | Dynamic FP8 | MXFP4 |
| ----------------------- | ------------------------------ | ---- | ----------- | ----- |
| Qwen/Qwen3-Embedding-8B | Qwen3ForTextEmbedding          | ✅   | ✅          |       |
| Qwen/Qwen3-Reranker-8B  | Qwen3ForSequenceClassification | ✅   | ✅          |       |

✅ Runs and optimized.  
🟨 Runs and correct but not optimized to green yet.  
❌ Does not pass accuracy test or does not run.  
