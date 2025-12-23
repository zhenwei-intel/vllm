# ENV
8xB60 with infiband

```bash
# install aiperf
pip install aiperf==0.3.0

# then apply aiperf_v0.3.0.patch

```


# PD commands
```bash
# prefill
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export UCX_TLS=ib,rc,ze_copy

export ZE_AFFINITY_MASK=0,1
export model_name=ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4
export tp_size=2


VLLM_USE_V1=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5577 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm serve $model_name -tp $tp_size --host localhost --port 8101 --seed 42 --enforce-eager --dtype float16 --gpu-memory-utilization 0.9 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"xpu"}' --max-model-len 8192 --block-size 64 --no-enable-prefix-caching --kv-cache-dtype fp8


# prefill2
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export UCX_TLS=ib,rc,ze_copy

export ZE_AFFINITY_MASK=2,3
export model_name=ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4
export tp_size=2


VLLM_USE_V1=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5377 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm serve $model_name -tp $tp_size --host localhost --port 8102 --seed 42 --enforce-eager --dtype float16 --gpu-memory-utilization 0.9 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"xpu"}' --max-model-len 8192 --block-size 64 --no-enable-prefix-caching --kv-cache-dtype fp8

# decode
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export UCX_TLS=ib,rc,ze_copy

export ZE_AFFINITY_MASK=4,5
export model_name=ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4
export tp_size=2


VLLM_USE_V1=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5177 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm serve $model_name -tp $tp_size --host localhost --port 8201 --seed 42 --enforce-eager --dtype float16 --gpu-memory-utilization 0.9 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"xpu"}' --max-model-len 8192 --block-size 64 --no-enable-prefix-caching --kv-cache-dtype fp8

# proxy
python3 ../toy_proxy_server.py --prefiller-hosts localhost localhost --prefiller-port 7101 7102 --decoder-host localhost --decoder-port 7201 --host localhost --port 7300 &> proxy.log &


# aiperf
bash perf_aiperf.sh --prefill-tp 2 --prefill-dp 1 --decode-tp 2 --decode-dp 1 --mode disaggregated --artifacts-root-dir artifacts_2p1d_1500 --url localhost:7300 --isl 1500 --concurrency 1.2,1.4,1.6,1.8,2 --model ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4

```

