> **Note:** There is a separate document for [setting up PD](https://github.com/HabanaAI/vllm-fork/blob/habana_main/pd_xpyd/readme.md) on `habanamain`. This guide focuses on using the `deepseek_r1` branch on CT3 & CT4.  

## Environment Setup

1. Log in to CT4.
2. Set the proxy: `export https_proxy=http://child-igk.intel.com:912`
3. Navigate to the project directory and install dependencies:
    ```bash
    cd vllm-fork
    git checkout deepseek_r1
    pip install -v .
    ```
4. Install etcd: `sudo apt install etcd -y`
5. Install Mooncake Transfer Engine: `pip3 install mooncake-transfer-engine==0.3.0b3`

Repeat the same steps as above on CT3.

## Prepare the Following Scripts

You can find the scripts in the following path on CT4: `/home/jarvis3@user/lzw/vllm-fork/scrpts_for_ds_r1`
![image](https://github.com/user-attachments/assets/567073e6-1e3b-4635-b9af-64167cc9e768)

## Execution Steps

1. On **CT4**, run: `bash start_etcd_mc.sh`
2. On **CT4**, run: `bash start_prefill.sh`
3. On **CT3**, run: `bash start_decode.sh`
4. Wait for model initialization to complete on both CT4 and CT3.
5. On **CT4**, run: `bash start_proxy.sh`
6. On **CT4**, verify the output: `bash curl.sh`

## Commands to run baidu/tencent case

```bash
#baidu

python3 benchmarks/benchmark_serving.py --backend vllm --model /software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/ --dataset-name sonnet --request-rate inf --port 8868 --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --trust-remote-code --max-concurrency 256 --num-prompts 256 --ignore-eos --burstiness 1000 --dataset-path benchmarks/sonnet.txt

#tencent

python3 benchmarks/benchmark_serving.py --backend vllm --model /software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/ --dataset-name sonnet --request-rate inf --port 8868 --sonnet-input-len 13000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --trust-remote-code --max-concurrency 256 --num-prompts 256 --ignore-eos --burstiness 1000 --dataset-path benchmarks/sonnet.txt  
```