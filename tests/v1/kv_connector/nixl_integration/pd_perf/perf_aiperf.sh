#/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# bash perf_aiperf.sh --tp 4 --dp 1 --url localhost:8101 --artifacts-root-dir artifacts_nonpd_1500_rate_tp4 --isl 1500 --request-rate 0.1,0.2,0.3,0.4,0.5,0.7,1,1.2,1.4,1.6,1.8,2 --model ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4

# Default Values
model="Qwen/Qwen3-8B"
url="http://localhost:8100"
mode="aggregated"
artifacts_root_dir="artifacts_root"
deployment_kind="dynamo"
request_rate_list="0.1,0.2,0.3,0.4,0.5,0.7,1"

# Input Sequence Length (isl) 3000 and Output Sequence Length (osl) 150 are
# selected for chat use case. Note that for other use cases, the results and
# tuning would vary.
isl=3000
osl=150

tp=0
dp=0
prefill_tp=0
prefill_dp=0
decode_tp=0
decode_dp=0

print_help() {
  echo "Usage: $0 [OPTIONS]"
  echo
  echo "Options:"
  echo "  --tensor-parallelism, --tp <int>           Tensor parallelism (default: $tp)"
  echo "  --data-parallelism, --dp <int>             Data parallelism (default: $dp)"
  echo "  --prefill-tensor-parallelism, --prefill-tp <int>   Prefill tensor parallelism (default: $prefill_tp)"
  echo "  --prefill-data-parallelism, --prefill-dp <int>     Prefill data parallelism (default: $prefill_dp)"
  echo "  --decode-tensor-parallelism, --decode-tp <int>     Decode tensor parallelism (default: $decode_tp)"
  echo "  --decode-data-parallelism, --decode-dp <int>       Decode data parallelism (default: $decode_dp)"
  echo "  --model <model_id>                         Hugging Face model ID to benchmark (default: $model)"
  echo "  --input-sequence-length, --isl <int>       Input sequence length (default: $isl)"
  echo "  --output-sequence-length, --osl <int>      Output sequence length (default: $osl)"
  echo "  --url <http://host:port>                   Target URL for inference requests (default: $url)"
  echo "  --request-rate <list>                       Comma-separated request rates (default: $request_rate_list)"
  echo "  --mode <aggregated|disaggregated>          Serving mode (default: $mode)"
  echo "  --artifacts-root-dir <path>                Root directory to store benchmark results (default: $artifacts_root_dir)"
  echo "  --deployment-kind <type>                   Deployment tag used for pareto chart labels (default: $deployment_kind)"
  echo "  --help                                     Show this help message and exit"
  echo
  exit 0
}

# Parse command line arguments
# The defaults can be overridden by command line arguments.
while [[ $# -gt 0 ]]; do
  case $1 in
    --tensor-parallelism|--tp)
      tp="$2"
      shift 2
      ;;
    --data-parallelism|--dp)
      dp="$2"
      shift 2
      ;;
    --prefill-tensor-parallelism|--prefill-tp)
      prefill_tp="$2"
      shift 2
      ;;
    --prefill-data-parallelism|--prefill-dp)
      prefill_dp="$2"
      shift 2
      ;;
    --decode-tensor-parallelism|--decode-tp)
      decode_tp="$2"
      shift 2
      ;;
    --decode-data-parallelism|--decode-dp)
      decode_dp="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --input-sequence-length|--isl)
      isl="$2"
      shift 2
      ;;
    --output-sequence-length|--osl)
      osl="$2"
      shift 2
      ;;
    --url)
      url="$2"
      shift 2
      ;;
    --request-rate)
      request_rate_list="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --artifacts-root-dir)
      artifacts_root_dir="$2"
      shift 2
      ;;
    --deployment-kind)
      deployment_kind="$2"
      shift 2
      ;;
    --help)
      print_help
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

IFS=',' read -r -a request_rate_array <<< "$request_rate_list"

if [ "${mode}" == "aggregated" ]; then
  if [ "${tp}" == "0" ] && [ "${dp}" == "0" ]; then
    echo "--tensor-parallelism and --data-parallelism must be set for aggregated mode."
    exit 1
  fi
  echo "Starting benchmark for the deployment service with the following configuration:"
  echo "  - Tensor Parallelism: ${tp}"
  echo "  - Data Parallelism: ${dp}"
elif [ "${mode}" == "disaggregated" ]; then
  if [ "${prefill_tp}" == "0" ] && [ "${prefill_dp}" == "0" ] && [ "${decode_tp}" == "0" ] && [ "${decode_dp}" == "0" ]; then
    echo "--prefill-tensor-parallelism, --prefill-data-parallelism, --decode-tensor-parallelism and --decode-data-parallelism must be set for disaggregated mode."
    exit 1
  fi
  echo "Starting benchmark for the deployment service with the following configuration:"
  echo "  - Prefill Tensor Parallelism: ${prefill_tp}"
  echo "  - Prefill Data Parallelism: ${prefill_dp}"
  echo "  - Decode Tensor Parallelism: ${decode_tp}"
  echo "  - Decode Data Parallelism: ${decode_dp}"
else
  echo "Unknown mode: ${mode}. Only 'aggregated' and 'disaggregated' modes are supported."
  exit 1
fi

echo "--------------------------------"
echo "WARNING: This script does not validate tensor_parallelism=${tp} and data_parallelism=${dp} settings."
echo "         The user is responsible for ensuring these match the deployment configuration being benchmarked."
echo "         Incorrect settings may lead to misleading benchmark results."
echo "--------------------------------"


# Create artifacts root directory if it doesn't exist
if [ ! -d "${artifacts_root_dir}" ]; then
    mkdir -p "${artifacts_root_dir}"
fi

# Find the next available artifacts directory index
index=0
while [ -d "${artifacts_root_dir}/artifacts_${index}" ]; do
    index=$((index + 1))
done

# Create the new artifacts directory
artifact_dir="${artifacts_root_dir}/artifacts_${index}"
mkdir -p "${artifact_dir}"

# Print warning about existing artifacts directories
if [ $index -gt 0 ]; then
    echo "--------------------------------"
    echo "WARNING: Found ${index} existing artifacts directories:"
    for ((i=0; i<index; i++)); do
        if [ -f "${artifacts_root_dir}/artifacts_${i}/deployment_config.json" ]; then
            echo "artifacts_${i}:"
            cat "${artifacts_root_dir}/artifacts_${i}/deployment_config.json"
            echo "--------------------------------"
        fi
    done
    echo "Creating new artifacts directory: artifacts_${index}"
    echo "--------------------------------"
fi

echo "Running aiperf with:"
echo "Model: $model"
echo "ISL: $isl"
echo "OSL: $osl"
echo "Request rates: ${request_rate_array[@]}"

# Request rates to test
for request_rate in "${request_rate_array[@]}"; do
  echo "Run request rate: $request_rate"
  request_count=200
  warmup_request_count=10
  num_dataset_entries=300

  # NOTE: For Dynamo HTTP OpenAI frontend, use `nvext` for fields like
  # `ignore_eos` since they are not in the official OpenAI spec.
  aiperf profile \
    --model ${model} \
    --tokenizer ${model} \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url ${url} \
    --synthetic-input-tokens-mean ${isl} \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean ${osl} \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:${osl} \
    --extra-inputs min_tokens:${osl} \
    --extra-inputs ignore_eos:true \
    --request-rate ${request_rate} \
    --streaming \
    --request-count ${request_count} \
    --warmup-request-count ${warmup_request_count} \
    --num-dataset-entries ${num_dataset_entries} \
    --random-seed 100 \
    --artifact-dir ${artifact_dir} \
    --profile-export-prefix profile_req_${request_rate} \
    --ui simple \
    -v \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'

done

# The configuration is dumped to a JSON file which hold details of the OAI service
# being benchmarked.
deployment_config=$(cat << EOF
{
  "kind": "${deployment_kind}",
  "model": "${model}",
  "input_sequence_length": ${isl},
  "output_sequence_length": ${osl},
  "tensor_parallelism": ${tp},
  "data_parallelism": ${dp},
  "prefill_tensor_parallelism": ${prefill_tp},
  "prefill_data_parallelism": ${prefill_dp},
  "decode_tensor_parallelism": ${decode_tp},
  "decode_data_parallelism": ${decode_dp},
  "mode": "${mode}"
}
EOF
)

mkdir -p "${artifact_dir}"
if [ -f "${artifact_dir}/deployment_config.json" ]; then
  echo "Deployment configuration already exists. Overwriting..."
  rm -f "${artifact_dir}/deployment_config.json"
fi
echo "${deployment_config}" > "${artifact_dir}/deployment_config.json"

echo "Benchmarking Successful!!!"
