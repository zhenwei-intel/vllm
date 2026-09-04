#!/bin/bash
set -euo pipefail

PIPELINE_STAGE="${1:-${PIPELINE_STAGE:-build}}"
WORKSPACE_DIR="${WORKSPACE:-$(pwd)}"
REPO_ROOT="${WORKSPACE_DIR}"
DIST_DIR="${WORKSPACE_DIR}/dist"
ARTIFACT_DIR="${WORKSPACE_DIR}/.ci-artifacts"
LOG_DIR="${WORKSPACE_DIR}/logs/vllm-ci"
TEST_COMMAND_FILE="${WORKSPACE_DIR}/.jenkins_vllm_ci_test_command.sh"
TEST_SCOPE_FILE="${WORKSPACE_DIR}/.jenkins_vllm_ci_test_scope.log"
TEST_FRAMEWORK_VENV_DIR="${WORKSPACE_DIR}/.venvs/vllm-test-framework"
ORCHESTRATOR_DIR="${WORKSPACE_DIR}/.ci-support/orchestrator"
RUNTIME_TEST_IMAGE_NAME_FILE="${ARTIFACT_DIR}/runtime-test-image.name"
RUNTIME_TEST_IMAGE_TAR="${ARTIFACT_DIR}/runtime-test-image.tar"
RUNTIME_TEST_IMAGE_LOAD_LOG="${LOG_DIR}/load-runtime-test-image.log"

BASE_IMAGE="${BASE_IMAGE:-gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:PO_216}"
PREBUILT_KERNEL_WHL="${PREBUILT_KERNEL_WHL:-}"
RUN_TESTS="${RUN_TESTS:-false}"
COMMIT_RUNTIME_TEST_IMAGE="${COMMIT_RUNTIME_TEST_IMAGE:-false}"
PERSIST_RUNTIME_TEST_IMAGE="${PERSIST_RUNTIME_TEST_IMAGE:-false}"
RUN_TESTS_IN_BUILD_CONTAINER="${RUN_TESTS_IN_BUILD_CONTAINER:-false}"
MAX_JOBS="${MAX_JOBS:-}"
CCACHE_HOST_DIR="${CCACHE_HOST_DIR:-}"
VLLM_WHL_BUILD_COMMAND="${VLLM_WHL_BUILD_COMMAND:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-}"
TEST_SCOPE="${TEST_SCOPE:-${TEST_CASE_PARAM:-}}"
TEST_MODEL_CACHE="${TEST_MODEL_CACHE:-None}"
TEST_LOG_LEVEL="${TEST_LOG_LEVEL:-INFO}"
TEST_CLIENT_DOCKER_TIMEOUT="${TEST_CLIENT_DOCKER_TIMEOUT:-1800}"
ORCHESTRATOR_REPO="${ORCHESTRATOR_REPO:-https://github.com/intel-sandbox/ipex_cpu_nightly_github.git}"
ORCHESTRATOR_BRANCH="${ORCHESTRATOR_BRANCH:-multi-BMG-main_scriptRefactor_BAK-ze}"
VLLM_XPU_KERNEL_REPO="${VLLM_XPU_KERNEL_REPO:-https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu-kernels.git}"
VLLM_XPU_KERNEL_BRANCH="${VLLM_XPU_KERNEL_BRANCH:-main}"

mkdir -p "${DIST_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}"

log() {
    printf '[INFO] %s\n' "$*"
}

error() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

run_with_live_log() {
    local log_path="$1"
    shift

    mkdir -p "$(dirname "${log_path}")"
    : > "${log_path}"

    set +e
    "$@" 2>&1 | tee -a "${log_path}"
    local cmd_status=${PIPESTATUS[0]}
    set -e

    return "${cmd_status}"
}

default_vllm_version_override() {
    local commit_id=""
    local build_date=""

    commit_id="$(git -C "${REPO_ROOT}" rev-parse --short=10 HEAD)"
    build_date="$(date '+%Y%m%d')"
    printf '0.0.0+g%s.d%s.xpu' "${commit_id}" "${build_date}"
}

ensure_no_proxy_host() {
    local host="$1"
    [[ -n "${host}" ]] || return 0

    local env_name current_value updated_value
    for env_name in no_proxy NO_PROXY; do
        current_value="${!env_name:-}"
        case ",${current_value}," in
            *,${host},*)
                ;;
            *)
                updated_value="${current_value}"
                if [[ -n "${updated_value}" ]]; then
                    updated_value+="," 
                fi
                updated_value+="${host}"
                printf -v "${env_name}" '%s' "${updated_value}"
                export "${env_name}"
                ;;
        esac
    done
}

clone_repo() {
    local name="$1"
    local repo_url="$2"
    local repo_branch="$3"
    local dst_dir="$4"
    local clone_log="${LOG_DIR}/${name}-clone.log"

    rm -rf "${dst_dir}"
    if ! git clone --depth 1 --branch "${repo_branch}" --single-branch "${repo_url}" "${dst_dir}" >"${clone_log}" 2>&1; then
        cat "${clone_log}" >&2 || true
        error "Failed to clone ${name} from ${repo_url}#${repo_branch}"
    fi
}

framework_tests_requested() {
    [[ -n "${TEST_SCOPE}" ]]
}

commit_runtime_test_image_requested() {
    [[ "${COMMIT_RUNTIME_TEST_IMAGE}" == "true" ]]
}

persist_runtime_test_image_requested() {
    [[ "${PERSIST_RUNTIME_TEST_IMAGE}" == "true" ]]
}

direct_test_in_build_container_requested() {
    [[ "${RUN_TESTS_IN_BUILD_CONTAINER}" == "true" ]]
}

ensure_test_framework_host_env() {
    local python_bin="${TEST_FRAMEWORK_VENV_DIR}/bin/python"
    local prepare_log="${LOG_DIR}/prepare-test-framework-env.log"

    mkdir -p "$(dirname "${TEST_FRAMEWORK_VENV_DIR}")"
    if [[ ! -x "${python_bin}" ]]; then
        python3 -m venv "${TEST_FRAMEWORK_VENV_DIR}" >"${prepare_log}" 2>&1 || {
            cat "${prepare_log}" >&2 || true
            error "Failed to create test framework virtualenv at ${TEST_FRAMEWORK_VENV_DIR}"
        }
    fi

    if ! "${python_bin}" -c 'import docker, git' >/dev/null 2>&1; then
        "${python_bin}" -m pip install --upgrade pip setuptools wheel >"${prepare_log}" 2>&1 || {
            cat "${prepare_log}" >&2 || true
            error 'Failed to bootstrap test framework virtualenv'
        }
        "${python_bin}" -m pip install docker GitPython >>"${prepare_log}" 2>&1 || {
            cat "${prepare_log}" >&2 || true
            error 'Failed to install test framework host dependencies'
        }
    fi

    printf '%s' "${python_bin}"
}

prepare_framework_workspace() {
    clone_repo orchestrator "${ORCHESTRATOR_REPO}" "${ORCHESTRATOR_BRANCH}" "${ORCHESTRATOR_DIR}"
    clone_repo kernel_repo "${VLLM_XPU_KERNEL_REPO}" "${VLLM_XPU_KERNEL_BRANCH}" "${WORKSPACE_DIR}/vllm_xpu_kernel"

    rm -rf "${WORKSPACE_DIR}/vllm_test_framework" "${WORKSPACE_DIR}/vllm_scripts" \
        "${WORKSPACE_DIR}/vllm_benchmark.sh" "${WORKSPACE_DIR}/vllm_server_launch.sh"

    # Relative targets: the whole WORKSPACE_DIR is bind-mounted into the test
    # container at /workspace1, and an absolute host-path symlink target (e.g.
    # /home/.../workspace/.ci-support/orchestrator/...) does not exist inside that
    # container's filesystem, so it must resolve relative to the symlink's own dir.
    local orchestrator_relative_dir=".ci-support/orchestrator"
    ln -sfn "${orchestrator_relative_dir}/vllm_test_framework" "${WORKSPACE_DIR}/vllm_test_framework"
    ln -sfn "${orchestrator_relative_dir}/vllm_scripts" "${WORKSPACE_DIR}/vllm_scripts"
    ln -sfn "${orchestrator_relative_dir}/vllm_benchmark.sh" "${WORKSPACE_DIR}/vllm_benchmark.sh"
    ln -sfn "${orchestrator_relative_dir}/vllm_server_launch.sh" "${WORKSPACE_DIR}/vllm_server_launch.sh"
}

normalize_case_file_rows() {
    # Jenkins textarea params wrap long quoted CSV fields (e.g. extra_args) across
    # physical lines, and can also glue the last line of one "Specify,..." row
    # directly onto the next one with no newline in between; the framework's
    # case_generator.py reads cases with plain readlines(), so either case silently
    # merges two rows into one and corrupts both (see build #49 "UT case does not
    # contain a pytest invocation" failure, where an E2E row's extra_args ran into
    # the following UT row). Re-join wrapped lines until quotes balance, then split
    # back out any row boundaries that ended up glued together.
    local case_file="$1"
    python3 - "${case_file}" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    lines = f.readlines()

normalized = []
buffer = None
quote_count = 0

def flush():
    global buffer, quote_count
    if buffer is not None:
        normalized.append(buffer.rstrip('\n'))
    buffer = None
    quote_count = 0

for line in lines:
    if buffer is None:
        buffer = line
    else:
        buffer = buffer.rstrip('\n') + ' ' + line.lstrip()
    quote_count += line.count('"')
    if quote_count % 2 == 0:
        flush()
flush()

final = []
for line in normalized:
    for part in re.split(r'(?=Specify,|Category,)', line):
        if part.strip():
            final.append(part + '\n')

with open(path, 'w') as f:
    f.writelines(final)
PY
}

resolve_test_cases_file() {
    [[ -n "${TEST_SCOPE}" ]] || return 1

    if [[ -f "${TEST_SCOPE}" ]]; then
        printf '%s' "${TEST_SCOPE}"
        return 0
    fi
    if [[ -f "${WORKSPACE_DIR}/${TEST_SCOPE}" ]]; then
        printf '%s' "${WORKSPACE_DIR}/${TEST_SCOPE}"
        return 0
    fi
    if [[ -f "${ORCHESTRATOR_DIR}/${TEST_SCOPE}" ]]; then
        printf '%s' "${ORCHESTRATOR_DIR}/${TEST_SCOPE}"
        return 0
    fi

    printf '%s\n' "${TEST_SCOPE}" | tr -d '\r' > "${TEST_SCOPE_FILE}"
    normalize_case_file_rows "${TEST_SCOPE_FILE}"
    printf '%s' "${TEST_SCOPE_FILE}"
}

persist_runtime_test_image() {
    local image_name="$1"

    docker image inspect "${image_name}" >/dev/null 2>&1 || error "Runtime test image not found locally: ${image_name}"
    docker save -o "${RUNTIME_TEST_IMAGE_TAR}" "${image_name}" >"${LOG_DIR}/save-runtime-test-image.log" 2>&1 || {
        cat "${LOG_DIR}/save-runtime-test-image.log" >&2 || true
        error "Failed to save runtime test image ${image_name}"
    }
    printf '%s' "${image_name}" > "${RUNTIME_TEST_IMAGE_NAME_FILE}"
}

ensure_runtime_test_image_available() {
    local image_name=""

    [[ -f "${RUNTIME_TEST_IMAGE_NAME_FILE}" ]] || error "Missing runtime test image metadata: ${RUNTIME_TEST_IMAGE_NAME_FILE}"
    image_name="$(tr -d '\r' < "${RUNTIME_TEST_IMAGE_NAME_FILE}")"
    [[ -n "${image_name}" ]] || error "Runtime test image metadata is empty: ${RUNTIME_TEST_IMAGE_NAME_FILE}"

    if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
        [[ -f "${RUNTIME_TEST_IMAGE_TAR}" ]] || error "Missing runtime test image archive: ${RUNTIME_TEST_IMAGE_TAR}"
        if ! docker load -i "${RUNTIME_TEST_IMAGE_TAR}" >"${RUNTIME_TEST_IMAGE_LOAD_LOG}" 2>&1; then
            cat "${RUNTIME_TEST_IMAGE_LOAD_LOG}" >&2 || true
            error "Failed to load runtime test image ${image_name}"
        fi
    fi

    printf '%s' "${image_name}"
}

run_framework_tests() {
    local full_image="$1"
    local test_log="${LOG_DIR}/run-tests.log"
    local framework_python=""
    local cases_file=""
    local image_repo=""
    local image_tag=""

    prepare_framework_workspace
    cases_file="$(resolve_test_cases_file)"
    framework_python="$(ensure_test_framework_host_env)"
    image_repo="${full_image%:*}"
    image_tag="${full_image##*:}"

    log "Running framework tests from ${cases_file} against ${full_image}"
    if ! (
        cd "${WORKSPACE_DIR}/vllm_test_framework"
        "${framework_python}" main.py \
            --cases-file-path "${cases_file}" \
            --docker-repo "${image_repo}" \
            --docker-tag "${image_tag}" \
            --node-label "${TEST_NODE_LABEL:-${BUILD_NODE_LABEL:-xpu}}" \
            --workspace-path "${WORKSPACE_DIR}" \
            --jenkins-build-url "${BUILD_URL:-}" \
            --HF-TOKEN "${HF_TOKEN:-}" \
            --MODEL-CACHE "${TEST_MODEL_CACHE}" \
            --vllm-branch "${VLLM_BRANCH:-}" \
            --extra-ENV "${EXTRA_ENV:-}" \
            --log-level "${TEST_LOG_LEVEL}" \
            --client-docker-timeout "${TEST_CLIENT_DOCKER_TIMEOUT}" \
            --skip-db-update \
            --vllm-xpu-kernel-repo "${VLLM_XPU_KERNEL_REPO}" \
            --vllm-xpu-kernel-branch "${VLLM_XPU_KERNEL_BRANCH}" \
            --vllm-xpu-kernel-whl-url "${PREBUILT_KERNEL_WHL}" \
            --non-blocking-test-modes UT
    ) >"${test_log}" 2>&1; then
        tail -n 200 "${test_log}" >&2 || true
        error 'framework test stage failed'
    fi
}

append_proxy_env_args() {
    local -n docker_args_ref=$1
    local proxy_vars=(http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY)
    local var_name
    local var_value

    for var_name in "${proxy_vars[@]}"; do
        var_value="${!var_name:-}"
        if [[ -n "${var_value}" ]]; then
            docker_args_ref+=( -e "${var_name}=${var_value}" )
        fi
    done
}

resolve_prebuilt_kernel_wheel() {
    [[ -n "${PREBUILT_KERNEL_WHL}" ]] || error 'PREBUILT_KERNEL_WHL is required'

    local configured_source="${PREBUILT_KERNEL_WHL}"
    local destination_path=""
    local source_name=""
    local artifact_hostname=""

    if [[ "${configured_source}" =~ ^https?:// ]]; then
        source_name="${configured_source##*/}"
        [[ -n "${source_name}" ]] || source_name='vllm_xpu_kernels.whl'
        destination_path="${ARTIFACT_DIR}/${source_name}"
        artifact_hostname="$(python3 - <<'PY' "${configured_source}"
import sys
from urllib.parse import urlparse
print(urlparse(sys.argv[1]).hostname or '')
PY
)"
        ensure_no_proxy_host "${artifact_hostname}"
        if ! curl -L --fail --show-error --silent \
            --connect-timeout 30 \
            --retry 3 \
            --retry-delay 2 \
            --output "${destination_path}" \
            "${configured_source}"; then
            case "${artifact_hostname}" in
                *.intel.com)
                    log "Retrying kernel wheel download with curl --insecure for internal certificate chain on ${artifact_hostname}"
                    curl -L --fail --show-error --silent --insecure \
                        --connect-timeout 30 \
                        --retry 3 \
                        --retry-delay 2 \
                        --output "${destination_path}" \
                        "${configured_source}" || error "Failed to download prebuilt kernel wheel from ${configured_source}"
                    ;;
                *)
                    error "Failed to download prebuilt kernel wheel from ${configured_source}"
                    ;;
            esac
        fi
    else
        [[ -f "${configured_source}" ]] || error "Prebuilt kernel wheel not found: ${configured_source}"
        destination_path="${ARTIFACT_DIR}/$(basename "${configured_source}")"
        cp -f "${configured_source}" "${destination_path}"
    fi

    [[ -f "${destination_path}" ]] || error "Failed to materialize kernel wheel at ${destination_path}"
    printf '%s' "${destination_path}"
}

run_in_build_container() {
    local stage_name="$1"
    local kernel_wheel_host_path="$2"
    local stage_log="${LOG_DIR}/${stage_name}.log"
    local container_name="vllm-xpu-ci-${stage_name}-${BUILD_NUMBER:-$$}"
    local test_image_name="vllm-xpu-ci-runtime:${BUILD_NUMBER:-local}"
    local kernel_wheel_in_container="/workspace/artifacts/$(basename "${kernel_wheel_host_path}")"
    local workspace_uid="$(id -u)"
    local workspace_gid="$(id -g)"
    local version_override="${VLLM_VERSION_OVERRIDE:-$(default_vllm_version_override)}"
    local ccache_mount_args=()
    local docker_args=(
        --name "${container_name}"
        --shm-size=16g
        --network=host
        --ipc=host
        -v "${REPO_ROOT}:/workspace/vllm"
        -v "${DIST_DIR}:/workspace/dist"
        -v "${ARTIFACT_DIR}:/workspace/artifacts"
        -v "${LOG_DIR}:/workspace/logs"
        -e "PIP_BREAK_SYSTEM_PACKAGES=1"
        -e "MAX_JOBS=${MAX_JOBS}"
        -e "EXTRA_ENV=${EXTRA_ENV}"
        -e "VLLM_VERSION_OVERRIDE=${version_override}"
        -e "VLLM_WHL_BUILD_COMMAND=${VLLM_WHL_BUILD_COMMAND}"
        -e "KERNEL_WHEEL_IN_CONTAINER=${kernel_wheel_in_container}"
        -e "RUN_TESTS_IN_BUILD_CONTAINER=${RUN_TESTS_IN_BUILD_CONTAINER}"
    )

    if direct_test_in_build_container_requested; then
        [[ -f "${TEST_COMMAND_FILE}" ]] || error "Missing test command file for in-container test: ${TEST_COMMAND_FILE}"
        docker_args+=( -v "${TEST_COMMAND_FILE}:/workspace/run-test.sh:ro" )
    fi

    if [[ -n "${CCACHE_HOST_DIR}" ]]; then
        mkdir -p "${CCACHE_HOST_DIR}"
        ccache_mount_args+=( -v "${CCACHE_HOST_DIR}:/workspace/ccache" -e 'CCACHE_DIR=/workspace/ccache' )
    fi

    append_proxy_env_args docker_args
    docker_args+=( "${ccache_mount_args[@]}" )

    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    log "${stage_name} container create started: ${container_name} (base image: ${BASE_IMAGE})"
    if ! run_with_live_log "${stage_log}" docker create "${docker_args[@]}" "${BASE_IMAGE}" /bin/bash -lc '
set -euo pipefail
if [[ -f /opt/gfx-deps/env.sh ]]; then
    source /opt/gfx-deps/env.sh
fi
git config --global --add safe.directory /workspace/vllm
cd /workspace/vllm

mapfile -t torch_wheels < <(find /opt/gfx-deps/whl -maxdepth 1 -type f -name "torch-*.whl" | sort)
mapfile -t triton_wheels < <(find /opt/gfx-deps/whl -maxdepth 1 -type f \( -name "pytorch_triton_xpu*.whl" -o -name "triton-*.whl" \) | sort)

if [[ ${#torch_wheels[@]} -eq 0 ]]; then
    echo "[ERROR] No torch wheels found under /opt/gfx-deps/whl" >&2
    exit 1
fi
if [[ ${#triton_wheels[@]} -eq 0 ]]; then
    echo "[ERROR] No triton wheels found under /opt/gfx-deps/whl" >&2
    exit 1
fi

python3 -m pip install --no-deps --ignore-installed "${torch_wheels[@]}"
python3 -m pip install --no-deps --ignore-installed "${triton_wheels[@]}"
python3 -m pip install --no-deps --force-reinstall "${KERNEL_WHEEL_IN_CONTAINER}"

python3 use_existing_torch.py
sanitized_requirements=requirements/xpu.sanitized.ci.txt
sed "/extra-index-url/d;/torch/d;/torchaudio/d;/torchvision/d;/triton/d;/vllm_xpu_kernels/d" requirements/xpu.txt > "${sanitized_requirements}"
export CMAKE_PREFIX_PATH="$(python3 -c "import site; print(site.getsitepackages()[0])"):${CMAKE_PREFIX_PATH:-}"
python3 -m pip install --ignore-installed -r "${sanitized_requirements}"
python3 -m pip install --ignore-installed grpcio-tools protobuf nanobind accelerate hf_transfer pytest pytest_asyncio modelscope pyelftools

if [[ -n "${EXTRA_ENV}" ]]; then
    eval "export ${EXTRA_ENV}"
fi

if [[ -n "${VLLM_WHL_BUILD_COMMAND}" ]]; then
    eval "${VLLM_WHL_BUILD_COMMAND}"
else
    VLLM_TARGET_DEVICE=xpu python3 setup.py bdist_wheel --dist-dir=/workspace/dist --py-limited-api=cp38
fi

python3 -m pip install --no-deps --force-reinstall /workspace/dist/*.whl
python3 -m pip install --no-deps --force-reinstall "${torch_wheels[@]}"
python3 -m pip install --no-deps --force-reinstall "${triton_wheels[@]}"
python3 -m pip install --no-deps --force-reinstall "${KERNEL_WHEEL_IN_CONTAINER}"

if [[ -f /workspace/run-test.sh && "${RUN_TESTS_IN_BUILD_CONTAINER:-false}" == "true" ]]; then
    /bin/bash /workspace/run-test.sh
fi
'; then
        cat "${stage_log}" >&2 || true
        error "${stage_name} stage failed; see ${stage_log}"
    fi
    log "${stage_name} container create finished: ${container_name}"

    log "${stage_name} container started: ${container_name}"
    if ! run_with_live_log "${stage_log}" docker start -a "${container_name}"; then
        cat "${stage_log}" >&2 || true
        docker rm -f "${container_name}" >/dev/null 2>&1 || true
        error "${stage_name} stage failed; see ${stage_log}"
    fi
    log "${stage_name} container finished: ${container_name}"

    if commit_runtime_test_image_requested; then
        docker image rm -f "${test_image_name}" >/dev/null 2>&1 || true
        log "${stage_name} runtime image commit started: ${test_image_name}"
        if ! docker commit "${container_name}" "${test_image_name}" >"${LOG_DIR}/commit-runtime-test-image.log" 2>&1; then
            cat "${LOG_DIR}/commit-runtime-test-image.log" >&2 || true
            docker rm -f "${container_name}" >/dev/null 2>&1 || true
            error "Failed to commit runtime test image ${test_image_name}"
        fi
        log "${stage_name} runtime image committed: ${test_image_name}"
        printf '%s' "${test_image_name}" > "${RUNTIME_TEST_IMAGE_NAME_FILE}"
        if persist_runtime_test_image_requested; then
            persist_runtime_test_image "${test_image_name}"
        fi
    fi

    log "${stage_name} container cleanup: ${container_name}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true

    log "${stage_name} stage finished using base image ${BASE_IMAGE}"
}

main() {
    local kernel_wheel_path=""
    local runtime_test_image=""
    kernel_wheel_path="$(resolve_prebuilt_kernel_wheel)"
    log "Using prebuilt kernel wheel ${kernel_wheel_path}"

    case "${PIPELINE_STAGE}" in
        build)
            run_in_build_container build "${kernel_wheel_path}"
            ;;
        test)
            runtime_test_image="$(ensure_runtime_test_image_available)"
            if framework_tests_requested; then
                run_framework_tests "${runtime_test_image}"
                return
            fi
            [[ -f "${TEST_COMMAND_FILE}" ]] || error "Missing test command file: ${TEST_COMMAND_FILE}"
            cp -f "${TEST_COMMAND_FILE}" "${ARTIFACT_DIR}/run-test.sh"
            chmod +x "${ARTIFACT_DIR}/run-test.sh"
            docker run --rm \
                -v "${ARTIFACT_DIR}/run-test.sh:/workspace/run-test.sh:ro" \
                -v "${ARTIFACT_DIR}:/workspace/artifacts" \
                -v "${DIST_DIR}:/workspace/dist" \
                -v "${LOG_DIR}:/workspace/logs" \
                -v "${REPO_ROOT}:/workspace/vllm" \
                ${CCACHE_HOST_DIR:+-v "${CCACHE_HOST_DIR}:/workspace/ccache"} \
                ${http_proxy:+-e "http_proxy=${http_proxy}"} \
                ${https_proxy:+-e "https_proxy=${https_proxy}"} \
                ${HTTP_PROXY:+-e "HTTP_PROXY=${HTTP_PROXY}"} \
                ${HTTPS_PROXY:+-e "HTTPS_PROXY=${HTTPS_PROXY}"} \
                ${no_proxy:+-e "no_proxy=${no_proxy}"} \
                ${NO_PROXY:+-e "NO_PROXY=${NO_PROXY}"} \
                "${runtime_test_image}" /bin/bash -lc '/bin/bash /workspace/run-test.sh' >"${LOG_DIR}/test.log" 2>&1 || {
                cat "${LOG_DIR}/test.log" >&2 || true
                error 'test stage failed'
            }
            ;;
        *)
            error "Unsupported stage: ${PIPELINE_STAGE}"
            ;;
    esac
}

main "$@"