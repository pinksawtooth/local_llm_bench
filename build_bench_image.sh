#!/usr/bin/env bash

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
DEFAULT_IMAGE_TAG="local-llm-bench:bench"
DEFAULT_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
DEFAULT_ARM64_GHIDRA_URL="https://github.com/ghidra-user-jp/mecha_ghidra/releases/download/v0.1.0-rc.1/mecha_ghidra_docker_arm64_ghidra_12.0.4_patched.zip"
DEFAULT_ARM64_GHIDRA_SHA256="8df75ea0fff62d0e417038771fe851d65164c4d3366d8f14e3e8e100d65f6a13"
DEFAULT_GHIDRA_MCP_REPO_URL="https://github.com/ghidra-user-jp/mecha_ghidra.git"
DEFAULT_GHIDRA_MCP_REF="v0.1.0-rc.1"

IMAGE_TAG="${DEFAULT_IMAGE_TAG}"
PLATFORM="${DEFAULT_PLATFORM}"
GHIDRA_URL="${GHIDRA_URL:-}"
GHIDRA_SHA256="${GHIDRA_SHA256:-}"
GHIDRA_MCP_REPO_URL="${GHIDRA_MCP_REPO_URL:-${DEFAULT_GHIDRA_MCP_REPO_URL}}"
GHIDRA_MCP_REF="${GHIDRA_MCP_REF:-${DEFAULT_GHIDRA_MCP_REF}}"

usage() {
  cat <<'EOF'
Usage:
  ./build_bench_image.sh [options]

Options:
  --ghidra-url URL     Ghidra ZIP URL
  --ghidra-sha256 HEX  Ghidra ZIP SHA256
  --ghidra-mcp-repo-url URL
                       mecha_ghidra repository URL
  --ghidra-mcp-ref REF mecha_ghidra git ref (tag/branch/commit). Default: v0.1.0-rc.1
  --tag NAME           Docker image tag. Default: local-llm-bench:bench
  --platform VALUE     Docker build platform. Default: linux/amd64
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ghidra-url)
      [[ $# -ge 2 ]] || { echo "Error: --ghidra-url requires a value." >&2; exit 2; }
      GHIDRA_URL="$2"
      shift 2
      ;;
    --ghidra-sha256)
      [[ $# -ge 2 ]] || { echo "Error: --ghidra-sha256 requires a value." >&2; exit 2; }
      GHIDRA_SHA256="$2"
      shift 2
      ;;
    --ghidra-mcp-repo-url)
      [[ $# -ge 2 ]] || { echo "Error: --ghidra-mcp-repo-url requires a value." >&2; exit 2; }
      GHIDRA_MCP_REPO_URL="$2"
      shift 2
      ;;
    --ghidra-mcp-ref)
      [[ $# -ge 2 ]] || { echo "Error: --ghidra-mcp-ref requires a value." >&2; exit 2; }
      GHIDRA_MCP_REF="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "Error: --tag requires a value." >&2; exit 2; }
      IMAGE_TAG="$2"
      shift 2
      ;;
    --platform)
      [[ $# -ge 2 ]] || { echo "Error: --platform requires a value." >&2; exit 2; }
      PLATFORM="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${PLATFORM}" == "linux/arm64" && -z "${GHIDRA_URL}" ]]; then
  GHIDRA_URL="${DEFAULT_ARM64_GHIDRA_URL}"
fi
if [[ "${PLATFORM}" == "linux/arm64" && -z "${GHIDRA_SHA256}" && "${GHIDRA_URL}" == "${DEFAULT_ARM64_GHIDRA_URL}" ]]; then
  GHIDRA_SHA256="${DEFAULT_ARM64_GHIDRA_SHA256}"
fi

if [[ -z "${GHIDRA_URL}" ]]; then
  echo "Error: GHIDRA_URL is required. --ghidra-url または環境変数 GHIDRA_URL を指定してください。" >&2
  exit 2
fi
if [[ -z "${GHIDRA_MCP_REPO_URL}" || -z "${GHIDRA_MCP_REF}" ]]; then
  echo "Error: GHIDRA_MCP_REPO_URL と GHIDRA_MCP_REF は空にできません。" >&2
  exit 2
fi

echo "Building ${IMAGE_TAG} for ${PLATFORM}"
echo "Using mecha_ghidra repo: ${GHIDRA_MCP_REPO_URL}@${GHIDRA_MCP_REF}"
/usr/bin/env docker build \
  --platform "${PLATFORM}" \
  --build-arg "GHIDRA_URL=${GHIDRA_URL}" \
  --build-arg "GHIDRA_SHA256=${GHIDRA_SHA256}" \
  --build-arg "GHIDRA_MCP_REPO_URL=${GHIDRA_MCP_REPO_URL}" \
  --build-arg "GHIDRA_MCP_REF=${GHIDRA_MCP_REF}" \
  -f "${REPO_ROOT}/Dockerfile.bench" \
  -t "${IMAGE_TAG}" \
  "${REPO_ROOT}"
