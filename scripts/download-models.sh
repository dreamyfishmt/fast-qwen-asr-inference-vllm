#!/usr/bin/env bash
# Download the ONNX model folder for the CPU (0.6b) or ONNX GPU (1.7b) image.
#
#   scripts/download-models.sh 0.6b /srv/models   # -> /srv/models/qwen3-asr-0.6b-onnx
#   scripts/download-models.sh 1.7b /srv/models   # -> /srv/models/qwen3-asr-1.7b-onnx
#
# Files come from their original Hugging Face repos, pinned to the revisions the server was tested
# with. Uses `hf` if installed, otherwise `uvx --from huggingface_hub hf`.
set -euo pipefail

usage() { echo "usage: $0 {0.6b|1.7b} MODEL_DIR" >&2; exit 2; }
[ $# -eq 2 ] || usage
size=$1
root=$2

# Revisions the server was tested with
RHASSPY_0_6B=rhasspy/qwen3-asr-0.6b-onnx-int4-merged@9ea8c26bbf497ef74a84ce19202ce62246af8ab4
ANDREWLEECH_1_7B=andrewleech/qwen3-asr-1.7b-onnx@df916193ac67e59347769891a21e10d81d12acdd
SORRYHYUN_GQA=sorryhyun/qwen3-asr-onnx-gqa@075249f70b56cdded1cf4b189cbdde0fb77aeec1

if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif command -v uvx >/dev/null 2>&1; then
    HF=(uvx --from huggingface_hub hf)
else
    echo "error: needs the Hugging Face CLI (pip install huggingface_hub) or uv (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

# fetch REPO@REVISION DEST FILE...
fetch() {
    local spec=$1 dest=$2
    shift 2
    if ! "${HF[@]}" download "${spec%@*}" "$@" --revision "${spec#*@}" --local-dir "$dest"; then
        # Some proxies/networks reject Hugging Face's Xet storage; plain HTTP downloads still work
        echo "Download failed; retrying without Xet (HF_HUB_DISABLE_XET=1)" >&2
        HF_HUB_DISABLE_XET=1 "${HF[@]}" download "${spec%@*}" "$@" --revision "${spec#*@}" --local-dir "$dest"
    fi
}

case $size in
    0.6b)
        dest=$root/qwen3-asr-0.6b-onnx
        fetch "$RHASSPY_0_6B" "$dest" config.json tokenizer.json embed_tokens.bin encoder.int4.onnx encoder.int4.onnx.data
        fetch "$SORRYHYUN_GQA" "$dest" decoder-0.6b-fp32.onnx decoder-0.6b-fp32.onnx.data
        ;;
    1.7b)
        dest=$root/qwen3-asr-1.7b-onnx
        fetch "$ANDREWLEECH_1_7B" "$dest" config.json tokenizer.json embed_tokens.bin encoder.onnx
        fetch "$SORRYHYUN_GQA" "$dest" decoder-1.7b-fp16.onnx decoder-1.7b-fp16.onnx.data
        ;;
    *) usage ;;
esac

# hf keeps download metadata in .cache/ inside the target; the server doesn't need it
rm -rf "$dest/.cache"
echo "Model ready: $dest (set ASR_MODEL_DIR=$(basename "$dest"))"
