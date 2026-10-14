#!/usr/bin/env bash
# Download the ONNX model folder for the CPU (0.6b) or ONNX GPU (1.7b) image, and the optional forced aligner
# (word timestamps) and speaker diarization models of both ONNX images.
#
#   scripts/download-models.sh 0.6b /srv/models          # -> /srv/models/qwen3-asr-0.6b-onnx
#   scripts/download-models.sh 1.7b /srv/models          # -> /srv/models/qwen3-asr-1.7b-onnx
#   scripts/download-models.sh aligner /srv/models       # -> /srv/models/qwen3-forcedaligner-0.6b-onnx (~1 GB)
#   scripts/download-models.sh diarization /srv/models   # -> /srv/models/speaker-diarization (~35 MB)
#
# 0.6b files come from their original Hugging Face repos; 1.7b comes from dreamyfishmt/qwen3-asr-1.7b-onnx
# (files of andrewleech/qwen3-asr-1.7b-onnx and sorryhyun/qwen3-asr-onnx-gqa plus an FP16 encoder derived from
# andrewleech's FP32 one). The third-party 0.6b repos are pinned to the revisions the server was tested with;
# 1.7b follows the latest revision of our own repo and downloads all of it.
# aligner: the int4 graph of valoomba/Qwen3-ForcedAligner-0.6B-ONNX (community export of Qwen/Qwen3-ForcedAligner-0.6B).
# diarization: pyannote segmentation-3.0 as converted by sherpa-onnx (csukuangfj/sherpa-onnx-pyannote-segmentation-3-0)
# and the 3D-Speaker CAM++ speaker embedding model (Chinese + English) from sherpa-onnx's GitHub releases.
# Uses `hf` if installed, otherwise `uvx --from huggingface_hub hf`.
set -euo pipefail

usage() { echo "usage: $0 {0.6b|1.7b|aligner|diarization} MODEL_DIR" >&2; exit 2; }
[ $# -eq 2 ] || usage
size=$1
root=$2

# Third-party repos: revisions the server was tested with
RHASSPY_0_6B=rhasspy/qwen3-asr-0.6b-onnx-int4-merged@9ea8c26bbf497ef74a84ce19202ce62246af8ab4
SORRYHYUN_GQA=sorryhyun/qwen3-asr-onnx-gqa@075249f70b56cdded1cf4b189cbdde0fb77aeec1
VALOOMBA_ALIGNER=valoomba/Qwen3-ForcedAligner-0.6B-ONNX@261c9ed100c1b18a4a1fbc488e05625dc9a4ae5c
PYANNOTE_SEGMENTATION=csukuangfj/sherpa-onnx-pyannote-segmentation-3-0@9403a6902bb58e3d5ae8c7e77c3422de279db2e0
CAMPPLUS_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
CAMPPLUS_SHA256=aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2
# Our own repo: latest revision
QWEN3_ASR_1_7B=dreamyfishmt/qwen3-asr-1.7b-onnx@main

if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif command -v uvx >/dev/null 2>&1; then
    HF=(uvx --from huggingface_hub hf)
else
    echo "error: needs the Hugging Face CLI (pip install huggingface_hub) or uv (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

# fetch REPO@REVISION DEST [FILE... | OPTIONS]  (no files = the whole repo)
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
        fetch "$QWEN3_ASR_1_7B" "$dest" --exclude README.md --exclude .gitattributes
        ;;
    aligner)
        dest=$root/qwen3-forcedaligner-0.6b-onnx
        fetch "$VALOOMBA_ALIGNER" "$dest" config.json tokenizer.json onnx/model_q4.onnx
        ;;
    diarization)
        dest=$root/speaker-diarization
        fetch "$PYANNOTE_SEGMENTATION" "$dest" model.onnx LICENSE
        mv "$dest/model.onnx" "$dest/segmentation.onnx"
        mv "$dest/LICENSE" "$dest/LICENSE-segmentation"
        curl -fL --retry 3 -o "$dest/embedding.onnx.part" "$CAMPPLUS_URL"
        echo "$CAMPPLUS_SHA256  $dest/embedding.onnx.part" | sha256sum -c --quiet
        mv "$dest/embedding.onnx.part" "$dest/embedding.onnx"
        ;;
    *) usage ;;
esac

# hf keeps download metadata in .cache/ inside the target; the server doesn't need it
rm -rf "$dest/.cache"
case $size in
    aligner) echo "Model ready: $dest (set ENABLE_ALIGNER_MODEL=true, ALIGNER_MODEL_DIR=$(basename "$dest"))" ;;
    diarization) echo "Models ready: $dest (set ENABLE_DIARIZATION=true, DIARIZATION_MODEL_DIR=$(basename "$dest"))" ;;
    *) echo "Model ready: $dest (set ASR_MODEL_DIR=$(basename "$dest"))" ;;
esac
