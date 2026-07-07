# ComfyUI-Qwen3-ASR

ComfyUI custom nodes for Qwen3-ASR transcription.

This package wraps the official `qwen-asr` Python package and defaults to
`Qwen/Qwen3-ASR-0.6B` for practical local testing. `Qwen/Qwen3-ASR-1.7B` is
also available from the model selector.

## Nodes

- `Qwen3 ASR - Transcribe File`
  - Selects an audio/video file from ComfyUI's `input` directory.
- `Qwen3 ASR - Transcribe Audio`
  - Accepts a standard ComfyUI `AUDIO` input.

Use the `model_name` dropdown for known models, or set `custom_model_id` to a
new Hugging Face model id when Qwen publishes a newer `qwen-asr` compatible
variant. When `custom_model_id` is not empty, it takes precedence over
`model_name`.

Both nodes output:

- `transcript`: recognized speech text
- `language`: detected or requested language
- `metadata_json`: JSON metadata returned by the model wrapper

Word-level timestamps are not exposed in this initial ASR node because
`qwen-asr` requires the separate Qwen3 ForcedAligner model for that mode.

## Install

Install dependencies in the same Python environment used by ComfyUI:

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-Qwen3-ASR/requirements.txt
```

Restart ComfyUI after installation.

The node installs `qwen-asr` from the compatibility fork:

```text
git+https://github.com/endman100/Qwen3-ASR.git#egg=qwen-asr
```

The fork intentionally uses lower-bound-only runtime dependencies so newer
Qwen3-ASR runtime/model support can be picked up with:

```bash
pip install -U -r ComfyUI/custom_nodes/ComfyUI-Qwen3-ASR/requirements.txt
```

When newer dependency versions break compatibility, fix the fork and reinstall
from this node's `requirements.txt`.

## Notes

The first run downloads model weights through Hugging Face. Qwen3-ASR accepts
full language names such as `English` and `Chinese`; this node maps common ISO
codes such as `en`, `zh`, and `ja` to the names expected by `qwen-asr`.

Sources:

- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- https://huggingface.co/collections/Qwen/qwen3-asr
