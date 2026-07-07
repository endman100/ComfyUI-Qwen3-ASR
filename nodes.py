import hashlib
import json
import os
import tempfile
from functools import lru_cache

import folder_paths


MODEL_OPTIONS = [
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
DTYPE_OPTIONS = ["auto", "bfloat16", "float16", "float32"]
LANGUAGE_OPTIONS = [
    "auto",
    "English",
    "Chinese",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]
LANGUAGE_ALIASES = {
    "": None,
    "auto": None,
    "none": None,
    "en": "English",
    "eng": "English",
    "zh": "Chinese",
    "cn": "Chinese",
    "yue": "Cantonese",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def _audio_files():
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    files = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    files = folder_paths.filter_files_content_types(files, ["audio", "video"])
    return sorted(files) or [""]


def _resolve_device(device):
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_dtype(dtype, device):
    import torch

    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if device == "cuda" and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _normalize_language(language):
    value = str(language or "").strip()
    mapped = LANGUAGE_ALIASES.get(value.lower())
    if mapped is not None or value.lower() in LANGUAGE_ALIASES:
        return mapped
    return value


def _metadata_from_item(item, requested_model, requested_language):
    data = dict(getattr(item, "__dict__", {}) or {})
    if not data:
        data = {
            "language": getattr(item, "language", None),
            "text": getattr(item, "text", None),
            "time_stamps": getattr(item, "time_stamps", None),
        }
    data["model"] = requested_model
    data["requested_language"] = requested_language
    return data


def _resolve_model_name(model_name, custom_model_id):
    custom = str(custom_model_id or "").strip()
    return custom or model_name


@lru_cache(maxsize=2)
def _load_model(model_name, device, dtype_name, max_new_tokens):
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise ImportError(
            "ComfyUI-Qwen3-ASR requires qwen-asr. "
            "Install requirements.txt in the same Python environment used by ComfyUI, "
            "then restart ComfyUI."
        ) from exc

    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(dtype_name, resolved_device)
    device_map = "cuda:0" if resolved_device == "cuda" else "cpu"
    if device_map == "cpu":
        dtype = torch.float32

    return Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=1,
        max_new_tokens=int(max_new_tokens),
    )


def _transcribe_path(
    audio_path,
    model_name,
    device,
    dtype,
    language,
    context,
    max_new_tokens,
):
    requested_language = _normalize_language(language)
    model = _load_model(model_name, device, dtype, int(max_new_tokens))
    result = model.transcribe(
        audio_path,
        context=str(context or ""),
        language=requested_language,
        return_time_stamps=False,
    )

    item = result[0] if result else None
    if item is None:
        metadata = {
            "model": model_name,
            "requested_language": requested_language,
            "language": "",
            "text": "",
            "time_stamps": None,
        }
        return "", "", json.dumps(metadata, ensure_ascii=False)

    transcript = str(getattr(item, "text", "") or "")
    detected_language = str(getattr(item, "language", "") or requested_language or "")
    metadata = _metadata_from_item(item, model_name, requested_language)
    return transcript, detected_language, json.dumps(metadata, ensure_ascii=False)


def _audio_to_temp_wav(audio):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "ComfyUI-Qwen3-ASR requires soundfile for AUDIO inputs."
        ) from exc

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    samples = waveform[0].detach().cpu().float().permute(1, 0).numpy()

    handle = tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix="qwen3_asr_",
        delete=False,
    )
    handle.close()
    sf.write(handle.name, samples, sample_rate, format="WAV", subtype="PCM_16")
    return handle.name


def _ui_result(transcript, language, metadata_json):
    preview = transcript or "[No speech detected]"
    return {
        "ui": {"text": (preview,)},
        "result": (transcript, language, metadata_json),
    }


class Qwen3ASRTranscribeFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (_audio_files(),),
                "model_name": (MODEL_OPTIONS, {"default": "Qwen/Qwen3-ASR-0.6B"}),
                "custom_model_id": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Optional HF id, overrides model_name",
                    },
                ),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "dtype": (DTYPE_OPTIONS, {"default": "auto"}),
                "language": (LANGUAGE_OPTIONS, {"default": "auto"}),
                "context": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional vocabulary, names, or context",
                    },
                ),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcript", "language", "metadata_json")
    FUNCTION = "transcribe"
    CATEGORY = "audio/qwen3-asr"
    OUTPUT_NODE = True

    def transcribe(
        self,
        audio,
        model_name,
        custom_model_id,
        device,
        dtype,
        language,
        context,
        max_new_tokens,
    ):
        audio_path = folder_paths.get_annotated_filepath(audio)
        resolved_model_name = _resolve_model_name(model_name, custom_model_id)
        result = _transcribe_path(
            audio_path,
            resolved_model_name,
            device,
            dtype,
            language,
            context,
            max_new_tokens,
        )
        return _ui_result(*result)

    @classmethod
    def IS_CHANGED(cls, audio, **kwargs):
        if not folder_paths.exists_annotated_filepath(audio):
            return audio
        audio_path = folder_paths.get_annotated_filepath(audio)
        digest = hashlib.sha256()
        with open(audio_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(json.dumps(kwargs, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, audio, **kwargs):
        if not audio or not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True


class Qwen3ASRTranscribeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (MODEL_OPTIONS, {"default": "Qwen/Qwen3-ASR-0.6B"}),
                "custom_model_id": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Optional HF id, overrides model_name",
                    },
                ),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "dtype": (DTYPE_OPTIONS, {"default": "auto"}),
                "language": (LANGUAGE_OPTIONS, {"default": "auto"}),
                "context": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional vocabulary, names, or context",
                    },
                ),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcript", "language", "metadata_json")
    FUNCTION = "transcribe"
    CATEGORY = "audio/qwen3-asr"
    OUTPUT_NODE = True

    def transcribe(
        self,
        audio,
        model_name,
        custom_model_id,
        device,
        dtype,
        language,
        context,
        max_new_tokens,
    ):
        temp_path = _audio_to_temp_wav(audio)
        try:
            resolved_model_name = _resolve_model_name(model_name, custom_model_id)
            result = _transcribe_path(
                temp_path,
                resolved_model_name,
                device,
                dtype,
                language,
                context,
                max_new_tokens,
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return _ui_result(*result)


NODE_CLASS_MAPPINGS = {
    "Qwen3ASRTranscribeFile": Qwen3ASRTranscribeFile,
    "Qwen3ASRTranscribeAudio": Qwen3ASRTranscribeAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRTranscribeFile": "Qwen3 ASR - Transcribe File",
    "Qwen3ASRTranscribeAudio": "Qwen3 ASR - Transcribe Audio",
}
