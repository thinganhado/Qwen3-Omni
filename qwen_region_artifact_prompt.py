#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "prompts" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "prompts" / "region_forensics_user.txt"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "qwen3_omni"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/ALLM/Qwen3-Omni-30B-A3B-Thinking/"
DEFAULT_REGION_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv"
DEFAULT_WAV_ROOT = "/datasets/work/dss-deepfake-audio/work/data/voc-v4/voc.v4/wav/"
DEFAULT_SPEC_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid/"


def _load_text_file(path: Path, field_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.system_file) if args.system_file else DEFAULT_SYSTEM_FILE, "--system-file")


def _resolve_user_template(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.user_template_file) if args.user_template_file else DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")


def _read_input_items(args: argparse.Namespace):
    items = []
    if args.input_jsonl:
        input_path = Path(args.input_jsonl).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"--input-jsonl does not exist: {input_path}")
        with input_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {lineno} must be a JSON object")
                items.append(obj)
    elif args.prompt:
        items = [{"sample_id": "single", "region_id": 0, "input_text": args.prompt}]
    elif args.region_csv:
        csv_path = Path(args.region_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"--region-csv does not exist: {csv_path}")

        by_sample = defaultdict(list)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid_raw = str(row.get("region_id", "")).strip()
                if not rid_raw:
                    continue
                try:
                    rid = int(rid_raw)
                except ValueError:
                    continue

                sample_id_raw = str(row.get("sample_id", "")).strip()
                sample_stem = Path(sample_id_raw).stem if sample_id_raw else "global"
                by_sample[sample_stem].append(
                    {
                        "sample_id_raw": sample_id_raw,
                        "sample_id": sample_stem,
                        "region_id": rid,
                        "time": str(row.get("T", "")).strip(),
                        "frequency": str(row.get("F", "")).strip(),
                        "phonetic": str(row.get("P_type", "")).strip(),
                        "raw_row": dict(row),
                    }
                )

        wav_root = Path(args.wav_root).expanduser().resolve()
        spec_root = Path(args.spec_root).expanduser().resolve()
        skipped_missing_media = 0
        for sample_id, rows in sorted(by_sample.items()):
            rows_sorted = sorted(rows, key=lambda x: x["region_id"])
            if len(rows_sorted) < 3:
                continue
            top3 = rows_sorted[:3]
            audio_path = wav_root / f"{sample_id}.wav"
            image_path = spec_root / f"{sample_id}_grid_img_edge_number_axes.png"
            if (not audio_path.exists()) or (not image_path.exists()):
                skipped_missing_media += 1
                continue
            items.append(
                {
                    "sample_id": sample_id,
                    "region_id": top3[0]["region_id"],
                    "audio": str(audio_path),
                    "image": str(image_path),
                    "ID1": top3[0]["region_id"],
                    "time1": top3[0]["time"],
                    "frequency1": top3[0]["frequency"],
                    "phonetic1": top3[0]["phonetic"],
                    "ID2": top3[1]["region_id"],
                    "time2": top3[1]["time"],
                    "frequency2": top3[1]["frequency"],
                    "phonetic2": top3[1]["phonetic"],
                    "ID3": top3[2]["region_id"],
                    "time3": top3[2]["time"],
                    "frequency3": top3[2]["frequency"],
                    "phonetic3": top3[2]["phonetic"],
                    "rows_top3": top3,
                    "input_text": "",
                }
            )
        if skipped_missing_media:
            print(f"[csv] skipped_samples_missing_media={skipped_missing_media}")
    else:
        raise ValueError("Provide one of --region-csv, --input-jsonl, or --prompt.")

    if args.max_items is not None:
        items = items[: args.max_items]
    if not items:
        raise ValueError("No input items found.")

    normalized = []
    for idx, item in enumerate(items):
        sample_id = str(item.get("sample_id", "sample"))
        try:
            region_id = int(item.get("region_id", idx))
        except Exception:
            region_id = idx
        item2 = dict(item)
        item2["sample_id"] = sample_id
        item2["region_id"] = region_id
        normalized.append(item2)
    return normalized


def _normalize_media_ref(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value
    p = Path(value).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Media path does not exist: {p}")
    return str(p)


def build_messages(args: argparse.Namespace, item: dict):
    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)
    prompt_text = str(item.get("input_text", "")).strip()
    user_prompt = user_template.format_map(defaultdict(str, {"input_text": prompt_text, **item})).strip()
    if not user_prompt:
        user_prompt = prompt_text
    if not user_prompt:
        raise ValueError(f"Empty user prompt for sample_id={item['sample_id']} region_id={item['region_id']}")

    user_content = []
    if item.get("audio"):
        user_content.append({"type": "audio", "audio": _normalize_media_ref(str(item["audio"]))})
    if item.get("image"):
        user_content.append({"type": "image", "image": _normalize_media_ref(str(item["image"]))})
    if item.get("video"):
        user_content.append({"type": "video", "video": _normalize_media_ref(str(item["video"]))})
    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    return messages, user_prompt


def _resolve_torch_dtype(dtype_str: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported --dtype: {dtype_str}. Use one of: {list(mapping.keys())}")
    return mapping[dtype_str]


def _generate_one(model, processor, messages, max_new_tokens, do_sample, temperature, top_p, top_k, use_audio_in_video):
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    ).to(model.device).to(model.dtype)

    sample_flag = bool(do_sample and temperature > 0.0)
    text_ids, _audio = model.generate(
        **inputs,
        return_audio=False,
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=sample_flag,
        thinker_temperature=temperature,
        thinker_top_p=top_p,
        thinker_top_k=top_k,
        use_audio_in_video=use_audio_in_video,
    )
    return processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _write_sample_grouped_json(output_dir: Path, records_by_sample: dict):
    for sample_id, records in records_by_sample.items():
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        by_region = {}
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            by_region[int(rid)] = rec
        payload = {
            "sample_id": sample_id,
            "num_regions": len(by_region),
            "regions": [by_region[rid] for rid in sorted(by_region.keys())],
        }
        (sample_dir / "json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Omni prompt runner scaffold for region artifact analysis.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")
    parser.add_argument(
        "--region-csv",
        default=DEFAULT_REGION_CSV,
        help="CSV with region_id/T/F/P_type (and optional sample_id).",
    )
    parser.add_argument("--wav-root", default=DEFAULT_WAV_ROOT, help="Root for WAV files named <sample_id>.wav")
    parser.add_argument(
        "--spec-root",
        default=DEFAULT_SPEC_ROOT,
        help="Root for GRID spectrogram files named <sample_id>_grid_img_edge_number_axes.png",
    )
    parser.add_argument("--input-jsonl", default=None, help="JSONL input file. Schema to be finalized.")
    parser.add_argument("--prompt", default=None, help="Single text prompt for quick testing.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--system-file", default=None, help=f"Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument("--user-template-file", default=None, help=f"Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--flash-attn2", action="store_true", help="Use flash_attention_2 backend in transformers.")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-audio-in-video", dest="use_audio_in_video", action="store_true", default=True)
    parser.add_argument("--no-use-audio-in-video", dest="use_audio_in_video", action="store_false")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--print-messages", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    items = _read_input_items(args)
    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("Invalid sharding args.")
    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        if not items:
            raise ValueError("No items assigned to this shard.")

    torch_dtype = _resolve_torch_dtype(args.dtype)
    model_kwargs = {
        "dtype": torch_dtype,
        "device_map": args.device_map,
    }
    if args.flash_attn2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_id)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_fp = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = out_path.open("w" if args.overwrite else "a", encoding="utf-8")

    records_by_sample = defaultdict(list)
    try:
        for idx, item in enumerate(items, start=1):
            messages, user_prompt = build_messages(args, item)
            if args.print_messages:
                print(messages)
            response = _generate_one(
                model=model,
                processor=processor,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_audio_in_video=args.use_audio_in_video,
            )
            record = {
                "sample_id": item["sample_id"],
                "region_id": item["region_id"],
                "prompt": user_prompt,
                "response": response,
                "input": item,
            }
            records_by_sample[item["sample_id"]].append(record)
            print(f"[{idx}/{len(items)}] {item['sample_id']}__r{item['region_id']}")
            print(response)
            if jsonl_fp:
                jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if jsonl_fp:
            jsonl_fp.close()

    _write_sample_grouped_json(output_root, records_by_sample)


if __name__ == "__main__":
    main()
