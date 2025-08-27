#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prestige Prep - Curriculum Generator (Gemini JSON-safe)
- Groups by (section x type)
- Generates items in batches until per-group 'scale' reached
- Enforces strict JSON with response_schema
- Sanitizes minor formatting issues before json.loads
- Uploads artifacts to GCS and (optionally) writes Firestore index

Env (.agent/agent.env):
  GOOGLE_API_KEY=...
  GCP_PROJECT=learning-470300
  GCP_BUCKET=learning-470300-prestige-content
  FIREBASE_PROJECT_ID=learning-470300   # optional; only used if --write-firestore on
  GENAI_MODEL=gemini-2.5-pro            # default if not set

Usage (example):
  nohup python3 agent.py \
    --exam sat \
    --sections Math,Reading,Writing \
    --types CCR,APP,TEC \
    --scale 10 \
    --batch-size 5 \
    --max-retries 4 \
    --set-name "sat-smoke-$(date +%Y%m%d-%H%M%S)" \
    --upload gcs \
    --write-firestore on \
    > ".agent/reports/generate-${SET_NAME}.log" 2>&1 &
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm

# Gemini (google-genai)
from google import genai
from google.genai.types import GenerateContentConfig, Schema

# GCS + Firestore (Application Default Credentials on GCE)
from google.cloud import storage
from google.cloud import firestore


# --------------------------------------------------------------------------------------
# Constants / Directories
# --------------------------------------------------------------------------------------

STATE_DIR = Path(".agent")
REPORTS_DIR = STATE_DIR / "reports"
OUT_DIR = STATE_DIR / "out"

STATE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Logging helpers (stdout; let caller redirect to .agent/reports/*.log)
# --------------------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[AGENT] {ts} {msg}", flush=True)


# --------------------------------------------------------------------------------------
# JSON sanitation & schema
# --------------------------------------------------------------------------------------

def sanitize_json_array(raw: str) -> str:
    """
    Remove code fences, keep outermost JSON array, fix trailing commas, normalize quotes.
    """
    if not isinstance(raw, str):
        raise ValueError("Model output was not text.")

    # strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    # keep the outermost array only
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        raise ValueError("No JSON array found in model output.")
    frag = m.group(0)

    # remove trailing commas before ] or }
    frag = re.sub(r",\s*(\]|\})", r"\1", frag)

    # normalize curly quotes to straight
    frag = (frag
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'"))

    return frag


def parse_items_strict(raw: str) -> List[Dict[str, Any]]:
    frag = sanitize_json_array(raw)
    return json.loads(frag)


ITEM_SCHEMA = Schema(
    type="OBJECT",
    properties={
        "id": Schema(type="STRING"),
        "exam": Schema(type="STRING"),
        "section": Schema(type="STRING"),
        "category": Schema(type="STRING"),
        "type": Schema(type="STRING"),
        "difficulty": Schema(type="NUMBER"),
        "tags": Schema(type="ARRAY", items=Schema(type="STRING")),
        "stem_latex": Schema(type="STRING"),
        "choices": Schema(
            type="OBJECT",
            properties={
                "A": Schema(type="STRING"),
                "B": Schema(type="STRING"),
                "C": Schema(type="STRING"),
                "D": Schema(type="STRING"),
            },
            required=["A", "B", "C", "D"],
        ),
        "answer": Schema(type="STRING"),
        "explanation_latex": Schema(type="STRING"),
    },
    required=[
        "id", "exam", "section", "category", "type", "difficulty",
        "tags", "stem_latex", "choices", "answer", "explanation_latex",
    ],
)

ITEMS_ARRAY_SCHEMA = Schema(type="ARRAY", items=ITEM_SCHEMA)


# --------------------------------------------------------------------------------------
# Gemini client
# --------------------------------------------------------------------------------------

def build_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    return genai.Client(api_key=api_key)


def generate_items_json(client: genai.Client, model_name: str, prompt: str) -> List[Dict[str, Any]]:
    cfg = GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=ITEMS_ARRAY_SCHEMA,
        max_output_tokens=8192,
    )
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=cfg,
    )
    raw = getattr(resp, "text", "") or ""
    return parse_items_strict(raw)


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are generating SAT {section} questions of type {qtype}.
Return ONLY a JSON array (no prose) of exactly {n} items. Each item must match this schema:
["id","exam","section","category","type","difficulty","tags","stem_latex","choices","answer","explanation_latex"]

Constraints:
- exam="SAT", section="{section}", type="{qtype}"
- difficulty: integers 1..5 (mix them)
- tags: small array of short keywords
- choices: object with string keys A,B,C,D (exactly)
- answer: one of "A","B","C","D"
- stem_latex / explanation_latex: valid LaTeX; escape backslashes correctly; do NOT truncate text.
- Do NOT include markdown fences, commentary, or trailing text—JSON array only.

Notes by type:
- CCR: Command of Evidence / Reading detail or proof, or Math Heart-of-Algebra style clarity.
- APP: Application / real-world contexts.
- TEC: Textual evidence / data, charts, cross-lines reference (Reading) or calculator/data interpretation (Math).

Produce the array now with {n} complete items only.
"""


# --------------------------------------------------------------------------------------
# Storage / Firestore
# --------------------------------------------------------------------------------------

def upload_to_gcs(project: str, bucket_name: str, prefix: str, local_path: Path) -> str:
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}{local_path.name}")
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{prefix}{local_path.name}"


def write_firestore_index(project: str, set_name: str, meta: Dict[str, Any]) -> None:
    db = firestore.Client(project=project)
    col = db.collection("contentSets")
    doc = col.document(set_name)
    doc.set(meta, merge=True)


# --------------------------------------------------------------------------------------
# Main generation
# --------------------------------------------------------------------------------------

def main() -> int:
    # Load .env if present
    env_path = STATE_DIR / "agent.env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        log(f"Loaded env from {env_path}")

    parser = argparse.ArgumentParser(description="Prestige Prep curriculum generator")
    parser.add_argument("--exam", default="sat", help="sat | act (string)")
    parser.add_argument("--sections", default="Math,Reading,Writing", help="Comma list")
    parser.add_argument("--types", default="CCR,APP,TEC", help="Comma list (include PF if desired)")
    parser.add_argument("--scale", type=int, default=10, help="Target items per (section,type)")
    parser.add_argument("--batch-size", type=int, default=5, help="Items per model call")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per batch on parse failure")
    parser.add_argument("--set-name", default=f"sat-set-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--upload", choices=["gcs", "none"], default="gcs")
    parser.add_argument("--write-firestore", choices=["on", "off"], default="off")
    args = parser.parse_args()

    exam = args.exam.upper()
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    qtypes = [t.strip() for t in args.types.split(",") if t.strip()]
    per_group_target = max(1, int(args.scale))
    batch_size = max(1, int(args.batch_size))
    retries = max(0, int(args.max_retries))
    set_name = args.set_name.strip()
    model_name = os.environ.get("GENAI_MODEL", "gemini-2.5-pro")

    gcp_project = os.environ.get("GCP_PROJECT", "")
    gcs_bucket = os.environ.get("GCP_BUCKET", "")
    fb_project = os.environ.get("FIREBASE_PROJECT_ID", "")

    log(f"Start set={set_name} exam={exam} sections={sections} types={qtypes} "
        f"scale={per_group_target} batch={batch_size} model={model_name}")

    client = build_client()

    # Collect items
    all_items: List[Dict[str, Any]] = []
    per_group_counts = {}

    total_groups = len(sections) * len(qtypes)
    pbar = tqdm(total=total_groups, desc="Groups", ncols=80)

    for section in sections:
        for qtype in qtypes:
            generated_for_group = 0
            attempts_for_batch = 0
            log(f"Generating up to {per_group_target} items for {section}/{qtype} …")

            while generated_for_group < per_group_target:
                n = min(batch_size, per_group_target - generated_for_group)
                prompt = PROMPT_TEMPLATE.format(section=section, qtype=qtype, n=n)

                success = False
                attempts_for_batch = 0
                while attempts_for_batch <= retries and not success:
                    attempts_for_batch += 1
                    log(f"Generate batch n={n} for {section}/{qtype} (attempt {attempts_for_batch}) …")
                    try:
                        items = generate_items_json(client, model_name, prompt)
                        # enforce fields / patch exam/section/type if missing
                        for it in items:
                            it.setdefault("exam", exam)
                            it.setdefault("section", section)
                            it.setdefault("type", qtype)
                        all_items.extend(items)
                        generated_for_group += len(items)
                        success = True
                    except Exception as e:
                        log(f"PARSE ERROR: {e}")
                        if attempts_for_batch > retries:
                            log(f"FATAL: Could not produce valid JSON after {retries} retries for {section}/{qtype}.")
                            # Move on to next group without killing entire run
                            success = False
                            break
                        else:
                            time.sleep(1.0)  # small backoff

                if not success:
                    # stop trying this group; continue to next group
                    break

            per_group_counts[f"{section}/{qtype}"] = generated_for_group
            pbar.update(1)

    pbar.close()

    # Write local artifacts
    set_dir = OUT_DIR / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    items_path = set_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    meta = {
        "setName": set_name,
        "exam": exam,
        "sections": sections,
        "types": qtypes,
        "counts": per_group_counts,
        "totalCount": len(all_items),
        "model": model_name,
        "createdAt": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    meta_path = set_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Wrote local artifacts: {items_path} and {meta_path}")

    # Upload to GCS
    gcs_prefix = ""
    if args.upload == "gcs":
        if not gcp_project or not gcs_bucket:
            log("WARN: GCP_PROJECT/GCP_BUCKET not set; skipping GCS upload.")
        else:
            gcs_prefix = f"content/sets/{set_name}/"
            try:
                uri_items = upload_to_gcs(gcp_project, gcs_bucket, gcs_prefix, items_path)
                uri_meta = upload_to_gcs(gcp_project, gcs_bucket, gcs_prefix, meta_path)
                log(f"Uploaded: {uri_items}")
                log(f"Uploaded: {uri_meta}")
            except Exception as e:
                log(f"WARN: GCS upload failed: {e}")

    # Firestore index (optional)
    if args.write_firestore == "on":
        if not fb_project:
            log("WARN: FIREBASE_PROJECT_ID not set; skipping Firestore.")
        else:
            try:
                meta_for_fs = dict(meta)
                meta_for_fs["gcsPath"] = f"gs://{gcs_bucket}/{gcs_prefix}" if gcs_prefix else ""
                write_firestore_index(fb_project, set_name, meta_for_fs)
                log(f"Firestore index updated: contentSets/{set_name}")
            except Exception as e:
                log(f"WARN: Firestore write failed: {e}")

    log("## FINISHED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)

