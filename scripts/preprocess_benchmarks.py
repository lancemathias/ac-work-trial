"""Download pinned public evaluation sources and build an explicit local protocol.

No GPU, model calls, training data, calibration data, or remote dataset scripts.
The output is NOT asserted to reproduce Perplexity's unpublished panel manifest.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
REPORT = ROOT / "data/manifests"
FILES = {}
PREVIOUS_FILES = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(suite, url, name):
    path = RAW / suite / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "decision-research-preprocessing"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    data = path.read_bytes()
    key = str(path.relative_to(ROOT))
    if key in PREVIOUS_FILES and sha(data) != PREVIOUS_FILES[key]["sha256"]:
        raise ValueError(f"Cached source checksum changed: {key}")
    FILES[key] = {"url": url, "sha256": sha(data), "bytes": len(data)}
    return data


def hf(suite, repo, revision, file):
    return fetch(suite, f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}", file.replace("/", "__"))


def gh(suite, repo, revision, file):
    return fetch(suite, f"https://raw.githubusercontent.com/{repo}/{revision}/{file}", file.replace("/", "__"))


def parquet(data):
    return pq.read_table(io.BytesIO(data)).to_pylist()


def jsonl(data):
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def example(suite, identifier, state, question, label, source, group=None):
    row = {"id": f"{suite}:{identifier}", "suite": suite, "state": state,
           "question": question, "label": label, "group": str(identifier if group is None else group),
           "source": source, "evaluation_only": True}
    if question["type"] == "choice":
        assert label in question["criteria"], (suite, identifier, label)
        assert 1 <= len(question["criteria"]) <= 255
    elif question["type"] == "noul":
        assert type(label) is bool
    elif question["type"] == "score":
        assert type(label) is int and 0 <= label < len(question["criteria"])
    else:
        raise ValueError("Unsupported question type in this accuracy protocol")
    return row


def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions):
    return {"type": "noul", "instructions": instructions}


def build():
    global PREVIOUS_FILES
    previous = REPORT / "benchmark-preprocessing.json"
    if previous.exists():
        PREVIOUS_FILES = json.loads(previous.read_text())["files"]
    suites = {}
    notes = {}
    def add(name, rows, note):
        suites[name] = rows
        notes[name] = note
        print(f"{name}: {len(rows)} decisions", flush=True)

    s = "WinoGrande"
    rows = parquet(hf(s, "allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "winogrande_xl/validation-00000-of-00001.parquet"))
    add(s, [example(s, i, r["sentence"], choice("Which option correctly fills the blank?", {"1": r["option1"], "2": r["option2"]}), r["answer"], {"split": "validation", "config": "winogrande_xl"}) for i, r in enumerate(rows)], "Labelled validation; official test labels are not used. Not a verified PPLX subset.")

    s = "FinancialPhraseBank"
    data = hf(s, "takala/financial_phrasebank", "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964", "data/FinancialPhraseBank-v1.0.zip")
    z = zipfile.ZipFile(io.BytesIO(data))
    member = next(n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt"))
    rows = []
    for i, line in enumerate(z.read(member).decode("latin-1").splitlines()):
        if not line.strip():
            continue
        text, label = line.rsplit("@", 1)
        rows.append(example(s, i, text, choice("What is the financial sentiment?", {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}), label.strip(), {"subset": "all_agree", "member": member}))
    add(s, rows, "All-agree subset; complete labelled source is evaluation-only. Other agreement subsets are not merged.")

    s = "RAGTruth"
    rev = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
    sources = {r["source_id"]: r for r in jsonl(gh(s, "ParticleMedia/RAGTruth", rev, "dataset/source_info.jsonl"))}
    rows = []
    for r in jsonl(gh(s, "ParticleMedia/RAGTruth", rev, "dataset/response.jsonl")):
        if r["split"] != "test":
            continue
        context = sources[r["source_id"]]
        rows.append(example(s, r["id"], {"source": context["source_info"], "response": r["response"]}, noul("Does the response contain any hallucinated or unsupported content relative to the source?"), bool(r["labels"]), {"split": "test", "task_type": context["task_type"], "quality": r["quality"]}, r["source_id"]))
    add(s, rows, "Response-level binary: any annotated span is positive; all test quality categories retained. Gold spans and source prompts never enter the model input.")

    s = "JudgeBench"
    file = "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
    rows = jsonl(gh(s, "ScalerLab/JudgeBench", "e2c52c284e735e139b3daa61c206ee208f36c461", file))
    add(s, [example(s, r["pair_id"], {"question": r["question"], "response_A": r["response_A"], "response_B": r["response_B"]}, choice("Which response is more correct?", {"A": "Response A", "B": "Response B"}), {"A>B": "A", "B>A": "B"}[r["label"]], {"cohort": "gpt-4o-2024-05-13", "source": r["source"]}, f"{r['source']}:{r['original_id']}") for r in rows], "GPT-4o response cohort only; preserve source question text. Position-swap testing belongs to evaluation, not this dataset.")

    s = "BBH"
    repo, rev = "lukaemon/bbh", "982bb89fd79532a8ac676a61fc42eb1aeec63f99"
    # Enumerate files at the immutable revision, never the moving main branch.
    meta = json.loads(fetch(s, f"https://huggingface.co/api/datasets/{repo}/revision/{rev}", "repository.json"))
    rows, excluded = [], Counter()
    for f in meta["siblings"]:
        file = f["rfilename"]
        if not file.endswith(".parquet"):
            continue
        task = file.split("/")[0]
        for i, r in enumerate(parquet(hf(s, repo, rev, file))):
            options = dict(re.findall(r"^\(([A-Z])\)\s+(.+)$", r["input"], re.MULTILINE))
            target = r["target"].strip()
            key = target.strip("()")
            if not options:
                fixed = {"boolean_expressions": ["False", "True"], "causal_judgement": ["No", "Yes"], "formal_fallacies": ["invalid", "valid"], "navigate": ["No", "Yes"], "sports_understanding": ["no", "yes"], "web_of_lies": ["No", "Yes"]}.get(task)
                options = {k: k for k in fixed} if fixed else {}
                key = target
            if key not in options:
                excluded[task] += 1
                continue
            rows.append(example(s, f"{task}:{i}", r["input"], choice("Select the correct answer to the supplied task.", options), key, {"split": "test", "subtask": task}))
    add(s, rows, {"protocol": "Closed-choice and explicit Boolean tasks only; free-form tasks retained raw, not forced into invented choices.", "excluded_by_subtask": dict(excluded), "aggregate": "Report per subtask then macro-average included subtasks; reference task selection unresolved."})

    s = "JevBench_public_hard"
    rows = jsonl(gh(s, "fstandhartinger/jevbench", "7625ecb1b075cfe4d09d34141d353325825eea1d", "datasets/public/hard.jsonl"))
    normalized = []
    for r in rows:
        label = r["expected"]
        if r["question"]["type"] == "noul":
            assert label in (True, False, "true", "false", "yes", "no"), label
            label = label is True or label in ("true", "yes")
        normalized.append(example(s, r["id"], r["state"], r["question"], label, {"split": "public_hard", "family": r["family"]}, r["group"]))
    add(s, normalized, "Public hard only; preserve original question and option order, including Score rubrics. Author rationales and surface answers excluded from inputs. Noul yes/no strings mapped to Boolean labels; Score gold is a zero-based level index.")

    s = "TabFact"
    rows = parquet(hf(s, "wenhu/tab_fact", "cecfbe87432b0e7fdcac4578f8ac6fc87d08db76", "tab_fact/test/0000.parquet"))
    add(s, [example(s, i, {"table": r["table_text"], "caption": r["table_caption"], "statement": r["statement"]}, noul("Is the statement supported by the table?"), bool(r["label"]), {"split": "test", "config": "tab_fact", "table_id": r["table_id"]}, r["table_id"]) for i, r in enumerate(rows)], "Pinned HF converted-parquet export of full labelled test. Preserve table text and caption; 1=entailed, 0=refuted.")

    s = "ContractNLI"
    z = zipfile.ZipFile(io.BytesIO(gh(s, "stanfordnlp/contract-nli", "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9", "resources/contract-nli.zip")))
    obj = json.loads(z.read("contract-nli/test.json"))
    rows = []
    criteria = {"Entailment": "The contract entails the hypothesis", "Contradiction": "The contract contradicts the hypothesis", "NotMentioned": "The contract does not establish either"}
    for doc in obj["documents"]:
        assert len(doc["annotation_sets"]) == 1, "Unexpected multiple annotation sets"
        annotations = doc["annotation_sets"][0]["annotations"]
        for key, annotation in annotations.items():
            rows.append(example(s, f"{doc['id']}:{key}", {"contract": doc["text"], "hypothesis": obj["labels"][key]["hypothesis"]}, choice("What is the relation between the contract and the hypothesis?", criteria), annotation["choice"], {"split": "test", "document_id": doc["id"]}, doc["id"]))
    add(s, rows, "Full test documents, each hypothesis a 3-way decision. Gold evidence spans excluded. No context truncation.")

    s = "Circa"
    data = gh(s, "google-research-datasets/circa", "02ad965518ab2fbd8bb24463d312ebb03bac5368", "circa-data.tsv")
    raw = list(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"))
    gh(s, "google-research-datasets/circa", "02ad965518ab2fbd8bb24463d312ebb03bac5368", "README.md")
    categories = sorted({r["goldstandard1"] for r in raw if r["goldstandard1"] and r["goldstandard1"] not in ("nan", "NA")})
    rows = [example(s, r["id"], {"context": r["context"], "question": r["question-X"], "answer": r["answer-Y"]}, choice("How should the indirect answer to the yes/no question be interpreted?", {k: k for k in categories}), r["goldstandard1"], {"split": "source_unsplit", "label_field": "goldstandard1"}, r["context"]) for r in raw if r["goldstandard1"] in categories]
    add(s, rows, {"protocol": "Full source, native fine-grained goldstandard1 labels; no invented train/test split.", "labels": categories, "unlabelled_excluded": len(raw)-len(rows), "reference_label_granularity": "unresolved"})

    s = "Belebele"
    rows = jsonl(hf(s, "facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", "data/eng_Latn.jsonl"))
    add(s, [example(s, i, {"passage": r["flores_passage"], "question": r["question"]}, choice("Which answer is correct based on the passage?", {str(k): r[f"mc_answer{k}"] for k in range(1,5)}), str(r["correct_answer_num"]), {"language": "eng_Latn", "split": "test"}, r["link"]) for i, r in enumerate(rows)], "English only, all 900 questions; reference language and subset unresolved.")

    s = "TruthfulQA_binary"
    rows = []
    raw = parquet(hf(s, "truthfulqa/truthful_qa", "741b8276f2d1982aa3d5b832d3ee81ed3b896490", "multiple_choice/validation-00000-of-00001.parquet"))
    for i, r in enumerate(raw):
        for j, (answer, label) in enumerate(zip(r["mc2_targets"]["choices"], r["mc2_targets"]["labels"], strict=True)):
            rows.append(example(s, f"{i}:{j}", {"question": r["question"], "answer": answer}, noul("Is the proposed answer truthful?"), bool(label), {"split": "validation", "conversion": "mc2_answer_truth_binary"}, i))
    add(s, rows, "Explicit local binary variant: each MC2 candidate answer becomes a truth decision. NOT asserted to match PPLX's binary conversion; grouped by original question.")

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "reference_protocol_matched": False,
               "script_sha256": sha(Path(__file__).read_bytes()), "files": FILES, "suites": {}}
    for suite, rows in suites.items():
        assert len({r["id"] for r in rows}) == len(rows)
        payload = b"".join((json.dumps(r, ensure_ascii=False, allow_nan=False, sort_keys=True)+"\n").encode() for r in rows)
        (OUT / f"{suite}.jsonl").write_bytes(payload)
        summary["suites"][suite] = {"rows": len(rows), "groups": len({r["group"] for r in rows}), "sha256": sha(payload), "labels": dict(Counter(str(r["label"]) for r in rows)), "question_types": dict(Counter(r["question"]["type"] for r in rows)), "notes": notes[suite]}
    (REPORT / "benchmark-preprocessing.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print("Saved all 11 suite files and provenance manifest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
