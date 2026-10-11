#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikidata popularity annotation used to build the popularity scores of DUET (`pop_sum`).

Input: CSV file(s) with the columns subjectLabel, relation, objectLabel (one fact per row).
For every fact the script resolves the subject / object to Wikidata items and the relation to a
Wikidata property (public SPARQL endpoint, no API key needed), then writes:

  subject_qid, property_pid, object_qid, question, answer,
  pop_subject = sitelinks + identifiers + statements of the subject item,
  pop_object  = sitelinks + identifiers + statements of the object item,
  pop_sum     = pop_subject + pop_object   (the popularity used by AdaPop),
  triple_pop  = number of Wikidata items with the same (property, object).

Usage:
  pip install pandas tqdm SPARQLWrapper
  export WDQS_USER_AGENT="MyProject/1.0 (contact: <your email>)"   # Wikidata asks for a contact
  python community/methods/AdaPop/wikidata_popularity.py --file facts.csv --out_dir popularity/
  python community/methods/AdaPop/wikidata_popularity.py --in_dir facts/ --combine

Then train AdaPop with `+data.forget.<DATASET>.args.popularity_key=pop_sum`.

ENV knobs: WDQS_USER_AGENT, WDQS_BATCH, WDQS_RETRIES, WDQS_BACKOFF, WDQS_RATE_SLEEP
"""

import os
import time
import glob
import argparse
import urllib.error
from typing import Optional, Dict, Tuple, List

import pandas as pd
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON


# =====================
# Config
# =====================

WDQS_URL = "https://query.wikidata.org/sparql"
USER_AGENT = os.environ.get("WDQS_USER_AGENT", "TripletQA/1.0 (contact: <your email>)")
RATE_SLEEP = float(os.environ.get("WDQS_RATE_SLEEP", 0.25))  # pause between queries
RETRIES = int(os.environ.get("WDQS_RETRIES", 3))
BACKOFF = float(os.environ.get("WDQS_BACKOFF", 1.7))
BATCH = int(os.environ.get("WDQS_BATCH", 80))

REQUIRED_COLS = {"subjectLabel", "relation", "objectLabel"}

PREFIXES = (
    "PREFIX wd: <http://www.wikidata.org/entity/>\n"
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
    "PREFIX wikibase: <http://wikiba.se/ontology#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
    "PREFIX bd: <http://www.bigdata.com/rdf#>\n"
    "PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>\n"
)


# =====================
# SPARQL init
# =====================

sparql = SPARQLWrapper(WDQS_URL)
sparql.setReturnFormat(JSON)
try:
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
except Exception:
    pass
try:
    sparql.setTimeout(120)
except Exception:
    pass


# =====================
# Helpers
# =====================


def _esc_lit_single(s: str) -> str:
    """Escape for single-quoted SPARQL literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _run_once(query: str) -> dict:
    q = PREFIXES + "\n" + query
    sparql.setQuery(q)
    res = sparql.query().convert()
    time.sleep(RATE_SLEEP)
    return res


def _run_sparql(query: str) -> dict:
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return _run_once(query)
        except Exception as e:
            last = e
            time.sleep(BACKOFF**attempt)
    raise last if last else RuntimeError("SPARQL failed without exception")


def _run_chunked(
    items, builder_fn, max_batch: int, stage_name: str = "", position: int = 1
):
    """
    Chunked runner that splits on 504/timeouts. Returns list of JSON responses.
    Shows a simple tqdm counter per stage for visibility.
    """
    items = list(items)
    results = []
    pbar = tqdm(
        total=len(items),
        desc=stage_name,
        position=position,
        leave=False,
        dynamic_ncols=True,
    )

    def _process(chunk, cur_batch):
        if not chunk:
            return
        if cur_batch <= 0:
            cur_batch = 1
        for i in range(0, len(chunk), cur_batch):
            sub = chunk[i : i + cur_batch]
            q = builder_fn(sub)
            try:
                results.append(_run_sparql(q))
                pbar.update(len(sub))
            except Exception as e:
                # split subchunk on gateway/timeout
                if isinstance(e, urllib.error.HTTPError) and e.code == 504:
                    _process(sub, max(1, cur_batch // 2))
                else:
                    msg = str(e).lower()
                    if (
                        "timeout" in msg
                        or "timed out" in msg
                        or "gateway timeout" in msg
                    ):
                        _process(sub, max(1, cur_batch // 2))
                    else:
                        pbar.close()
                        raise

    _process(items, max_batch)
    pbar.close()
    return results


def _ensure_outdir_for(in_path: str, out_dir: Optional[str]) -> str:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    d = os.path.join(os.path.dirname(os.path.abspath(in_path)), "qa_res")
    os.makedirs(d, exist_ok=True)
    return d


# =====================
# Resolve QIDs / PIDs
# =====================


def resolve_items_qids(labels: List[str]) -> Dict[str, Optional[str]]:
    """label -> QID (exact rdfs:label|skos:altLabel; fallback mwapi)."""
    out: Dict[str, Optional[str]] = {lbl: None for lbl in labels}
    uniq = sorted({str(x).strip() for x in labels if str(x).strip()})

    # pass 1: exact (chunked)
    def builder_exact(chunk):
        vals = " ".join([f"'{_esc_lit_single(x)}'@en" for x in chunk])
        return (
            "SELECT ?needle ?item ?sitelinks WHERE {\n"
            f"  VALUES ?needle {{ {vals} }}\n"
            "  ?item a wikibase:Item ; (rdfs:label|skos:altLabel) ?lbl ; wikibase:sitelinks ?sitelinks .\n"
            "  FILTER(LANG(?lbl)='en')\n"
            "  FILTER(LCASE(STR(?lbl)) = LCASE(STR(?needle)))\n"
            "}\n"
        )

    best: Dict[str, Tuple[str, int]] = {}
    for data in _run_chunked(
        uniq, builder_exact, BATCH, stage_name="resolve items exact", position=1
    ):
        for b in data.get("results", {}).get("bindings", []):
            needle = b["needle"]["value"]
            qid = b["item"]["value"].split("/")[-1]
            sitelinks = int(b["sitelinks"]["value"])
            cur = best.get(needle)
            if cur is None or sitelinks > cur[1]:
                best[needle] = (qid, sitelinks)
    for lbl in uniq:
        if lbl in best:
            out[lbl] = best[lbl][0]

    # pass 2: mwapi fallback (one-by-one)
    unresolved = [x for x in uniq if not out.get(x)]
    for lbl in tqdm(
        unresolved,
        desc="resolve items mwapi",
        position=1,
        leave=False,
        dynamic_ncols=True,
    ):
        val = _esc_lit_single(lbl)
        q = f"""
SELECT ?item ?sitelinks WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:endpoint "www.wikidata.org" ;
                    wikibase:api "EntitySearch" ;
                    mwapi:search '{val}' ;
                    mwapi:language "en" ;
                    mwapi:type "item" .
    ?item wikibase:apiOutputItem mwapi:item .
  }}
  OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks }}
}} ORDER BY DESC(?sitelinks) LIMIT 1
"""
        try:
            data = _run_sparql(q)
            b = data.get("results", {}).get("bindings", [])
            if b:
                out[lbl] = b[0]["item"]["value"].split("/")[-1]
        except Exception:
            pass
    return out


def resolve_props_pids(labels: List[str]) -> Dict[str, Optional[str]]:
    """relation label -> PID (exact; fallback mwapi)."""
    out: Dict[str, Optional[str]] = {lbl: None for lbl in labels}
    uniq = sorted({str(x).strip() for x in labels if str(x).strip()})

    def builder_exact(chunk):
        vals = " ".join([f"'{_esc_lit_single(x)}'@en" for x in chunk])
        return (
            "SELECT ?needle ?prop WHERE {\n"
            f"  VALUES ?needle {{ {vals} }}\n"
            "  ?prop a wikibase:Property ; rdfs:label ?lbl .\n"
            "  FILTER(LANG(?lbl)='en')\n"
            "  FILTER(LCASE(STR(?lbl)) = LCASE(STR(?needle)))\n"
            "}\n"
        )

    for data in _run_chunked(
        uniq, builder_exact, BATCH, stage_name="resolve props exact", position=1
    ):
        for b in data.get("results", {}).get("bindings", []):
            needle = b["needle"]["value"]
            pid = b["prop"]["value"].split("/")[-1]
            out[needle] = pid

    unresolved = [x for x in uniq if not out.get(x)]
    for lbl in tqdm(
        unresolved,
        desc="resolve props mwapi",
        position=1,
        leave=False,
        dynamic_ncols=True,
    ):
        val = _esc_lit_single(lbl)
        q = f"""
SELECT ?prop WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:endpoint "www.wikidata.org" ;
                    wikibase:api "EntitySearch" ;
                    mwapi:search '{val}' ;
                    mwapi:language "en" ;
                    mwapi:type "property" .
    ?prop wikibase:apiOutputItem mwapi:item .
  }}
}} LIMIT 1
"""
        try:
            data = _run_sparql(q)
            b = data.get("results", {}).get("bindings", [])
            if b:
                out[lbl] = b[0]["prop"]["value"].split("/")[-1]
        except Exception:
            pass
    return out


# =====================
# Popularity
# =====================


def fetch_items_popularity(qids: List[str]) -> Dict[str, Tuple[int, int, int]]:
    out: Dict[str, Tuple[int, int, int]] = {}
    uniq = [q for q in sorted(set(qids)) if q]

    def builder(chunk):
        vals = " ".join([f"wd:{q}" for q in chunk])
        return (
            "SELECT ?item ?sitelinks ?identifiers ?statements WHERE {\n"
            f"  VALUES ?item {{ {vals} }}\n"
            "  ?item wikibase:sitelinks ?sitelinks ;\n"
            "        wikibase:identifiers ?identifiers ;\n"
            "        wikibase:statements ?statements .\n"
            "}\n"
        )

    for data in _run_chunked(
        uniq, builder, BATCH, stage_name="fetch item popularity", position=1
    ):
        for b in data.get("results", {}).get("bindings", []):
            qid = b["item"]["value"].split("/")[-1]
            out[qid] = (
                int(b["sitelinks"]["value"]),
                int(b["identifiers"]["value"]),
                int(b["statements"]["value"]),
            )
    return out


def fetch_triple_popularity_bulk(
    pid_obj_pairs: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    uniq = sorted(set([(p, o) for (p, o) in pid_obj_pairs if p and o]))

    def builder(chunk):
        vals = " ".join([f"(wd:{p} wd:{o})" for (p, o) in chunk])
        return (
            "SELECT ?prop ?obj (COUNT(?s) AS ?cnt) WHERE {\n"
            f"  VALUES (?prop ?obj) {{ {vals} }}\n"
            "  ?prop wikibase:directClaim ?p .\n"
            "  ?s ?p ?obj .\n"
            "} GROUP BY ?prop ?obj\n"
        )

    for data in _run_chunked(
        uniq, builder, BATCH, stage_name="fetch triple popularity", position=1
    ):
        for b in data.get("results", {}).get("bindings", []):
            p = b["prop"]["value"].split("/")[-1]
            o = b["obj"]["value"].split("/")[-1]
            out[(p, o)] = int(b["cnt"]["value"])
    for k in uniq:
        out.setdefault(k, 0)
    return out


def pop_score(sitelinks: int, identifiers: int, statements: int) -> float:
    return float(sitelinks + identifiers + statements)


# =====================
# QA generation
# =====================


def make_question(subject: str, relation: str) -> str:
    return f"What is the {relation} of {subject}?"


# =====================
# Core build
# =====================


def build_qa_from_csv(csv_path: str, inner_position: int = 2) -> pd.DataFrame:
    base = os.path.basename(csv_path)
    df = pd.read_csv(csv_path)
    if not REQUIRED_COLS.issubset(df.columns):
        missing = REQUIRED_COLS - set(df.columns)
        raise ValueError(f"{base} missing columns: {missing}")

    # announce phases to keep console alive
    tqdm.write(f"[{base}] resolving items/props…")

    subj_labels = sorted(set(df["subjectLabel"].fillna("").astype(str).str.strip()))
    obj_labels = sorted(set(df["objectLabel"].fillna("").astype(str).str.strip()))
    rel_labels = sorted(set(df["relation"].fillna("").astype(str).str.strip()))

    subj_map = resolve_items_qids(subj_labels)
    obj_map = resolve_items_qids(obj_labels)
    prop_map = resolve_props_pids(rel_labels)

    rows_resolved: List[Tuple[str, str, str, str, str, str]] = []
    for _, r in df.iterrows():
        s = str(r["subjectLabel"]).strip()
        p = str(r["relation"]).strip()
        o = str(r["objectLabel"]).strip()
        q_s = subj_map.get(s)
        q_o = obj_map.get(o)
        p_id = prop_map.get(p)
        if s and p and o and q_s and q_o and p_id:
            rows_resolved.append((s, p, o, q_s, p_id, q_o))

    if not rows_resolved:
        tqdm.write(f"[{base}] 0 rows after ID resolution")
        return pd.DataFrame(
            columns=[
                "source_csv",
                "subjectLabel",
                "relation",
                "objectLabel",
                "subject_qid",
                "property_pid",
                "object_qid",
                "question",
                "answer",
                "pop_subject",
                "pop_object",
                "pop_sum",
                "triple_pop",
            ]
        )

    tqdm.write(f"[{base}] fetching popularity…")

    all_qids = sorted({q for (_, _, _, q, _, qo) in rows_resolved for q in (q, qo)})
    pop_map = fetch_items_popularity(all_qids)
    pid_obj_pairs = [(p, qo) for (_, _, _, _, p, qo) in rows_resolved]
    triple_map = fetch_triple_popularity_bulk(pid_obj_pairs)

    out_rows: List[Dict] = []
    # INNER TQDM over rows: visible bar for per-file progress
    for s, p, o, q_s, p_id, q_o in tqdm(
        rows_resolved,
        desc=f"{base} rows",
        position=inner_position,
        leave=False,
        dynamic_ncols=True,
    ):
        sp = pop_map.get(q_s)
        op = pop_map.get(q_o)
        if not sp or not op:
            continue
        ps = pop_score(*sp)
        po = pop_score(*op)
        tri = int(triple_map.get((p_id, q_o), 0))
        out_rows.append(
            {
                "source_csv": base,
                "subjectLabel": s,
                "relation": p,
                "objectLabel": o,
                "subject_qid": q_s,
                "property_pid": p_id,
                "object_qid": q_o,
                "question": make_question(s, p),
                "answer": o,
                "pop_subject": ps,
                "pop_object": po,
                "pop_sum": ps + po,
                "triple_pop": tri,
            }
        )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "source_csv",
            "subjectLabel",
            "relation",
            "objectLabel",
            "subject_qid",
            "property_pid",
            "object_qid",
            "question",
            "answer",
            "pop_subject",
            "pop_object",
            "pop_sum",
            "triple_pop",
        ],
    )
    tqdm.write(f"[{base}] built {len(out_df)} rows")
    return out_df


# =====================
# Report/Save/Combine
# =====================


def extract_failed_from_report(path: str) -> List[str]:
    import re

    failed = set()
    name = os.path.basename(path).lower()
    if name.endswith((".txt", ".log")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m1 = re.search(
                    r"Built\s+0\s+QA\s+rows\s+from\s+([A-Za-z0-9_\-\.]+\.csv)", line
                )
                if m1:
                    failed.add(m1.group(1))
                    continue
                m2 = re.search(r"Skip\s+([A-Za-z0-9_\-\.]+\.csv)\s+due to error", line)
                if m2:
                    failed.add(m2.group(1))
                    continue
    elif name.endswith(".csv"):
        tmp = pd.read_csv(path)
        cols = {c.lower(): c for c in tmp.columns}
        name_col = cols.get("source_csv") or cols.get("file") or cols.get("filename")
        if name_col:
            if "rows" in cols:
                failed |= set(
                    tmp.loc[(tmp[cols["rows"]].fillna(0) == 0), name_col].astype(str)
                )
            elif "status" in cols:
                failed |= set(
                    tmp.loc[
                        tmp[cols["status"]]
                        .astype(str)
                        .str.lower()
                        .isin(["fail", "error", "failed"]),
                        name_col,
                    ].astype(str)
                )
            else:
                failed |= set(tmp[name_col].astype(str))
    else:
        tqdm.write(f"[WARN] Unknown report format: {path}")
    return sorted(failed)


def save_per_file(
    csv_in: str, out_dir: str, force: bool = False, inner_position: int = 2
) -> int:
    out_path = os.path.join(
        out_dir, os.path.basename(csv_in).replace(".csv", "__qa.csv")
    )
    # skip if exists and not empty, unless forced
    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        tqdm.write(
            f"[SKIP existing] {os.path.basename(csv_in)} -> {os.path.basename(out_path)}"
        )
        return 0

    df_out = build_qa_from_csv(csv_in, inner_position=inner_position)
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    tqdm.write(f"[SAVE] {out_path} ({len(df_out)} rows)")
    return len(df_out)


def combine_qa(out_dir: str) -> int:
    parts = sorted(glob.glob(os.path.join(out_dir, "*__qa.csv")))
    parts = [p for p in parts if os.path.basename(p) != "combined_all__qa.csv"]
    if not parts:
        tqdm.write("[COMBINE] nothing to combine")
        return 0
    df_all = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    key = [
        "source_csv",
        "subjectLabel",
        "relation",
        "objectLabel",
        "subject_qid",
        "property_pid",
        "object_qid",
    ]
    for k in key:
        if k not in df_all.columns:
            df_all[k] = None
    df_all = df_all.drop_duplicates(subset=key, keep="first")
    out_path = os.path.join(out_dir, "combined_all__qa.csv")
    df_all.to_csv(out_path, index=False, encoding="utf-8")
    tqdm.write(f"[COMBINE] {out_path} ({len(df_all)} rows)")
    return len(df_all)


# =====================
# CLI
# =====================


def main():
    ap = argparse.ArgumentParser(description="Triplet→QA (Wikidata-grounded, robust)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="Path to a single CSV file")
    g.add_argument("--in_dir", help="Path to a directory with CSV files")
    ap.add_argument(
        "--pattern", default="*.csv", help="Glob pattern for --in_dir (default: *.csv)"
    )
    ap.add_argument(
        "--out_dir", default=None, help="Output folder (default: qa_res next to input)"
    )
    ap.add_argument(
        "--report_failed",
        default=None,
        help="Reparse only files listed in report (.txt/.log/.csv)",
    )
    ap.add_argument(
        "--combine", action="store_true", help="Also build combined_all__qa.csv"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if per-file __qa.csv exists",
    )
    args = ap.parse_args()

    if args.file:
        out_dir = _ensure_outdir_for(args.file, args.out_dir)
        save_per_file(
            os.path.abspath(args.file), out_dir, force=args.force, inner_position=2
        )
        return

    # directory mode
    in_dir = os.path.abspath(args.in_dir)
    out_dir = _ensure_outdir_for(in_dir, args.out_dir)

    if args.report_failed:
        failed_names = extract_failed_from_report(args.report_failed)
        if not failed_names:
            tqdm.write(f"[INFO] No failed filenames in {args.report_failed}")
            return
        candidates = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(in_dir, args.pattern))
        }
        missing = [bn for bn in failed_names if bn not in candidates]
        if missing:
            tqdm.write(f"[WARN] Not found in {in_dir}: {missing}")
        files = [candidates[bn] for bn in failed_names if bn in candidates]
        tqdm.write(f"[INFO] Re-parsing by report: {len(files)} files")
    else:
        files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
        tqdm.write(f"[INFO] Parsing directory: {len(files)} files")

    total = 0
    # outer tqdm at position 0; inner bars use position=1..2
    for idx, f in enumerate(tqdm(files, desc="Files", position=0, dynamic_ncols=True)):
        try:
            # rotate inner position a bit in case of overlapping bars
            added = save_per_file(f, out_dir, force=args.force, inner_position=2)
            total += max(0, added)
        except Exception as e:
            tqdm.write(f"[SKIP] {os.path.basename(f)}: {e}")

    tqdm.write(f"[DONE] per-file saved for {len(files)} files, total rows={total}")

    if args.combine:
        combine_qa(out_dir)


if __name__ == "__main__":
    main()
