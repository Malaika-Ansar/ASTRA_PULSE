#!/usr/bin/env python3
"""
ASTRA PULSE backend - checks what you enter against NASA-sourced predictors
(astra_nasa_predictors.json) AND against your own previous 5-7 days.

Core engine + SQLite use only the standard library.
HTTP: `serve` (stdlib server, no install) or `create_app()` (FastAPI, optional).

  python astra_backend.py check  --group crew --phase inflight --mission-day 40 \
         --set sleep_hours=5.1 --set co2_mmHg=2.7 --subject me
  python astra_backend.py record --group crew ...      # same, but stores the reading
  python astra_backend.py history --subject me --group crew --days 14
  python astra_backend.py serve --port 8000
  python astra_backend.py selftest

Not medical advice. Cited thresholds describe what NASA-linked studies found;
z-score cut-offs and 'actions' are this app's conventions (see the JSON meta).
"""
import argparse, datetime as dt, json, math, os, sqlite3, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_PATH = Path(os.environ.get("ASTRA_REFERENCE", HERE / "astra_nasa_predictors.json"))
DB_PATH = os.environ.get("ASTRA_DB", str(HERE / "astra_readings.db"))
RANK = {"no_reference": -1, "ok": 0, "watch": 1, "act": 2}
DISCLAIMER = ("Not medical advice. Findings describe NASA-linked study results; z-score cut-offs and actions are "
              "this app's conventions. For real symptoms contact a clinician / flight surgeon.")

REF = json.loads(REF_PATH.read_text(encoding="utf-8"))
STATS = REF["meta"]["statistics"]
BASE = REF["meta"]["personal_baseline"]


# ------------------------------------------------------------------ helpers
def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f"):
            return False
    return None

def _worse(a, b):
    return a if RANK[a] >= RANK[b] else b

def _test(op, ref, x):
    if op == "lt": return x < ref
    if op == "le": return x <= ref
    if op == "gt": return x > ref
    if op == "ge": return x >= ref
    if op == "eq": return x == ref
    if op == "between": return ref[0] <= x < ref[1]
    raise ValueError(op)

def _when_ok(when, ctx):
    if not when:
        return True
    if "phase" in when and ctx["phase"] not in when["phase"]: return False
    if "posture" in when and ctx["posture"] not in when["posture"]: return False
    md = ctx.get("mission_day")
    if "mission_day_min" in when and (md is None or md < when["mission_day_min"]): return False
    if "mission_day_max" in when and (md is None or md > when["mission_day_max"]): return False
    dl = ctx.get("days_since_landing")
    if "days_since_landing_min" in when and (dl is None or dl < when["days_since_landing_min"]): return False
    if "days_since_landing_max" in when and (dl is None or dl > when["days_since_landing_max"]): return False
    return True

def _cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def _status_from_z(z, bad):
    a = abs(z)
    if bad == "high" and z < 0: return "ok"
    if bad == "low" and z > 0: return "ok"
    if a >= STATS["z_act"]: return "act"
    if a >= STATS["z_watch"]: return "watch"
    return "ok"


# ------------------------------------------------------------------ storage
def connect(path=None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS readings(
        id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id TEXT NOT NULL, grp TEXT NOT NULL,
        date TEXT NOT NULL, phase TEXT, posture TEXT, mission_day INTEGER, days_since_landing INTEGER,
        values_json TEXT NOT NULL, context_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_r ON readings(subject_id, grp, date)")
    return conn

def _daily_means(conn, subject, grp, field, start, end, phase=None):
    """{date: mean of that field} for start <= date <= end (ISO strings)."""
    q = "SELECT date, phase, values_json FROM readings WHERE subject_id=? AND grp=? AND date>=? AND date<=?"
    out = {}
    for r in conn.execute(q, (subject, grp, start, end)):
        if phase and r["phase"] != phase:
            continue
        v = json.loads(r["values_json"]).get(field)
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            out.setdefault(r["date"], []).append(float(v))
    return {d: sum(v) / len(v) for d, v in out.items()}

def save_reading(conn, req, values):
    conn.execute("""INSERT INTO readings(subject_id,grp,date,phase,posture,mission_day,days_since_landing,values_json,context_json)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                 (req["subject_id"], req["group"], req["date"], req["phase"], req.get("posture"),
                  req.get("mission_day"), req.get("days_since_landing"),
                  json.dumps(values), json.dumps(req.get("context") or {})))
    conn.commit()

def history(conn, subject, grp, days=14, end=None):
    end = end or dt.date.today().isoformat()
    start = (dt.date.fromisoformat(end) - dt.timedelta(days=days)).isoformat()
    rows = conn.execute("SELECT * FROM readings WHERE subject_id=? AND grp=? AND date>=? AND date<=? ORDER BY date,id",
                        (subject, grp, start, end)).fetchall()
    return [{"id": r["id"], "date": r["date"], "phase": r["phase"], "posture": r["posture"],
             "mission_day": r["mission_day"], "values": json.loads(r["values_json"])} for r in rows]


# ------------------------------------------------------------------ engine
def _alias_map(group_spec):
    m = {}
    for name, spec in group_spec["indicators"].items():
        m[name.lower()] = name
        for a in spec.get("aliases", []):
            m[a.lower()] = name
    return m

def _reason(kind, status, message, sources=None, action=None, **extra):
    d = {"type": kind, "status": status, "message": message, "sources": sources or []}
    if action: d["action"] = action
    d.update(extra)
    return d

def _ref_lookup(spec, ctx):
    refs = spec.get("reference")
    if not refs:
        return None, None
    phase = ctx["phase"]
    pk = phase
    if phase == "postflight":
        dl = ctx.get("days_since_landing")
        if dl is not None and 7 <= dl <= 10:
            pk = "postflight_d7_10"
        else:
            return None, "postflight_gap"
    for key in (f"{pk}|{ctx.get('posture')}", f"{pk}|any"):
        if key in refs:
            return refs[key], key
    if any(k.startswith(pk + "|") for k in refs):
        return None, "need_posture"
    return None, None

def _eval_number(name, spec, x, ctx, conn, req, flags):
    reasons = []
    # 1) cited tier rules (worst matching tier wins)
    best = None
    for r in spec.get("cited_rules", []):
        if _when_ok(r.get("when"), ctx) and _test(r["test"]["op"], r["test"]["value"], x):
            if best is None or RANK[r["status"]] > RANK[best["status"]]:
                best = r
    if best:
        reasons.append(_reason("nasa_rule", best["status"], best["message"], best["sources"], best.get("action"), rule=best["id"]))
    # 2) NASA reference table -> z-score
    ref, key = _ref_lookup(spec, ctx)
    if ref:
        sd = ref.get("sd") or (ref["sem"] * math.sqrt(ref["n"]) if ref.get("sem") and ref.get("n") else None)
        if sd:
            z = (x - ref["mean"]) / sd
            lo, hi = ref["mean"] - 1.96 * sd, ref["mean"] + 1.96 * sd
            msg = (f"NASA reference - {ref['label']}: mean {ref['mean']:g}, SD about {sd:.1f} (95% range {lo:.0f}-{hi:.0f}). "
                   f"Your value {x:g} is z={z:+.2f} (about the {_cdf(z)*100:.0f}th percentile).")
            st = "ok" if ref.get("info_only") else _status_from_z(z, ref.get("bad", "both"))
            if ref.get("info_only"):
                msg += " (Reference is preliminary/information only, so it is not scored.)"
            reasons.append(_reason("nasa_reference", st, msg, ref["sources"], z=round(z, 2), percentile=round(_cdf(z) * 100, 1)))
        elif ref.get("info_only"):
            reasons.append(_reason("nasa_reference", "ok", f"NASA reference - {ref['label']}: about {ref['mean']:g}. Information only.", ref["sources"]))
        # nearest group on landing day
        cg = spec.get("context_groups", {}).get(f"{ctx['phase']}|{ctx.get('posture')}")
        if cg:
            best_g = None
            for g in cg:
                gsd = g["sem"] * math.sqrt(g["n"])
                gz = abs(x - g["mean"]) / gsd
                if best_g is None or gz < best_g[1]:
                    best_g = (g, gz)
            g = best_g[0]
            reasons.append(_reason("nasa_context", "ok",
                f"Landing-day stand-test groups in EDOMP (Table 1-6): your value is closest to the '{g['label']}' average ({g['mean']:g}). "
                "This is a description of which group you resemble, not a diagnosis.",
                ["edomp_cv"], action=spec.get("context_action")))
    else:
        if key == "postflight_gap":
            reasons.append(_reason("note", "no_reference", "No NASA numeric reference for postflight days 1-6 (NASA measured landing day and days 7-10).", ["edomp_cv"]))
        elif key == "need_posture":
            reasons.append(_reason("note", "no_reference", "A NASA reference exists for this phase, but it is posture-specific: add posture='supine' or 'standing'.", ["edomp_cv"]))
    # 3) delta from own preflight (core temperature)
    d = spec.get("delta_from_phase")
    if d and ctx["phase"] in d["applies_to_phase"]:
        pre = _daily_means(conn, req["subject_id"], req["group"], name, "0000-01-01", req["date"], phase=d["baseline_phase"])
        if len(pre) >= d["min_days"]:
            delta = x - statistics.mean(pre.values())
            k = "none" if delta < 0.5 else ("typical" if delta <= d["expected_delta"] + 0.5 else "above")
            reasons.append(_reason("nasa_delta", "ok", d["text"][k].format(delta=delta), d["sources"], delta=round(delta, 2)))
        else:
            reasons.append(_reason("note", "no_reference", f"Enter at least {d['min_days']} preflight days to compare your in-flight rise with the +1.01 C ISS mean.", d["sources"]))
    # 4) phase notes
    pn = spec.get("phase_notes", {})
    for k in (ctx["phase"], "all"):
        if k in pn:
            reasons.append(_reason("note", "ok", pn[k], spec.get("extra_sources", []) + spec.get("reference_sources", [])))
    if spec.get("reference_info") and ctx["phase"] in ("landing_day", "preflight"):
        reasons.append(_reason("note", "ok", spec["reference_info"], spec.get("reference_sources", [])))
    ln = spec.get("late_mission_note")
    if ln and ctx.get("mission_day") and ctx["mission_day"] >= ln["mission_day_min"]:
        reasons.append(_reason("note", "ok", ln["message"], ln["sources"]))
    # 5) personal baseline over previous days
    bad = spec.get("baseline_bad", "none")
    if bad != "none":
        win = spec.get("baseline_window_days", BASE["window_days"])
        need = spec.get("baseline_min_days", BASE["min_days"])
        end = (dt.date.fromisoformat(req["date"]) - dt.timedelta(days=1)).isoformat()
        start = (dt.date.fromisoformat(req["date"]) - dt.timedelta(days=win)).isoformat()
        prior = _daily_means(conn, req["subject_id"], req["group"], name, start, end, phase=ctx["phase"] if BASE["same_phase_only"] else None)
        if len(prior) >= need:
            vals = list(prior.values())
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            floor = spec.get("baseline_min_sd") or abs(mean) * BASE["fallback_min_sd_fraction"]
            sd_eff = max(sd, floor, 1e-9)
            z = (x - mean) / sd_eff
            st = _status_from_z(z, bad)
            side = "above" if z > 0 else "below"
            msg = (f"Your own last {len(vals)} days ({ctx['phase']}): mean {mean:.2f}, SD {sd:.2f} (floor {floor:g}). "
                   f"Today's {x:g} is z={z:+.2f}, {side} your usual.")
            if st == "ok":
                # slow drift: 3 consecutive worsening days ending today
                ordered = [prior[k] for k in sorted(prior)][-3:] + [x]
                step = (lambda a, b: b > a) if bad in ("high",) else (lambda a, b: b < a)
                if bad in ("high", "low") and len(ordered) == 4 and all(step(ordered[i], ordered[i + 1]) for i in range(3)) \
                        and abs(ordered[-1] - ordered[0]) >= sd_eff:
                    st = "watch"
                    msg += " It has also moved the wrong way for 3 days in a row (drift)."
            if st != "ok":
                flags.add(name)
            reasons.append(_reason("personal_baseline", st, msg, [], baseline={"days": len(vals), "mean": round(mean, 3), "sd": round(sd, 3), "z": round(z, 2)}))
        else:
            reasons.append(_reason("note", "no_reference", f"Personal baseline not ready: {len(prior)} of {need} previous days available in this phase.", []))
    # 6) multi-day window rules (include today)
    for w in spec.get("window_rules", []):
        start = (dt.date.fromisoformat(req["date"]) - dt.timedelta(days=w["days"] - 1)).isoformat()
        end = (dt.date.fromisoformat(req["date"]) - dt.timedelta(days=1)).isoformat()
        prior = _daily_means(conn, req["subject_id"], req["group"], name, start, end)
        vals = list(prior.values()) + [x]
        if len(vals) >= w["min_days"]:
            stat = statistics.mean(vals) if w["stat"] == "mean" else sum(1 for v in vals if v >= 0.5)
            if _test(w["op"], w["value"], stat):
                reasons.append(_reason("window_rule", w["status"], w["message"] + f" (value {stat:.2f} over {len(vals)} days)", w["sources"], w.get("action"), rule=w["id"]))
    # 7) uncited defaults (notice only unless enabled)
    return reasons

def _eval_bool(name, spec, v, ctx, conn, req, flags):
    reasons = []
    for r in spec.get("cited_rules", []):
        if _when_ok(r.get("when"), ctx) and _test(r["test"]["op"], r["test"]["value"], v):
            reasons.append(_reason("nasa_rule", r["status"], r["message"], r["sources"], r.get("action"), rule=r["id"]))
    pt = spec.get("label_passthrough")
    if pt and str(v).lower() in pt:
        reasons.append(_reason("your_label", pt[str(v).lower()], "Passed through from your own label (" + name + "); not a NASA rule."))
    if pt is None or True:
        for w in spec.get("window_rules", []):
            start = (dt.date.fromisoformat(req["date"]) - dt.timedelta(days=w["days"] - 1)).isoformat()
            end = (dt.date.fromisoformat(req["date"]) - dt.timedelta(days=1)).isoformat()
            prior = _daily_means(conn, req["subject_id"], req["group"], name, start, end)
            vals = list(prior.values()) + [1.0 if v else 0.0]
            if len(vals) >= w["min_days"] and w["stat"] == "count_true" and _test(w["op"], w["value"], sum(1 for q in vals if q >= 0.5)):
                reasons.append(_reason("window_rule", w["status"], w["message"], w["sources"], w.get("action"), rule=w["id"]))
    if v and name in ("circadian_misaligned",):
        flags.add(name)
    return reasons

def _uncited(spec, x, include):
    notes, reasons = [], []
    for r in spec.get("uncited_defaults", []):
        if _test(r["op"], r["value"], x):
            txt = f"{r['message']} [{r['basis']}]"
            if include:
                reasons.append(_reason("uncited_default", r["status"], txt))
            else:
                notes.append(f"Generic safety notice ({r['status']}): {txt} Not counted in the NASA-based status; set include_uncited_defaults=true to count it.")
    return reasons, notes

def _compound(group_spec, findings, flags):
    out = []
    by = {f["indicator"]: f for f in findings}
    def ok_req(req):
        if "any_of" in req:
            return any(ok_req(r) for r in req["any_of"])
        f = by.get(req["indicator"])
        if not f:
            return False
        if "status_in" in req and f["status"] not in req["status_in"]:
            return False
        if req.get("baseline_bad") and req["indicator"] not in flags:
            return False
        return True
    for c in group_spec.get("compound_rules", []):
        if all(ok_req(r) for r in c["requires"]):
            out.append(_reason("compound", c["status"], c["message"], c["sources"], c.get("action"), rule=c["id"]))
    return out

def normalize_request(req):
    r = dict(req)
    r.setdefault("subject_id", "me")
    r["date"] = r.get("date") or dt.date.today().isoformat()
    dt.date.fromisoformat(r["date"])  # validates
    r["phase"] = (r.get("phase") or "inflight").lower()
    if r.get("posture"):
        r["posture"] = r["posture"].lower()
    if r.get("hours_since_transition") is not None and r.get("mission_day") is None:
        r["mission_day"] = int(float(r["hours_since_transition"]) // 24) + 1
    for k in ("mission_day", "days_since_landing"):
        if r.get(k) is not None:
            r[k] = int(r[k])
    if r.get("group") not in REF["groups"]:
        raise ValueError(f"group must be one of {list(REF['groups'])}")
    if not isinstance(r.get("values"), dict) or not r["values"]:
        raise ValueError("'values' must be a non-empty object, e.g. {\"sleep_hours\": 5.4}")
    return r

def check(req, conn=None, store=False):
    req = normalize_request(req)
    own = conn is None
    conn = conn or connect()
    try:
        gspec = REF["groups"][req["group"]]
        amap = _alias_map(gspec)
        ctx = {k: req.get(k) for k in ("phase", "posture", "mission_day", "days_since_landing")}
        include = bool(req.get("include_uncited_defaults"))
        findings, unrec, notices, flags, stored = [], [], [], set(), {}
        used_src = set()
        for raw_key, raw in req["values"].items():
            name = amap.get(str(raw_key).lower())
            if not name:
                unrec.append(raw_key)
                continue
            spec = gspec["indicators"][name]
            typ = spec.get("type", "number")
            f = {"indicator": name, "label": spec["label"], "unit": spec.get("unit"), "reasons": []}
            if typ == "bool":
                v = _bool(raw)
                if v is None:
                    f.update(value=raw, status="watch"); f["reasons"].append(_reason("input_check", "watch", f"Could not read '{raw}' as yes/no."))
                    findings.append(f); continue
                f["value"] = v; stored[name] = v
                f["reasons"] += _eval_bool(name, spec, v, ctx, conn, req, flags)
            else:
                x = _num(raw)
                if x is None:
                    f.update(value=raw, status="watch"); f["reasons"].append(_reason("input_check", "watch", f"'{raw}' is not a number."))
                    findings.append(f); continue
                f["value"] = x
                lo, hi = spec.get("sanity_range", (-1e9, 1e9))
                if not lo <= x <= hi:
                    f["status"] = "watch"
                    f["reasons"].append(_reason("input_check", "watch",
                        f"{x:g} {spec.get('unit','')} is outside the plausible range {lo:g}-{hi:g}; check units or typing. Not stored or scored."))
                    findings.append(f); continue
                stored[name] = x
                if spec.get("label_passthrough"):
                    st = spec["label_passthrough"].get(str(int(x)))
                    if st:
                        f["reasons"].append(_reason("your_label", st, "Passed through from your own label; not a NASA rule."))
                f["reasons"] += _eval_number(name, spec, x, ctx, conn, req, flags)
                ur, un = _uncited(spec, x, include)
                f["reasons"] += ur
                notices += un
                if spec.get("no_nasa_reference") and not any(r["type"] in ("nasa_rule", "nasa_reference") for r in f["reasons"]):
                    f["reasons"].append(_reason("note", "ok", spec["no_nasa_reference"]))
            sts = [r["status"] for r in f["reasons"] if r["status"] in ("ok", "watch", "act")]
            f["status"] = max(sts, key=lambda s: RANK[s]) if sts else "no_reference"
            findings.append(f)
        comp = _compound(gspec, findings, flags)
        # group-level context notes (motion sickness)
        cnotes = []
        for n in gspec.get("context_notes", []):
            w = n["when"]
            c = req.get("context") or {}
            if w.get("always") or all(_bool(c.get(k)) is v if isinstance(v, bool) else c.get(k) == v for k, v in w.items()):
                cnotes.append({"message": n["message"], "sources": n["sources"]})
        for f in findings:
            for r in f["reasons"]:
                used_src.update(r.get("sources", []))
        for r in comp:
            used_src.update(r["sources"])
        for n in cnotes:
            used_src.update(n["sources"])
        overall = "ok"
        for f in findings:
            overall = _worse(overall, f["status"]) if f["status"] != "no_reference" else overall
        for r in comp:
            overall = _worse(overall, r["status"])
        problems, todo = [], []
        for f in findings:
            for r in f["reasons"]:
                if r["status"] in ("watch", "act"):
                    problems.append({"indicator": f["indicator"], "status": r["status"], "what": r["message"]})
                    if r.get("action") and r["action"] not in todo:
                        todo.append(r["action"])
        for r in comp:
            problems.append({"indicator": "combined", "status": r["status"], "what": r["message"]})
            if r.get("action") and r["action"] not in todo:
                todo.append(r["action"])
        problems.sort(key=lambda p: -RANK[p["status"]])
        if store and stored:
            save_reading(conn, req, stored)
        return {
            "subject_id": req["subject_id"], "group": req["group"], "date": req["date"], "phase": req["phase"],
            "posture": req.get("posture"), "mission_day": req.get("mission_day"), "stored": bool(store and stored),
            "overall_status": overall,
            "summary": ("Nothing outside the NASA-based and personal-baseline checks." if not problems else
                        f"{len(problems)} item(s) need attention; worst is '{overall}'."),
            "whats_wrong": problems, "what_to_do": todo, "findings": findings,
            "compound_findings": comp, "context_notes": cnotes, "notices": notices,
            "unrecognized_fields": unrec,
            "sources": {k: {kk: REF["sources"][k][kk] for kk in ("citation", "url", "nasa_link", "verified")} for k in sorted(used_src) if k in REF["sources"]},
            "disclaimer": DISCLAIMER}
    finally:
        if own:
            conn.close()


# ------------------------------------------------------------------ HTTP
def _handle(method, path, query, body):
    """Shared router for stdlib server and FastAPI wrapper. Returns (status, json)."""
    try:
        if method == "GET" and path == "/reference":
            return 200, REF
        if method == "GET" and path == "/indicators":
            return 200, {g: {n: {"label": s["label"], "unit": s.get("unit"), "aliases": s.get("aliases", [])}
                             for n, s in v["indicators"].items()} for g, v in REF["groups"].items()}
        if method == "GET" and path.startswith("/subjects/") and path.endswith("/readings"):
            sid = path.split("/")[2]
            conn = connect()
            try:
                return 200, history(conn, sid, query.get("group", "crew"), int(query.get("days", 14)), query.get("end"))
            finally:
                conn.close()
        if method == "POST" and path in ("/check", "/record"):
            return 200, check(body, store=(path == "/record"))
        return 404, {"error": "not found", "routes": ["POST /check", "POST /record", "GET /subjects/{id}/readings?group=&days=", "GET /reference", "GET /indicators"]}
    except (ValueError, KeyError) as e:
        return 400, {"error": str(e)}

def serve(port=8000, host="127.0.0.1"):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs

    class H(BaseHTTPRequestHandler):
        def _go(self, method):
            u = urlparse(self.path)
            body = None
            if method == "POST":
                n = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid JSON"})
            code, data = _handle(method, u.path, {k: v[0] for k, v in parse_qs(u.query).items()}, body)
            self._send(code, data)
        def _send(self, code, data):
            b = json.dumps(data, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        def do_GET(self): self._go("GET")
        def do_POST(self): self._go("POST")
        def do_OPTIONS(self): self._send(204, {})
        def log_message(self, *a): pass
    print(f"ASTRA backend on http://{host}:{port}")
    HTTPServer((host, port), H).serve_forever()

def create_app():
    """Optional FastAPI wrapper (pip install fastapi uvicorn). Same logic as `serve`."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI(title="ASTRA PULSE NASA-reference backend")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    def out(res):
        code, data = res
        if code != 200:
            raise HTTPException(code, data)
        return data
    @app.post("/check")
    def _check(body: dict): return out(_handle("POST", "/check", {}, body))
    @app.post("/record")
    def _record(body: dict): return out(_handle("POST", "/record", {}, body))
    @app.get("/subjects/{sid}/readings")
    def _hist(sid: str, group: str = "crew", days: int = 14): return out(_handle("GET", f"/subjects/{sid}/readings", {"group": group, "days": days}, None))
    @app.get("/reference")
    def _ref(): return REF
    @app.get("/indicators")
    def _ind(): return out(_handle("GET", "/indicators", {}, None))
    return app


# ------------------------------------------------------------------ CLI / selftest
def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out

def _print(res):
    print(f"\n[{res['overall_status'].upper()}] {res['summary']}  ({res['group']}, {res['phase']}, {res['date']})")
    for p in res["whats_wrong"]:
        print(f"  - {p['status'].upper():5} {p['indicator']}: {p['what']}")
    if res["what_to_do"]:
        print("What to do:")
        for a in res["what_to_do"]:
            print("  *", a)
    for n in res["notices"]:
        print("  !", n)
    if res["unrecognized_fields"]:
        print("  unrecognized:", res["unrecognized_fields"])
    print("Sources:", ", ".join(res["sources"]))
    print(res["disclaimer"])

def selftest():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    conn = connect(path)
    d0 = dt.date(2026, 11, 1)
    def day(i): return (d0 + dt.timedelta(days=i)).isoformat()
    def run(vals, i, **kw):
        base = dict(group="crew", subject_id="t", date=day(i), phase="inflight", values=vals)
        base.update(kw)
        return check(base, conn=conn, store=True)
    # 1) cited rules without history
    r = run({"co2_mmHg": 3.4, "sleep_hours": 5.2, "hr": 60}, 0, mission_day=40)
    st = {f["indicator"]: f["status"] for f in r["findings"]}
    assert st["co2_mmHg"] == "act" and st["sleep_hours"] == "watch", st
    assert any("baseline not ready" in x["message"] for f in r["findings"] for x in f["reasons"])
    # 2) seed history, then a bad day -> baseline + window + compound
    for i, (s, rt, lap) in enumerate([(7.2, 250, 1), (7.0, 252, 1), (7.4, 248, 2), (7.1, 251, 1), (7.3, 249, 1), (7.2, 250, 1)], start=1):
        run({"sleep_hours": s, "reaction_time_ms": rt, "pvt_lapses": lap, "co2_mmHg": 1.8}, i)
    r = run({"sleep_hours": 4.9, "reaction_time_ms": 310, "co2_mmHg": 2.6}, 8)
    st = {f["indicator"]: f["status"] for f in r["findings"]}
    assert st["sleep_hours"] == "act" and st["reaction_time_ms"] == "act" and st["co2_mmHg"] == "watch", st
    assert any(c["rule"] == "short_sleep_plus_performance" for c in r["compound_findings"]), r["compound_findings"]
    assert r["overall_status"] == "act"
    # 3) EDOMP reference: landing day, standing HR 114 / SBP 80 -> nearest presyncopal group
    r = check(dict(group="crew", subject_id="t2", date=day(10), phase="landing_day", posture="standing",
                   values={"resting_hr_bpm": 114, "systolic_mmHg": 80}), conn=conn)
    msgs = " ".join(x["message"] for f in r["findings"] for x in f["reasons"])
    assert "presyncopal on landing day" in msgs and "z=" in msgs
    # posture missing -> asks for it
    r = check(dict(group="crew", subject_id="t2", date=day(10), phase="preflight", values={"resting_hr_bpm": 60}), conn=conn)
    assert any("posture" in x["message"] for f in r["findings"] for x in f["reasons"])
    # 4) motion sickness timing
    r = check(dict(group="motion_sickness", subject_id="m", date=day(3), phase="inflight", mission_day=5,
                   values={"severity": 2, "sick": "yes"}, context={"first_flight": True}), conn=conn)
    assert r["overall_status"] == "act" and r["context_notes"], r
    r = check(dict(group="motion_sickness", subject_id="m", date=day(3), phase="inflight", mission_day=1, values={"severity": 1}), conn=conn)
    assert r["overall_status"] == "ok"
    # 5) body temperature vs own preflight
    for i in range(4):
        check(dict(group="body_temperature", subject_id="b", date=day(i), phase="preflight", values={"core_temp_rest_c": 36.8 + 0.05 * i}), conn=conn, store=True)
    r = check(dict(group="body_temperature", subject_id="b", date=day(30), phase="inflight",
                   values={"core_temp_rest_c": 37.9, "core_temp_exercise_peak_c": 40.2, "fever": True}), conn=conn)
    msgs = " ".join(x["message"] for f in r["findings"] for x in f["reasons"])
    assert "preflight mean" in msgs and r["overall_status"] == "watch", r["overall_status"]
    # 6) SpO2 uncited default only a notice
    r = check(dict(group="crew", subject_id="t3", date=day(1), phase="inflight", values={"spo2": 92}), conn=conn)
    assert r["overall_status"] == "ok" and r["notices"]
    # 7) bad input
    r = check(dict(group="crew", subject_id="t3", date=day(1), values={"sleep_hours": 99, "foo": 1}), conn=conn)
    assert r["unrecognized_fields"] == ["foo"] and r["findings"][0]["status"] == "watch"
    for bad in ({"group": "x", "values": {"a": 1}}, {"group": "crew", "values": {}}):
        try:
            check(bad, conn=conn); raise SystemExit("should fail")
        except ValueError:
            pass
    # 8) history
    assert len(history(conn, "t", "crew", 30, day(20))) >= 8
    print("selftest: all checks passed")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "record"):
        p = sub.add_parser(name)
        p.add_argument("--group", required=True, choices=list(REF["groups"]))
        p.add_argument("--subject", default="me")
        p.add_argument("--date")
        p.add_argument("--phase", default="inflight")
        p.add_argument("--posture")
        p.add_argument("--mission-day", type=int)
        p.add_argument("--days-since-landing", type=int)
        p.add_argument("--set", action="append", help="indicator=value (repeatable)")
        p.add_argument("--ctx", action="append", help="context key=value, e.g. first_flight=true")
        p.add_argument("--include-uncited-defaults", action="store_true")
        p.add_argument("--json", action="store_true")
    h = sub.add_parser("history"); h.add_argument("--subject", default="me"); h.add_argument("--group", default="crew"); h.add_argument("--days", type=int, default=14)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.add_argument("--host", default="127.0.0.1")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest": return selftest()
    if a.cmd == "serve": return serve(a.port, a.host)
    if a.cmd == "history":
        conn = connect(); print(json.dumps(history(conn, a.subject, a.group, a.days), indent=1)); return
    req = {"group": a.group, "subject_id": a.subject, "date": a.date, "phase": a.phase, "posture": a.posture,
           "mission_day": a.mission_day, "days_since_landing": a.days_since_landing,
           "values": _kv(a.set), "context": _kv(a.ctx), "include_uncited_defaults": a.include_uncited_defaults}
    res = check(req, store=(a.cmd == "record"))
    print(json.dumps(res, indent=1) if a.json else "", end="")
    if not a.json:
        _print(res)

if __name__ == "__main__":
    main()
