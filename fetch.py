# fetch.py
# -*- coding: utf-8 -*-
"""
One-shot: python fetch.py pandas.concat
→ writes ./pandas_concat_spec.json with a test-friendly spec.
"""

import sys, os, io, re, json, tarfile, tempfile, textwrap, urllib.request, ast, shutil

PYPI_JSON = "https://pypi.org/pypi/{pkg}/json"

# ---------------- HTTP ----------------
def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "func-spec/1.2"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

# -------------- PyPI sdist ------------
def pypi_get_sdist_url(pkg: str):
    data = json.loads(http_get(PYPI_JSON.format(pkg=pkg)).decode("utf-8"))
    version = data["info"]["version"]
    files = data["releases"].get(version) or []
    for f in files:
        if f.get("packagetype") == "sdist" and f.get("url", "").endswith((".tar.gz", ".tgz")):
            return version, f["url"]
    for f in data.get("urls", []):
        if f.get("packagetype") == "sdist":
            return version, f["url"]
    raise RuntimeError(f"No sdist found for {pkg} {version}")

def extract_sdist(sdist_bytes: bytes, workdir: str) -> str:
    fileobj = io.BytesIO(sdist_bytes)
    with tarfile.open(fileobj=fileobj, mode="r:gz") as tf:
        tf.extractall(path=workdir)
        top = None
        for m in tf.getmembers():
            parts = m.name.split("/")
            if parts and parts[0]:
                top = parts[0]; break
    if not top:
        dirs = [d for d in os.listdir(workdir) if os.path.isdir(os.path.join(workdir, d))]
        if not dirs: raise RuntimeError("Extraction failed")
        top = dirs[0]
    return os.path.join(workdir, top)

def find_package_root(top_dir: str, pkg: str) -> str:
    for root, _, files in os.walk(top_dir):
        if os.path.basename(root) == pkg and "__init__.py" in files:
            return root
    raise RuntimeError(f"Package {pkg} not found")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -------------- AST helpers -----------
def ast_unparse(node: ast.AST, source: str | None = None) -> str:
    try:
        return ast.unparse(node)  # 3.9+
    except Exception:
        # minimal fallback
        return ""

def module_name_from_path(pkg_root: str, file_path: str) -> str:
    rel = os.path.relpath(file_path, pkg_root)
    parts = [os.path.basename(pkg_root)] + rel.split(os.sep)
    if parts[-1].endswith(".py"): parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__": parts = parts[:-1]
    return ".".join(parts)

def build_candidate_index(pkg_root: str, func_name: str):
    cands = []
    for root, _, files in os.walk(pkg_root):
        for fn in files:
            if not fn.endswith(".py"): continue
            path = os.path.join(root, fn)
            try:
                src = read_text(path)
                mod = ast.parse(src, filename=path)
            except Exception:
                continue
            for node in mod.body:
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    cands.append({"module": module_name_from_path(pkg_root, path),
                                  "file": path, "node": node, "source": src})
    return cands

def is_stub_body(node: ast.FunctionDef) -> bool:
    if not node.body: return True
    if len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Expr)):
        if isinstance(node.body[0], ast.Pass): return True
        val = getattr(node.body[0], "value", None)
        if isinstance(val, ast.Constant) and val.value is Ellipsis: return True
    return False

def get_source_segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = getattr(node, "end_lineno", None) or len(lines)
    return textwrap.dedent("".join(lines[start:end]))

def strip_leading_docstring_from_body(full_src: str) -> str:
    body = textwrap.dedent(full_src)
    m = re.search(r'^\s*[ruRU]?["\']{3}.*?["\']{3}\s*\n', body, flags=re.S)
    return body[m.end():] if m else body

# -------- signature (with defaults) --------
def extract_signature(node: ast.FunctionDef, source: str):
    args = node.args
    params = []

    def ann(a): return ast_unparse(a, source) if a is not None else None
    def defv(n): return ast_unparse(n, source) if n is not None else None

    posonly = getattr(args, "posonlyargs", [])
    total_pos = len(posonly) + len(args.args)

    # pos-only
    for a in posonly:
        params.append({"name": a.arg, "kind": "POSITIONAL_ONLY",
                       "type": ann(a.annotation), "default": None})

    # positional-or-keyword (with defaults mapping)
    for i, a in enumerate(args.args):
        default = None
        if args.defaults:
            idx_from_end = total_pos - (len(posonly) + i)
            if idx_from_end <= len(args.defaults):
                default = defv(args.defaults[-idx_from_end])
        params.append({"name": a.arg, "kind": "POSITIONAL_OR_KEYWORD",
                       "type": ann(a.annotation), "default": default})

    # *args
    if args.vararg:
        params.append({"name": args.vararg.arg, "kind": "VAR_POSITIONAL",
                       "type": ann(args.vararg.annotation), "default": None})

    # keyword-only (with kw_defaults)
    for i, a in enumerate(args.kwonlyargs):
        default = None
        if args.kw_defaults and args.kw_defaults[i] is not None:
            default = defv(args.kw_defaults[i])
        params.append({"name": a.arg, "kind": "KEYWORD_ONLY",
                       "type": ann(a.annotation), "default": default})

    # **kwargs
    if args.kwarg:
        params.append({"name": args.kwarg.arg, "kind": "VAR_KEYWORD",
                       "type": ann(args.kwarg.annotation), "default": None})

    ret_type = ann(node.returns)

    # signature string (pretty)
    parts = []
    for p in params:
        nm = p["name"]
        if p["kind"] == "VAR_POSITIONAL": nm = "*" + nm
        elif p["kind"] == "VAR_KEYWORD": nm = "**" + nm
        if p["type"]: nm += f": {p['type']}"
        if p["default"] is not None: nm += f" = {p['default']}"
        parts.append(nm)
    sig = "(" + ", ".join(parts) + ")"
    if ret_type: sig += f" -> {ret_type}"
    return {"parameters": params, "return_type": ret_type, "signature": sig}

# --------- Docstring parsing (NumPy style) ----------
_NP_PARAM = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*:\s*(.+?)(?:,\s*default\s*(?:=|\s)\s*([^,\n]+))?\s*$"
)
_ENUM_BRACES = re.compile(r"\{([^}]+)\}")

def parse_numpy_sections(doc: str) -> dict:
    out = {k: "" for k in ("Parameters", "Returns", "Raises", "Notes", "Examples")}
    if not doc: return out
    lines = doc.splitlines()
    headers = ["Parameters", "Returns", "Raises", "Notes", "Examples"]
    marks = []
    for i, ln in enumerate(lines):
        if ln.strip() in headers:
            marks.append((ln.strip(), i))
    marks.sort(key=lambda x: x[1])
    for idx, (name, i) in enumerate(marks):
        start = i + 1
        if start < len(lines) and set(lines[start].strip()) <= {"-"}:
            start += 1
        end = marks[idx + 1][1] if idx + 1 < len(marks) else len(lines)
        out[name] = "\n".join(lines[start:end]).strip("\n")
    if "Examples" in out["Notes"]:
        out["Notes"] = out["Notes"].split("Examples", 1)[0].rstrip()
    return out

def _split_enum_items(raw: str):
    items = []
    for tok in re.split(r"\s*,\s*", raw):
        tok = tok.strip()
        if "/" in tok and not (tok.startswith("'") or tok.startswith('"')):
            left, right = [t.strip() for t in tok.split("/", 1)]
            try: items.append(int(left))
            except Exception: items.append(left)
            if (right.startswith("'") and right.endswith("'")) or (right.startswith('"') and right.endswith('"')):
                items.append(right[1:-1])
            else:
                items.append(right)
            continue
        if (tok.startswith("'") and tok.endswith("'")) or (tok.startswith('"') and tok.endswith('"')):
            items.append(tok[1:-1])
        else:
            try: items.append(int(tok))
            except Exception: items.append(tok)
    return items

def parse_parameters_block(block: str) -> dict:
    res = {}
    if not block: return res
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        m = _NP_PARAM.match(lines[i])
        if m:
            name, typ, default = m.group(1), m.group(2).strip(), (m.group(3).strip() if m.group(3) else None)
            desc_lines = []
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")) and not _NP_PARAM.match(lines[i]):
                desc_lines.append(lines[i].strip()); i += 1
            desc = " ".join(desc_lines).strip() or None
            enum = None
            em = _ENUM_BRACES.search(typ)
            if em: enum = _split_enum_items(em.group(1))
            clean_typ = _ENUM_BRACES.sub("", typ).strip()
            nullable = ("None" in typ) or (default is None and (desc or "").lower().find("default none") >= 0)
            res[name] = {
                "type": clean_typ or None,
                "default": default,
                "description": desc,
                "enum": enum,
                "nullable": bool(nullable),
            }
        else:
            i += 1
    return res

def extract_examples_code(block: str):
    if not block: return []
    out, buf, in_ex = [], [], False
    for ln in block.splitlines():
        t = ln.strip()
        if t.startswith(">>>"): in_ex = True; buf.append(t[4:])
        elif in_ex and t.startswith("..."): buf.append(t[4:])
        elif in_ex and t == "":
            if buf: out.append("\n".join(buf).strip()); buf = []
            in_ex = False
        else:
            if in_ex:
                if buf: out.append("\n".join(buf).strip()); buf = []
                in_ex = False
    if buf: out.append("\n".join(buf).strip())
    return out

# -------------- Orchestrator -------------
def summarize(symbol: str) -> dict:
    parts = symbol.split(".")
    if len(parts) < 2: raise ValueError("Use dotted name like 'pandas.concat'")
    pkg, func = parts[0], parts[-1]

    version, url = pypi_get_sdist_url(pkg)
    tmp = tempfile.mkdtemp(prefix=f"spec_{pkg}_")
    try:
        sdist = http_get(url)
        top = extract_sdist(sdist, tmp)
        pkg_root = find_package_root(top, pkg)
        cands = build_candidate_index(pkg_root, func)
        if not cands: raise RuntimeError(f"Function '{func}' not found in {pkg} {version}")

        # pick best: prefer non-stub + filename hint
        best, score_best = None, -1
        for c in cands:
            score = 0
            if os.path.basename(c["file"]).startswith(func): score += 2
            if not is_stub_body(c["node"]): score += 5
            if score > score_best: best, score_best = c, score

        module, file_path, src, node = best["module"], best["file"], best["source"], best["node"]

        sig_info = extract_signature(node, src)
        source_full = get_source_segment(src, node)  # full function block
        docstring = ast.get_docstring(node) or ""

        sections = parse_numpy_sections(docstring)
        params_map = parse_parameters_block(sections.get("Parameters", ""))

        # prefer signature types; fill defaults from signature/doc
        sig_types = {p["name"]: p["type"] for p in sig_info["parameters"] if p.get("type")}

        enriched = []
        for p in sig_info["parameters"]:
            info = params_map.get(p["name"], {})
            typ = sig_types.get(p["name"]) or info.get("type") or p["type"]
            # default precedence: signature default > doc default
            default = p["default"] if p["default"] is not None else info.get("default")
            required = (default is None) and (p["kind"] not in ("VAR_POSITIONAL","VAR_KEYWORD"))
            enriched.append({
                "name": p["name"],
                "kind": p["kind"],
                "type": typ,
                "default": default,
                "required": bool(required),
                "nullable": info.get("nullable", None),
                "enum": info.get("enum", None),
                "description": info.get("description", None),
            })

        returns_type = sig_info["return_type"] or None  # از امضا
        returns_desc = sections.get("Returns", "").strip() or None

        raises_list = [ln.strip("-• \t") for ln in sections.get("Raises", "").splitlines() if ln.strip()] or None
        if not raises_list:
            found = sorted(set(m.group(1) for m in re.finditer(r"\b([A-Z][A-Za-z]+Error)\b", docstring)))
            raises_list = found or None

        notes_list = [ln.strip() for ln in sections.get("Notes", "").splitlines() if ln.strip()] or None
        examples_code = extract_examples_code(sections.get("Examples", "")) or None

        body_no_doc = strip_leading_docstring_from_body(source_full)

        return {
            "package": pkg,
            "version": version,
            "qualified_name": f"{module}.{node.name}",
            "file": file_path,
            "signature": sig_info["signature"],
            "parameters": enriched,
            "returns": {"type": returns_type, "description": returns_desc},
            "raises": raises_list,
            "notes": notes_list,
            "examples_code": examples_code,
            "body_stripped": body_no_doc,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# --------------- CLI (single mode) ---------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch.py <package.function>\nExample: python fetch.py pandas.concat", file=sys.stderr)
        sys.exit(1)
    symbol = sys.argv[1]
    try:
        spec = summarize(symbol)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(2)

    out_name = symbol.replace(".", "_") + ".json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_name}")