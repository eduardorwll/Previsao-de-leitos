"""Verificador mecânico do Floor de CONSTRAINTS.md (C-04).

Detecta padrões proibidos em código de produção/testes:
- Supressão global de warnings: warnings.filterwarnings("ignore")
- Blocos except vazios silenciosos (except ...: pass)
- Diretivas de supressão sem código/motivo (# noqa, # type: ignore, nosemgrep, gitleaks:allow)
- Deserialização insegura (allow_pickle=True)
- Interpolação f-string suspeita de SQL em src/
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_SUPPRESSION_RE = re.compile(
    r"#\s*(noqa\s*$|type:\s*ignore\s*$|nosemgrep|gitleaks:allow)", re.IGNORECASE
)
_SQL_KEYWORDS_RE = re.compile(r"\b(SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)


def check_file(path: Path) -> list[str]:
    """Audita um arquivo Python e retorna lista de violações formatadas."""
    violations: list[str] = []
    source = path.read_text(encoding="utf-8")

    for lineno, line in enumerate(source.splitlines(), start=1):
        if _SUPPRESSION_RE.search(line):
            violations.append(
                f"{path}:{lineno}: supressão proibida pelo Floor de CONSTRAINTS.md ({line.strip()})"
            )

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: SyntaxError: {exc.msg}"]

    is_src = "src" in path.parts
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _is_ignore_warnings_call(node):
                violations.append(
                    f"{path}:{node.lineno}: proibido warnings.filterwarnings('ignore') global"
                )
            for kw in node.keywords:
                if (
                    kw.arg == "allow_pickle"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    violations.append(f"{path}:{node.lineno}: proibido allow_pickle=True")

        if (
            isinstance(node, ast.ExceptHandler)
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
        ):
            violations.append(
                f"{path}:{node.lineno}: bloco 'except: pass' silencioso proibido pelo Floor"
            )

        if is_src and isinstance(node, ast.JoinedStr):
            literal_text = "".join(
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            if len(_SQL_KEYWORDS_RE.findall(literal_text)) >= 2:
                violations.append(
                    f"{path}:{node.lineno}: SQL via f-string proibido em src/ "
                    "— use queries parametrizadas"
                )

    return violations


def _is_ignore_warnings_call(node: ast.Call) -> bool:
    func = node.func
    is_filterwarnings = (
        isinstance(func, ast.Attribute) and func.attr == "filterwarnings"
    ) or (isinstance(func, ast.Name) and func.id == "filterwarnings")
    if not is_filterwarnings:
        return False
    if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "ignore":
        return True
    return any(
        kw.arg == "action" and isinstance(kw.value, ast.Constant) and kw.value.value == "ignore"
        for kw in node.keywords
    )


def main(argv: list[str]) -> int:
    targets = [Path(arg) for arg in argv[1:]] if len(argv) > 1 else [Path("src"), Path("tests")]
    all_violations: list[str] = []
    files_checked = 0

    for target in targets:
        if not target.exists():
            print(f"[ERRO] Caminho inexistente: {target}", file=sys.stderr)
            return 2
        py_files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for py_file in py_files:
            files_checked += 1
            all_violations.extend(check_file(py_file))

    if all_violations:
        for item in all_violations:
            print(f"[VIOLATION] {item}", file=sys.stderr)
        print(
            f"Falha no Floor de CONSTRAINTS.md: {len(all_violations)} violação(ões).",
            file=sys.stderr,
        )
        return 1

    print(f"[OK] Floor de CONSTRAINTS.md aprovado em {files_checked} arquivo(s) Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
