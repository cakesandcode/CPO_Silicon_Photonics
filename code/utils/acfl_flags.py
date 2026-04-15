"""
ACfL compiler flag parsing and vectorization report analysis utilities.

This module provides functions to parse ACfL compiler commands, map -mcpu targets
to implicit ISA extensions, and parse vectorization reports.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_acfl_command(cmd: str) -> Dict:
    """
    Parse an ACfL compiler command and return a structured breakdown.

    Args:
        cmd: Compiler command string.
             Example: "armclang -O3 -mcpu=neoverse-v3 -fvectorize file.c"

    Returns:
        Dictionary with keys:
            - compiler: str (armclang, armflang, armftn, etc.)
            - flags: dict mapping flag name to value or True
            - mcpu: str or None (e.g., 'neoverse-v3')
            - march: str or None (e.g., 'armv9.2-a+sve2')
            - opt_level: str ('-O0', '-O1', '-O2', '-O3', or 'unknown')
            - inferred_extensions: list[str] (SVE2, SME2, etc.)
            - warnings: list[str] (deprecated flags, conflicts)

    Raises:
        ValueError: If command is unparseable (e.g., empty or malformed).
    """
    if not cmd or not isinstance(cmd, str):
        raise ValueError("Command must be a non-empty string")

    cmd = cmd.strip()
    tokens = cmd.split()

    if not tokens:
        raise ValueError("Command is empty after splitting")

    # Extract compiler name from first token
    compiler_path = tokens[0]
    compiler = Path(compiler_path).name if "/" in compiler_path else compiler_path

    # Initialize result dict
    result = {
        "compiler": compiler,
        "flags": {},
        "mcpu": None,
        "march": None,
        "opt_level": "unknown",
        "inferred_extensions": [],
        "warnings": [],
    }

    # Parse tokens
    for token in tokens[1:]:
        # Skip source files
        if token.endswith((".c", ".f90", ".f", ".cc", ".cpp")):
            continue

        # Optimization level
        if token in ("-O0", "-O1", "-O2", "-O3"):
            result["opt_level"] = token

        # -mcpu=target
        elif token.startswith("-mcpu="):
            result["mcpu"] = token.split("=", 1)[1]
            result["flags"]["mcpu"] = result["mcpu"]

        # -march=target
        elif token.startswith("-march="):
            result["march"] = token.split("=", 1)[1]
            result["flags"]["march"] = result["march"]

        # Boolean flags
        elif token in ("-fvectorize", "-O3", "-fast"):
            result["flags"][token] = True

    # Infer extensions from -mcpu
    if result["mcpu"]:
        try:
            ext_info = mcpu_to_extensions(result["mcpu"])
            if ext_info.get("sve2"):
                result["inferred_extensions"].append("SVE2")
            if ext_info.get("sme2"):
                result["inferred_extensions"].append("SME2")
            if ext_info.get("sme"):
                result["inferred_extensions"].append("SME")
            if ext_info.get("sve"):
                result["inferred_extensions"].append("SVE")
            if ext_info.get("neon"):
                result["inferred_extensions"].append("NEON")
        except KeyError:
            result["warnings"].append(f"Unknown -mcpu={result['mcpu']}")

    # Infer extensions from -march
    if result["march"]:
        march_lower = result["march"].lower()
        if "+sve2" in march_lower:
            result["inferred_extensions"].append("SVE2")
        elif "+sve" in march_lower:
            result["inferred_extensions"].append("SVE")
        if "+sme2" in march_lower:
            result["inferred_extensions"].append("SME2")
        elif "+sme" in march_lower:
            result["inferred_extensions"].append("SME")
        if "armv" in march_lower:
            result["inferred_extensions"].append("NEON")

    # Deduplicate extensions
    result["inferred_extensions"] = list(set(result["inferred_extensions"]))

    return result


def mcpu_to_extensions(mcpu: str) -> Dict:
    """
    Map a -mcpu target to its implicit ISA extensions and properties.

    Args:
        mcpu: CPU target string (e.g., 'neoverse-v3', 'neoverse-n3', 'generic').

    Returns:
        Dictionary with keys:
            - cpu: str
            - isa_level: str (e.g., 'armv9.2-a')
            - sve: bool
            - sve2: bool
            - sme: bool
            - sme2: bool
            - neon: bool
            - notes: str

    Raises:
        KeyError: If mcpu is not recognized.
    """
    # Comprehensive lookup table for Neoverse and popular Cortex targets
    lookup_table = {
        "neoverse-v3": {
            "cpu": "Neoverse V3",
            "isa_level": "armv9.2-a",
            "sve": True,
            "sve2": True,
            "sme": True,
            "sme2": True,
            "neon": True,
            "notes": "128-bit SVE width, streaming SVE (SME2), highest performance tier",
        },
        "neoverse-v2": {
            "cpu": "Neoverse V2",
            "isa_level": "armv9.1-a",
            "sve": True,
            "sve2": True,
            "sme": False,
            "sme2": False,
            "neon": True,
            "notes": "128-bit SVE width, no streaming mode (SME2)",
        },
        "neoverse-n3": {
            "cpu": "Neoverse N3",
            "isa_level": "armv9.2-a",
            "sve": True,
            "sve2": True,
            "sme": False,
            "sme2": False,
            "neon": True,
            "notes": "General-purpose, SVE2 support, no streaming",
        },
        "neoverse-n2": {
            "cpu": "Neoverse N2",
            "isa_level": "armv9.0-a",
            "sve": True,
            "sve2": False,
            "sme": False,
            "sme2": False,
            "neon": True,
            "notes": "General-purpose, SVE only (not SVE2)",
        },
        "neoverse-n1": {
            "cpu": "Neoverse N1",
            "isa_level": "armv8.2-a",
            "sve": False,
            "sve2": False,
            "sme": False,
            "sme2": False,
            "neon": True,
            "notes": "General-purpose, NEON only",
        },
        "cortex-a78": {
            "cpu": "Cortex-A78",
            "isa_level": "armv8.2-a",
            "sve": False,
            "sve2": False,
            "sme": False,
            "sme2": False,
            "neon": True,
            "notes": "Mobile/edge, NEON only",
        },
        "cortex-a76": {
            "cpu": "Cortex-A76",
            "isa_level": "armv8.2-a",
            "sve": False,
            "sve2": False,
            "sme": False,
            "sme2": False,
            "neon": True,
            "notes": "Mobile, NEON only",
        },
        "generic": {
            "cpu": "Generic ARM",
            "isa_level": "armv8.0-a",
            "sve": False,
            "sve2": False,
            "sme": False,
            "sme2": False,
            "neon": True,
            "notes": "Baseline ARMv8, NEON only, maximum compatibility",
        },
    }

    if mcpu.lower() not in lookup_table:
        raise KeyError(f"Unknown -mcpu target: {mcpu}")

    return lookup_table[mcpu.lower()]


def load_vectorization_report(file_path: Path) -> Tuple[Dict, List[str]]:
    """
    Load and parse an annotated ACfL vectorization report (mock or real).

    Args:
        file_path: Path to the report text file.

    Returns:
        Tuple of:
            - report_dict: dict with keys 'vectorized_loops', 'scalar_loops', 'metadata'
            - annotated_lines: list[str] with HTML-safe annotations

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Report file not found: {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    # Parse metadata from header (target, compiler version, etc.)
    metadata = {}
    mcpu_match = re.search(r"-mcpu=(\S+)", content)
    if mcpu_match:
        metadata["mcpu"] = mcpu_match.group(1)

    # Count vectorized and scalar loops
    vectorized_count = len(re.findall(r"VECTORIZED", content, re.IGNORECASE)) - len(re.findall(r"NOT VECTORIZED", content, re.IGNORECASE))
    scalar_count = len(re.findall(r"NOT VECTORIZED", content, re.IGNORECASE))

    report_dict = {
        "vectorized_loops": vectorized_count,
        "scalar_loops": scalar_count,
        "metadata": metadata,
    }

    # Annotate lines for display
    lines = content.split("\n")
    annotated_lines = []

    for line in lines:
        if "VECTORIZED" in line.upper() and "NOT" not in line.upper():
            # Vectorized loop: annotate with SVE2 color
            annotated_lines.append(f'<span style="color: #00CC66">{line}</span>')
        elif "NOT VECTORIZED" in line.upper():
            # Non-vectorized loop: annotate with amber color
            annotated_lines.append(f'<span style="color: #FFAA00">{line}</span>')
        else:
            annotated_lines.append(line)

    return report_dict, annotated_lines
