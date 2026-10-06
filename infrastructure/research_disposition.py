"""Pure validation of a frozen non-execution declaration; no store writes."""

from .research_store import ResearchError


def declared_execution_disposition(cell):
    if "execution_disposition" not in cell:
        return None
    value = cell["execution_disposition"]
    if (type(value) is not dict or set(value) != {"schema_version", "status", "reason"}
            or value["schema_version"] != "pirc25-execution-disposition-v1"
            or type(value["status"]) is not str
            or value["status"] not in {"NOT_IMPLEMENTED", "INELIGIBLE", "MISSING_INPUT", "UNQUALIFIED"}
            or type(value["reason"]) is not str or not 0 < len(value["reason"]) <= 512
            or any(ord(char) < 32 for char in value["reason"])
            or "execution" in cell):
        raise ResearchError("CONTRACT_MISMATCH", "invalid non-executable cell declaration")
    return dict(value)


def require_executable_cell(cell):
    declaration = declared_execution_disposition(cell)
    if declaration is not None:
        raise ResearchError(declaration["status"], "registered cell is declared non-executable")
