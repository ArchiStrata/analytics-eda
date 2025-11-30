import json
from pathlib import Path


def load_and_validate_report(response: dict, report_dir: Path) -> dict:
    """
    Given the response and the directory
    where reports are written, this will:

      1. Assert that 'report_path' is present in the response.
      2. Assert that the file exists and is a regular file.
      3. Load it as JSON (failing if invalid).
      4. Return the parsed JSON.

    Usage in pytest:
        report = load_and_validate_report(out, tmp_path)
        # now you can make assertions about report['metadata'], report['data'], etc.
    """
    # 1. Key present
    assert "report_path" in response, "response must contain 'report_path'"
    report_file = response["report_path"]

    # Accept str or Path-like
    if isinstance(report_file, str | bytes):
        path = (report_dir / report_file) if not Path(report_file).is_absolute() else Path(report_file)
    else:
        # assume Path-like
        p = Path(report_file)
        path = p if p.is_absolute() else (report_dir / p)

    # 2. File exists
    assert path.exists() and path.is_file(), f"Report file not found at {path!s}"

    # 3. Load & validate JSON
    try:
        with open(path, encoding="utf-8") as f:
            full_report = json.load(f)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Report file is not valid JSON: {e}") from e

    # 4. Return parsed report
    return full_report
