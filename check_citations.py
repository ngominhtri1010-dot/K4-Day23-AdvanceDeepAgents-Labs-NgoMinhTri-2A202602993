"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list):
        return ["sources.json must be a JSON list of objects"]
    if not sources:
        problems.append("no sources in sources.json")

    by_number = {}
    urls = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source entry {index + 1} must be an object")
            continue
        number, url = source.get("n"), source.get("url")
        if type(number) is not int:
            problems.append(f"source entry {index + 1}: n must be an integer")
        elif number in by_number:
            problems.append(f"duplicate source number [{number}]")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source entry {index + 1}: invalid URL")
        elif url in urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            urls.add(url)

    # Mask code while preserving newlines so headings inside code are ignored.
    lines = []
    fence = None
    for line in report_text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            lines.append("\n")
        elif marker:
            fence = marker[1]
            lines.append("\n")
        elif line.startswith(("    ", "\t")):
            lines.append("\n")
        else:
            lines.append(line)
    text = "".join(lines)
    text = re.sub(r"(`+)(?!`)([\s\S]*?)(?<!`)\1(?!`)", "", text)
    headings = list(re.finditer(r"(?m)^##[ \t]+References[ \t]*\r?$", text))
    if not headings:
        problems.append("missing ## References heading")
        body, references = text, ""
    else:
        if len(headings) > 1:
            problems.append("duplicate ## References heading")
        body, references = text[:headings[0].start()], text[headings[0].end():]

    cited = set()
    pattern = r"(?<![\\!])\[(\d+(?:\s*[,–-]\s*\d+)*)\](?![ \t]*\()"
    for match in re.finditer(pattern, body):
        for part in match[1].split(","):
            span = re.fullmatch(r"\s*(\d+)\s*[–-]\s*(\d+)\s*", part)
            if span:
                start, end = map(int, span.groups())
                if end < start or end - start > 10000:
                    problems.append(f"invalid citation range [{part.strip()}]")
                else:
                    cited.update(range(start, end + 1))
            else:
                cited.add(int(part.strip()))
    for number in sorted(cited - by_number.keys()):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(by_number.keys() - cited):
        problems.append(f"source [{number}] never cited")

    counts = {}
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\]", line)
        if not match:
            continue
        number = int(match[1])
        counts[number] = counts.get(number, 0) + 1
        if number not in by_number:
            problems.append(f"reference [{number}] missing from sources.json")
        found_urls = re.findall(r"https?://[^\s<>]+", line)
        if len(found_urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in by_number and found_urls[0] != by_number[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")
    for number in sorted(by_number):
        if number not in counts:
            problems.append(f"missing reference line for [{number}]")
    for number, count in sorted(counts.items()):
        if count > 1:
            problems.append(f"duplicate reference lines for [{number}]")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
