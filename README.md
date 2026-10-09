# PrimateBehaviorAI Studio 4.8.0 — streamlined Windows desktop source

A research-oriented multi-camera (1–3) observation application for animal intervention experiments. Includes concise capture UI, rapid joint stream preflight, aspect-ratio preserving previews, automatic preflight mode fallback with recording confirmation, experiment stages, single/paired/Session analysis without upfront mode selection, expert video review, and workbook reporting.

**This zip is source + Windows installer build configuration, NOT a compiled Windows installer.** Build with `.github/workflows/windows-installer.yml` (GitHub Actions) or `BUILD_INSTALLER.bat` on Windows. The resulting setup installer bundles the Python runtime and dependencies; no Python installation on the destination PC.

Read `README_CN.md` and `docs/GITHUB_SETUP_CN.md`. Includes `python -m unittest discover -s tests -v` tests. Camera and installer validation require real Windows hardware. AI events are research candidates, not medical diagnoses.
