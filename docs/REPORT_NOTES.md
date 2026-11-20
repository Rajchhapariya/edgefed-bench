# Report Notes: Report Layout

## Version 2.1: standard paper layout

`report/edgefed_bench_v2.typ` now uses a plain research-paper layout instead of the shared template's dashboard style. The stat cards, coloured callouts and dark title banner are gone. In their place:

- A centred title, author line, abstract and keywords.
- New Computer Modern at 10 pt, justified paragraphs, A4 with 2.3 cm side margins.
- Numbered sections and cross-references (for example "Section 3.3", "Table 5").
- Numbered table captions above each table and figure captions below each figure.
- Tables with horizontal rules only (top, under the header, bottom) and no shading.
- Figures without in-figure titles and with serif labels; the caption carries the description.
- Author-year citations with an alphabetical reference list, and the reproduction steps in an appendix.

The report is 13 pages. The first version was 4 pages; version 2.1 adds the statistics, the mechanism measurements and eight extended studies.

## Why the report does not import the shared template

The build specification asks the report to reuse the shared template at `C:\Users\Rajch\Desktop\Research\templates\report_template.typ`. Typst 0.15 does not let a document read files outside the project root: an `#import "../../../templates/report_template.typ"` fails with `path would escape the project root`. Keeping the styling inside the repository also means the report compiles on any machine that has the repository.

## Numbers come from generated files

Every number in the report prose is read from `report/metrics.typ`, and every table from `report/tables.typ`. Both are written by `experiments/make_report_assets.py` from `results/results.duckdb`, so no number is typed by hand.
