# Report Notes: Why the Report Does Not Import the Shared Template

The build specification asks the report to reuse the shared template at `C:\Users\Rajch\Desktop\Research\templates\report_template.typ`. `report/edgefed_bench_v2.typ` instead carries its own copy of the template's styling and components (`stat-card`, `callout`, header banner, headers and footers). There are three reasons.

1. **Typst cannot import it.** Typst 0.15 does not let a document read files outside the project root. Compiling with `typst compile report/edgefed_bench_v2.typ report/edgefed_bench_v2.pdf` and an `#import "../../../templates/report_template.typ"` fails with `path would escape the project root`. A copy inside the repository also means the report compiles on any machine that has the repository.

2. **The template's spacing pushes the report to five pages.** The template uses 1.6 cm vertical and 1.8 cm horizontal margins, 9.5 pt text and 0.58 em leading. With four tables, two figures and the five required sections, that layout runs onto a fifth page. The report uses 1.4 cm top and bottom margins, 1.6 cm side margins and 8.8 pt text, which fits in four pages.

3. **Numbers come from generated files.** Every number in the report prose, stat cards and callouts is read from `report/metrics.typ`, and every table from `report/tables.typ`. Both are written by `experiments/make_report_assets.py` from `results/results.duckdb`, so no number is typed by hand.
