# Report Implementation Notes: Template Architecture & Typst Sandboxing

## 1. External Template Import vs. Self-Contained Document

The repository's build specification mentions reusing the general template at `C:\Users\Rajch\Desktop\Research\templates\report_template.typ`. In `report/edgefed_bench_v2.typ`, the styling and component semantics (`stat-card`, `callout`, header banners, dual-column headers and footers) are retained in a self-contained Typst document rather than an external `#import` for the following structural reasons:

1. **Typst Root Sandbox Isolation:**
   Typst 0.15 enforces strict filesystem sandboxing. Compiling from within the standalone repository (`typst compile report/edgefed_bench_v2.typ report/edgefed_bench_v2.pdf`) prohibits references to paths outside the repository root. An external `#import "../../../templates/report_template.typ"` causes a fatal compilation error (`path would escape the project root`). A standalone repository must remain fully compilable on any machine without external host dependencies.

2. **Strict Four-Page Layout Budget:**
   The generic template defines default margins of 1.6cm vertical and 1.8cm horizontal with 9.5pt text size and 0.58em leading. When combined with the four empirical tables (Tables 1, 2, 3, and 4), two high-resolution vector figures, and five technical sections mandated by the build spec, the generic layout causes page overflow into a 5th page. The report's calibrated margins (1.4cm top/bottom, 1.6cm left/right) and 8.8pt font size allow the complete technical content to fit into exactly 4 pages without truncating tables, figures, or prose.

3. **Database-Driven Metric Injection:**
   All quantitative claims in prose, stat-cards, and callouts are dynamically bound to `report/metrics.typ` (generated from `results/results.duckdb` via `experiments/make_report_assets.py`). The customized structure ensures zero manual numerical entry while maintaining complete mathematical fidelity.
