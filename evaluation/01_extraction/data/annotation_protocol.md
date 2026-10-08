# Annotation Protocol

To enable a rigorous evaluation of extraction quality, we design a **two‑tier evaluation benchmark** and define a complete protocol as follows.

## Inclusion / Exclusion Criteria
- The evaluation set consists of **1,485 airway event records** that were systematically extracted from the source documents.
- **Changeover points** are excluded because they do not represent airway entities.
- Only complete and unambiguous records are retained.

## Annotator Qualifications & Instructions
- Two independent annotators with background in aeronautical information and geographic information science participated in constructing the human gold standard.
- A unified guideline was provided prior to annotation:
  - Annotate **point‑by‑point** based solely on the printed content of the source PDF.
  - Preserve the **printed order** of waypoints.
  - Do **not** supplement missing information from external databases.
  - Merge records that span across pages.
  - Add notes for unclear images, endpoints‑only entries, and cross‑page cases.

## Inter‑Annotator Agreement (IAA)
- On a sample of **150** gold‑standard records:
  - **Point‑level F1 = 0.872** (P = 0.839, R = 0.908)
  - **Record‑level strict agreement rate = 0.567**
- The relatively low record‑level agreement mainly stems from tokenisation differences (e.g., `"SAN ANTONIO"` vs. `"SANANTONIO"`) and OCR‑level character variations (e.g., `"LBOCK"` vs. `"LUBBOCK"`), rather than substantive disagreements in waypoint sets. Therefore, point‑level F1 is considered a more reliable metric.

## Disagreement Resolution
- Annotations from the two annotators were compared record by record:
  - **86 records** were in complete agreement.
  - **50 records** showed only tokenisation or character differences – these were normalised to a canonical form.
  - **14 records** contained substantive content disagreements – these were adjudicated jointly by the two annotators against the original PDF screenshots to produce the final gold standard.
- All 150 sample records are accompanied by source document screenshots (see `Tier B human-gold benchmark/pdf_screenshot/`) for verification.