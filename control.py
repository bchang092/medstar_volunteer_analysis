from pathlib import Path
from ga_pipeline import processing, qualitative, report

# ----------------------------
# User-configurable switches
# ----------------------------
PROCESS_DATA   = True
RUN_QUAL       = True
BUILD_GRAPHS   = True         # graphs are generated inside the PDF
ASSEMBLE_PDF   = True
excel_name = "1106_rounding_survey.xlsx"
OUTPUT_PDF_NAME  = "1106_results_qual.pdf"        # final report name
DATE_START = "2025-06-30"   # or None
DATE_END   = "2025-11-06"   # or None

# ----------------------------
# Inputs / Outputs
# ----------------------------
DATA_PATH        = Path("data") / excel_name
OUTPUT_DIR       = Path("outputs") / (DATE_END or "latest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BOOTSTRAP        = False
BOOTSTRAP_QUANT  = 0


def main():
    df = None
    artifacts = {}

    if PROCESS_DATA:
        print("[1/3] Processing data...")
        df, artifacts = processing.run_processing(
            excel_path=DATA_PATH,
            date_start=DATE_START,
            date_end=DATE_END,
        )
        # artifacts keys:
        # 'department_ind_dict', 'monthly_data', 'dep_service', 'dep_wait',
        # 'pos_texts_by_dept', 'neg_texts_by_dept', 'months'

    pos_theme_df = None
    neg_theme_df = None
    qual = {}

    if RUN_QUAL:
        print("[2/3] Running qualitative analysis (LDA/topic maps)...")
        pos_theme_df, neg_theme_df, qual = qualitative.run_qualitative(
            pos_texts_by_dept=artifacts.get("pos_texts_by_dept", {}),
            neg_texts_by_dept=artifacts.get("neg_texts_by_dept", {}),
            bootstrap=BOOTSTRAP,
            bootstrap_quant=BOOTSTRAP_QUANT,
            output_dir=OUTPUT_DIR,
            date_start=DATE_START,
            date_end=DATE_END,
            min_docs=3,               # run only if >= 5 reviews
            # legacy args are accepted & ignored in the revised qualitative.py:
            ngram_range=(1, 2),
            max_features=6000,
            min_df=2,
            max_df=1.0,
            k_max=8,
            n_top_terms=8,
            random_state=42,
        )

    if ASSEMBLE_PDF and BUILD_GRAPHS:
        print("[3/3] Building PDF report...")
        # Prefer the live Matplotlib Figures created by qualitative.run_qualitative
        pos_figs_mpl = (qual or {}).get("pos_figs_mpl", {})
        neg_figs_mpl = (qual or {}).get("neg_figs_mpl", {})

        report.build_pdf(
            output_pdf_path=OUTPUT_DIR / OUTPUT_PDF_NAME,
            department_ind_dict=artifacts["department_ind_dict"],
            monthly_data=artifacts["monthly_data"],
            dep_service=artifacts["dep_service"],
            dep_wait=artifacts["dep_wait"],
            months=artifacts["months"],
            pos_theme_figs_mpl=qual.get("pos_figs_mpl", {}),
            neg_theme_figs_mpl=qual.get("neg_figs_mpl", {}),
            date_start=DATE_START,
            date_end=DATE_END,
        )

    print("Done!")

if __name__ == "__main__":
    main()
