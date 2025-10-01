from pathlib import Path
from ga_pipeline import processing, qualitative, report

# ----------------------------
# User-configurable switches
# ----------------------------
PROCESS_DATA   = True
RUN_QUAL       = False
BUILD_GRAPHS   = True         # graphs are generated inside the PDF
ASSEMBLE_PDF   = True
excel_name = "Guest Ambassador Rounding Survey.xlsx"
OUTPUT_PDF_NAME  = "09302025_results.pdf"        # final report name

# ----------------------------
# Inputs / Outputs
# ----------------------------
DATA_PATH        = Path("data") / excel_name
OUTPUT_DIR       = Path("outputs")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BOOTSTRAP        = True
BOOTSTRAP_QUANT  = 1000


########################################################################
########################################################################
########################################################################

def main():
    df = None
    artifacts = {}

    if PROCESS_DATA:
        print("[1/4] Processing data...")
        df, artifacts = processing.run_processing(DATA_PATH)
        # artifacts keys:
        # 'department_ind_dict', 'monthly_data', 'dep_service', 'dep_wait',
        # 'pos_texts_by_dept', 'neg_texts_by_dept', 'months'

    pos_theme_df = None
    neg_theme_df = None
    qual = {}

    if RUN_QUAL:
        print("[2/4] Running qualitative analysis (BERTopic)...")
        pos_theme_df, neg_theme_df, qual = qualitative.run_qualitative(
            pos_texts_by_dept=artifacts.get("pos_texts_by_dept", {}),
            neg_texts_by_dept=artifacts.get("neg_texts_by_dept", {}),
            bootstrap=BOOTSTRAP,
            bootstrap_quant=BOOTSTRAP_QUANT,
            output_dir=OUTPUT_DIR,
        )
        # qual contains 'pos_themes_by_dept', 'neg_themes_by_dept'

    if ASSEMBLE_PDF and BUILD_GRAPHS:
        print("[3/4] Building PDF report...")
        report.build_pdf(
            output_pdf_path=OUTPUT_DIR / OUTPUT_PDF_NAME,
            department_ind_dict=artifacts.get("department_ind_dict", {}),
            monthly_data=artifacts.get("monthly_data", {}),
            dep_service=artifacts.get("dep_service", {}),
            dep_wait=artifacts.get("dep_wait", {}),
            months=artifacts.get("months", []),
            # optional qualitative dfs; report can ignore if None
            pos_theme_df=pos_theme_df,
            neg_theme_df=neg_theme_df,
        )

    print("[4/4] Done!")

if __name__ == "__main__":
    main()
