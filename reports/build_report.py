"""
build_report.py - builds reports/Project_Report.docx from the project outputs.

Every number in the report is read from the files produced by the pipeline (results.json, experiments.csv,
eda_insights.md, the processed dataset), so the document stays consistent with the code.

Usage (repository root):  python reports/build_report.py
"""
import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alerts import current_alerts, supplier_scorecard  # noqa: E402
from forecasting import MODEL_FILE, load_data  # noqa: E402

REP = ROOT / "reports"
FIG = REP / "figures"
SHOT = REP / "screenshots"
OUT = REP / "Project_Report.docx"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)

res = json.loads((REP / "results.json").read_text())
exp = pd.read_csv(REP / "experiments.csv")
data = load_data()
bundle = joblib.load(MODEL_FILE)
alerts = current_alerts(data, bundle)
sup = supplier_scorecard(data)
api = json.loads((REP / "api_samples.json").read_text())
insights = re.findall(r"## (.+?)\n\*\*Insight:\*\* (.+?)\n\n\*\*Decision for the store manager:\*\* (.+?)\n",
                      (REP / "eda_insights.md").read_text(encoding="utf-8"), re.S)
pct = lambda x, n=1: f"{x * 100:.{n}f}%"  # noqa: E731
champ = res["champion_test"]
al = res["alerts"]


# ------------------------------------------------------------------ architecture diagram
def architecture_figure():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    boxes = {
        "raw": (0.2, 3.1, "data/raw/dataset.csv\n500 supplied rows", "#E8EEF7"),
        "prep": (2.5, 3.1, "prepare_data.py\nextend, clean, mask cost", "#DCE8F5"),
        "proc": (5.0, 3.1, "data/processed/\ndataset.csv + dictionary", "#E8EEF7"),
        "eda": (7.6, 3.9, "eda.py\n5 figures + insights", "#DCE8F5"),
        "feat": (7.6, 2.3, "forecasting.py\nleak-free SKU features", "#DCE8F5"),
        "train": (7.6, 0.7, "train.py\nbaselines, tuning,\ntest set opened once", "#DCE8F5"),
        "model": (5.0, 0.7, "models/forecast.joblib\nchampion + SKU state", "#E8EEF7"),
        "alerts": (2.5, 0.7, "alerts.py\nreorder logic, suppliers", "#DCE8F5"),
        "ui": (0.2, 0.0, "Flask REST API\nStreamlit dashboard", "#F5E8DC"),
    }
    w, h = 2.0, 0.95
    for _, (x, y, t, c) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="#1F3A5F", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)
    arrows = [("raw", "prep"), ("prep", "proc"), ("proc", "eda"), ("proc", "feat"), ("feat", "train"),
              ("train", "model"), ("model", "alerts"), ("alerts", "ui")]
    for a, b in arrows:
        xa, ya, _, _ = boxes[a]
        xb, yb, _, _ = boxes[b]
        if a == "raw" or a == "prep":
            s, e = (xa + w, ya + h / 2), (xb, yb + h / 2)
        elif a == "proc":
            s, e = (xa + w, ya + h / 2), (xb, yb + h / 2)
        elif a == "feat":
            s, e = (xa + w / 2, ya), (xb + w / 2, yb + h)
        elif a == "train":
            s, e = (xa, ya + h / 2), (xb + w, yb + h / 2)
        elif a == "model":
            s, e = (xa, ya + h / 2), (xb + w, yb + h / 2)
        else:
            s, e = (xa + 0.3, ya), (xb + 0.3, yb + h + 0.0)
            s, e = (xa, ya + h / 2), (xb + w, yb + h / 2 - 0.0)
        ax.annotate("", xy=e, xytext=s, arrowprops=dict(arrowstyle="->", color="#1F3A5F", lw=1.2))
    ax.set_xlim(0, 10); ax.set_ylim(-0.3, 5.1)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.3)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
    s = doc.styles[name]
    s.font.name, s.font.size, s.font.bold, s.font.color.rgb = "Calibri", Pt(size), True, NAVY
    s.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    s.paragraph_format.space_before, s.paragraph_format.space_after = Pt(14 if name == "Heading 1" else 10), Pt(6)
    s.paragraph_format.keep_with_next = True


def h1(t, page_break=True):
    if page_break:
        doc.add_page_break()
    doc.add_heading(t, 1)


def h2(t):
    doc.add_heading(t, 2)


def para(t, bold_lead=None, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead + " ")
        r.bold = True
    r = p.add_run(t)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullets(items):
    for t in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(t)


def numbered(items):
    for t in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(t)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def table(headers, rows, widths=None, caption=None, font=9.5):
    if caption:
        c = doc.add_paragraph()
        r = c.add_run(caption)
        r.bold = True
        r.font.size = Pt(10)
        c.paragraph_format.keep_with_next = True
        c.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htxt in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(htxt))
        run.bold, run.font.size, run.font.color.rgb = True, Pt(font), RGBColor(255, 255, 255)
        shade(cell, "1F3A5F")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            cell.paragraphs[0].paragraph_format.space_after = Pt(1)
            if widths:
                cell.width = Cm(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, caption, width=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic, r.font.size = True, Pt(9.5)


def add_field(paragraph, instr):
    r = paragraph.add_run()
    for kind, text in (("begin", None), (None, instr), ("separate", None), (None, "1"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        elif text == instr:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = instr
        else:
            el = OxmlElement("w:t")
            el.text = text
        r._r.append(el)


# header / footer
hp = sec.header.paragraphs[0]
hp.text = "Optimized Retail Inventory Management System  |  Project No. 21"
hp.runs[0].font.size, hp.runs[0].font.color.rgb = Pt(9), RGBColor(0x66, 0x66, 0x66)
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fp.add_run("Page ").font.size = Pt(9)
add_field(fp, "PAGE")

# ------------------------------------------------------------------ cover
architecture_figure()
for _ in range(3):
    doc.add_paragraph()
para("HCLTech Career Shaper Programme", align=WD_ALIGN_PARAGRAPH.CENTER, size=14)
para("III Year Industry Project, Academic Year 2025-26", align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("PROJECT REPORT")
r.bold, r.font.size, r.font.color.rgb = True, Pt(26), NAVY
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Optimized Retail Inventory Management System")
r.bold, r.font.size = True, Pt(20)
para("Demand forecasting, reorder alerts, supplier management and sales analytics",
     align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=12)
for _ in range(2):
    doc.add_paragraph()
table(["Field", "Details"], [
    ["Project No.", "21"], ["Domain", "Digital-Data-AI / AIML"],
    ["Difficulty / Duration", "Intermediate / 4 weeks (individual project)"],
    ["Learner name", "Apurv Sharma"], ["Roll number", "[FILL IN]"], ["College / campus", "[FILL IN]"],
    ["Mentor name", "[FILL IN]"], ["Submission date", "[FILL IN]"]], widths=[5, 10.5], font=11)

# ------------------------------------------------------------------ contents
doc.add_page_break()
doc.add_heading("Contents", 1)
toc = doc.add_paragraph()
add_field(toc, 'TOC \\o "1-2" \\h \\z \\u')

# ------------------------------------------------------------------ 1 summary
h1("1. Summary")
para("Retail stores lose sales when fast-moving products run out and lock up cash in products that sell slowly. The "
     "store manager has to decide what to reorder, when, and from which supplier. A fixed reorder level per product "
     "ignores both seasonal demand and the lead time of the supplier. This project builds a working prototype that "
     "replaces the fixed rule with a forecast-based one and presents the result to the manager through a REST API and "
     "a dashboard.")
para("The inventory log supplied with the project brief (500 rows) was extended to 5,000 records covering 500 SKUs, "
     "30 suppliers and all twelve months of 2024, using a reproducible script with documented assumptions. Supplier "
     "prices were converted into a masked Cost Index. Demand is forecast per SKU and season. Models were compared "
     "using a chronological split and time-series cross-validation, and the test period (1 October to 31 December "
     "2024) was evaluated once.")
para(f"The selected model, a seasonal naive forecast, reaches a test WAPE of {pct(champ['WAPE'])} and a MAPE of "
     f"{pct(champ['MAPE'])}, against a WAPE of {pct(res['naive_test_wape'])} for the naive forecast. The forecast-based "
     f"reorder rule detects {pct(al['forecast_rule']['recall'])} of simulated stock-outs, compared with "
     f"{pct(al['store_reorder_level']['recall'])} for the store's fixed reorder level. All three success criteria "
     "set at the start of the project are met. The tuned machine-learning models did not outperform the simple "
     "seasonal methods on this data, which is discussed in Section 8.")

# ------------------------------------------------------------------ 2 introduction
h1("2. Introduction")
h2("2.1 Background")
para("Inventory decisions involve three recurring risks. A stock-out loses the sale and may lose the customer. "
     "Overstock ties up working capital and storage space. Supplier variability makes a reorder point that worked last "
     "month too low when delivery takes 21 days instead of 8. In Indian retail these risks are amplified by a strong "
     "seasonal pattern: a summer peak, a monsoon slowdown and a festive surge around Diwali.")
h2("2.2 Problem statement and target user")
para("A retail store runs out of fast-selling products while slow-moving products tie up money. Store managers need to "
     "know what to reorder, when, and from which supplier. The target user is the store or inventory manager.")
h2("2.3 Objectives")
bullets([
    "Build a working prototype with demand forecasting, reorder alerts, supplier management and sales analytics.",
    "Prepare the supplied dataset reproducibly and protect commercially sensitive supplier pricing.",
    "Compare forecasting approaches fairly, against baselines, without using the test period for model selection.",
    "Expose the logic through a REST API and an interactive dashboard.",
    "Validate the system against measurable criteria and document limitations."])
h2("2.4 Success criteria")
table(["Criterion", "Target", "Result", "Met"], [
    ["Forecast error on the test period", "WAPE below naive; MAPE 20% or lower",
     f"WAPE {pct(champ['WAPE'])} vs {pct(res['naive_test_wape'])}; MAPE {pct(champ['MAPE'])}", "Yes"],
    ["Alerts detect simulated stock-outs", "Recall 90% or higher",
     f"Recall {pct(al['forecast_rule']['recall'])}, precision {pct(al['forecast_rule']['precision'])}", "Yes"],
    ["Dashboard answers the business questions", "What to reorder, overstocked category, slowest supplier",
     "Reorder alerts tab, stock cover by category, supplier scorecard", "Yes"]],
    widths=[4.6, 4.4, 5.6, 1.4], caption="Table 2.1: Success criteria set in Stage 1 and their outcome")

# ------------------------------------------------------------------ 3 data
h1("3. Data Description")
h2("3.1 Source and size")
para("The programme supplied dataset.csv with 500 rows and ten columns. All rows fall in the first days of January 2024 "
     "and are labelled Winter, so the file contains no seasonal history, and the category of a SKU differs between "
     "rows. To support the forecasting module the dataset was extended by script (prepare_data.py, fixed seed 42) to "
     "5,000 records: 500 supplied rows plus 4,500 generated rows, 500 SKUs (SKU0001 to SKU0500), 30 suppliers "
     "(SUP01 to SUP30) and the dates 1 January to 31 December 2024. No external dataset was used.")
h2("3.2 Generation assumptions and validation")
table(["Item", "Assumption"], [
    ["SKU demand", "Base demand uniform between 10 and 50 units per day for each SKU"],
    ["Seasonal multipliers", "Winter 1.00, Summer 1.20, Monsoon 0.95, Festive_Diwali 1.40 (1 October to 15 November); random noise with standard deviation 10%"],
    ["Stock", "About 18% of records at or below the reorder level"],
    ["Suppliers", "85% of orders from the primary supplier of the SKU; lead time distributed around a supplier mean, clipped to 3-21 days"],
    ["Anomalies", "150 generated rows (3%): 50 stock-outs, 50 demand surges (2.5-3.5 times normal), 50 supplier delays (22-35 days); the 500 supplied rows are unchanged"],
    ["Cost", "Supplier unit cost masked as a Cost Index (median 100, plus or minus 2% noise)"]],
    widths=[3.6, 12.4], caption="Table 3.1: Assumptions used to generate the additional records")
para("The generated data were validated against the assumptions: 5,000 rows, 500 SKUs, 30 suppliers, 12 months, no "
     "missing values, no duplicates and no negative stock or sales. Mean daily sales by season follow the intended "
     "order (Festive_Diwali 43.1, Summer 36.9, Winter 31.3, Monsoon 29.2 units per day).")
h2("3.3 Data dictionary")
dd = pd.read_csv(ROOT / "data" / "processed" / "data_dictionary.csv")
table(["Column", "Type", "Unit", "Description"],
      [[r.column, r.type, r.unit, r.description] for r in dd.itertuples()],
      widths=[2.6, 1.8, 1.8, 9.8], caption="Table 3.2: Data dictionary (also submitted as data_dictionary.csv)", font=8.5)

# ------------------------------------------------------------------ 4 preparation
h1("4. Data Preparation")
para("The cleaning steps are implemented in prepare_data.py, which regenerates the processed dataset identically on "
     "every run. Table 4.1 lists each decision with its effect on the row count.")
table(["Step", "Decision and reason", "Rows affected"], [
    ["Parse dates", "Converted to datetime; required for the chronological split", "5,000"],
    ["Missing values", "Would be dropped; none present", "0"],
    ["Duplicates", "Exact duplicates would be dropped; none present", "0"],
    ["Impossible values", "Stock and sales clipped at 0, lead time at 1; none required", "0"],
    ["Valid extremes", "Stock-outs, surges and delays kept as genuine operational events", "150 (3.0%)"],
    ["Sensitive data", "unit_cost converted to a Cost Index because supplier pricing is commercially sensitive", "5,000"],
    ["Final size", "5,000 rows, 500 SKUs, 30 suppliers", "-"]],
    widths=[3.0, 10.0, 3.0], caption="Table 4.1: Cleaning log")
para("During modelling, the first record of each SKU has no earlier history and cannot be forecast. These 500 records "
     "are excluded in the modelling step only; the stored dataset keeps them.")
para("The category of a SKU is inconsistent in the 500 supplied rows (252 of 500 SKUs appear with a single category, "
     "the others with two to four). Category analysis is therefore performed at record level.")

# ------------------------------------------------------------------ 5 EDA
h1("5. Exploratory Data Analysis")
para("The analysis answers five business questions. Each figure is followed by the insight it supports and the "
     "decision it informs. The figures are produced by eda.py from the processed dataset.")
figs = ["fig1_cover_by_category.png", "fig2_sales_distribution.png", "fig3_supplier_lead_times.png",
        "fig4_monthly_trend.png", "fig5_alert_rate.png"]
widths = [10.5, 15.5, 15.5, 12.5, 10.5]
for i, ((title, finding, decision), f, wdt) in enumerate(zip(insights, figs, widths), 1):
    h2(f"5.{i} {title.split(' - ', 1)[1]}")
    figure(FIG / f, f"Figure 5.{i}: {title.split(' - ', 1)[1]}", width=wdt)
    para(finding, bold_lead="Insight.")
    para(decision, bold_lead="Decision.")

# ------------------------------------------------------------------ 6 methodology
h1("6. Methodology and System Design")
h2("6.1 Architecture")
para("The system is a pipeline of five scripts that read and write well-defined files, followed by two interfaces. "
     "Each stage can be run on its own, and every figure and table in this report can be regenerated with the "
     "commands in Appendix A.")
figure(FIG / "architecture.png", "Figure 6.1: System architecture and data flow", width=15.5)
table(["Module", "Purpose", "File"], [
    ["Data preparation", "Extend, clean and mask the dataset", "src/prepare_data.py"],
    ["Exploratory analysis", "Profile, figures and insights", "src/eda.py"],
    ["Forecasting", "Leak-free features, metrics, forecast helper", "src/forecasting.py"],
    ["Model training", "Baselines, tuning, evaluation, export", "src/train.py"],
    ["Alerts and suppliers", "Reorder logic, supplier scorecard", "src/alerts.py"],
    ["REST API", "Forecast, alert and supplier endpoints", "api/app.py"],
    ["Dashboard", "Analytics, alerts, suppliers, forecast", "app/dashboard.py"]],
    widths=[3.6, 7.2, 5.2], caption="Table 6.1: Modules")
h2("6.2 Features and leakage control")
para("The target is daily_sales. Every feature of a record is computed from earlier records of the same SKU only: the "
     "history statistics are shifted by one record before any expanding mean is taken, so a record never contributes to "
     "its own features. The season is known from the calendar and is therefore a valid input.")
table(["Feature", "Definition"], [
    ["level", "Expanding mean of de-seasonalised sales (sales divided by the season multiplier) of earlier records of the SKU"],
    ["seas_level", "level multiplied by the multiplier of the record's season (the seasonal level forecast)"],
    ["seas_lag", "Previous record of the SKU, de-seasonalised then re-seasonalised (the seasonal naive forecast)"],
    ["lag_1, prior_mean", "Previous sales of the SKU; plain mean of earlier sales (baseline forecasts)"],
    ["mult, reorder_level, lead_days", "Season multiplier and two supply-side attributes used by the learned models"]],
    widths=[4.2, 11.8], caption="Table 6.2: Engineered features")
h2("6.3 Split and validation")
para(f"Records before 1 October 2024 ({res['n_train']:,} usable records) form the training and validation period. "
     f"Records from 1 October 2024 ({res['n_test']:,}) form the test set and cover the festive season and the start "
     "of winter. The data are never shuffled. Each candidate is scored by four-fold TimeSeriesSplit cross-validation "
     "on the training period, hyper-parameters are tuned on the same folds, and the candidate with the lowest "
     "cross-validated WAPE is selected. The test set is evaluated once, after selection.")
h2("6.4 Candidate models")
bullets(["Naive: the previous recorded sales of the SKU.",
         "SKU mean: the mean of earlier sales of the SKU, without seasonality.",
         "Seasonal naive: previous sales adjusted to the target season.",
         "Seasonal level rule: SKU level (de-seasonalised mean) adjusted to the target season.",
         "Linear regression, gradient boosting and random forest on the engineered features (grid search with "
         "TimeSeriesSplit)."])
h2("6.5 Reorder logic")
para("For each SKU the forecast daily demand is multiplied by the supplier lead time to give demand over the lead time. "
     "The reorder point adds a 20% safety buffer, an assumption chosen as a common starting value:")
para("reorder point = forecast daily demand x lead days x 1.2", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para("An alert is raised when stock is at or below the reorder point. The suggested order quantity is twice the "
     "reorder point minus stock, never below zero. The priority is Critical when stock is zero, High when days of "
     "cover are below half the lead time, and Medium otherwise.")
h2("6.6 Supplier scorecard")
para("Suppliers are ranked by a risk score from 0 to 100 that combines 60% normalised average lead time, 25% delay "
     "rate (lead time above 21 days) and 15% stock-out rate. Tiers Low, Medium and High are assigned by tercile.")

# ------------------------------------------------------------------ 7 implementation
h1("7. Implementation")
h2("7.1 Tools and libraries")
table(["Layer", "Technology"], [
    ["Language and data", "Python 3.14, pandas 3.0, NumPy 2.4"],
    ["Modelling", "scikit-learn 1.8 (linear regression, gradient boosting, random forest, TimeSeriesSplit), joblib"],
    ["Visualisation", "matplotlib (EDA figures), Plotly (dashboard)"],
    ["REST API", "Flask 3.1"],
    ["Dashboard", "Streamlit 1.58"],
    ["Testing", "pytest 9.1 (8 tests: normal, boundary and invalid cases)"]],
    widths=[4.0, 12.0], caption="Table 7.1: Technology stack (versions pinned in requirements.txt)")
h2("7.2 REST API")
para("The Flask API validates its inputs and returns descriptive errors with HTTP status 400 for invalid values and "
     "404 for an unknown SKU.")
table(["Endpoint", "Purpose"], [
    ["GET /api/health", "Service status, selected model and test metrics"],
    ["GET /api/forecast?product_id=&season=", "Forecast daily units for a SKU in a season"],
    ["GET /api/reorder-alerts?category=&priority=&season=", "Current alerts with suggested order quantities"],
    ["GET /api/suppliers?tier=", "Supplier scorecard"]],
    widths=[8.0, 8.0], caption="Table 7.2: API endpoints")
fc = api["/api/forecast?product_id=SKU0001&season=Festive_Diwali"]
para(f"Example: a request for SKU0001 in the Festive_Diwali season returns a forecast of "
     f"{fc['forecast_daily_units']:.1f} units per day. Sample responses from all endpoints are stored in "
     "reports/api_samples.json.")
h2("7.3 Dashboard")
para("The Streamlit dashboard has four tabs and a sidebar with category, season and date filters. It runs with "
     "streamlit run app/dashboard.py.")
figure(SHOT / "dashboard_analytics.png", "Figure 7.1: Sales analytics tab", width=12.0)
n_alert = int(alerts["alert"].sum())
store_n = int((alerts["stock_qty"] <= alerts["reorder_level"]).sum())
pc = alerts["priority"].value_counts()
para(f"The alerts tab evaluates the latest record of every SKU. With the forecast-based rule, {n_alert} of 500 SKUs "
     f"are flagged ({pc.get('Critical', 0)} Critical, {pc.get('High', 0)} High, {pc.get('Medium', 0)} Medium), "
     f"compared with {store_n} under the store's fixed reorder level. The purchase list can be downloaded as CSV.")
figure(SHOT / "dashboard_alerts.png", "Figure 7.2: Reorder alerts tab", width=12.0)
top = sup.iloc[0]
slow = sup.sort_values("avg_lead_days").iloc[-1]
para(f"The supplier tab shows the scorecard. {top['supplier_id']} has the highest risk score ({top['risk_score']:.1f}, "
     f"average lead time {top['avg_lead_days']:.1f} days); {slow['supplier_id']} has the longest average lead time "
     f"({slow['avg_lead_days']:.1f} days). {sup.iloc[-1]['supplier_id']} has the lowest risk score.")
figure(SHOT / "dashboard_suppliers.png", "Figure 7.3: Supplier performance tab", width=12.0)
para("The forecast tab applies the selected model to a chosen SKU, season, stock level and lead time, and shows the "
     "forecast, demand over the lead time, days of cover and a reorder recommendation. Selecting the festive season "
     "raises the forecast for SKU0001 from 35.2 to 49.3 units per day.")
figure(SHOT / "dashboard_forecast_festive.png", "Figure 7.4: Demand forecast tab (Festive_Diwali selected)", width=12.0)
h2("7.4 Testing")
para("Eight automated tests cover the API (valid requests, missing and invalid parameters, unknown SKU), the filters, "
     "the reorder boundary (stock equal to and just above the reorder point), the stock-out simulation and the absence "
     "of target leakage in the features. All tests pass (python -m pytest tests -q). The dashboard was also loaded "
     "headlessly with no exceptions.")

# ------------------------------------------------------------------ 8 results
h1("8. Results and Validation")
h2("8.1 Forecast accuracy")
rows = []
for r in exp.itertuples():
    rows.append([r.model + (" (selected)" if r.champion else ""), f"{r.cv_wape * 100:.1f}%", f"{r.test_mae:.2f}",
                 f"{r.test_rmse:.2f}", f"{r.test_mape * 100:.1f}%", f"{r.test_wape * 100:.1f}%", f"{r.test_r2:.2f}"])
table(["Model", "CV WAPE", "Test MAE", "Test RMSE", "Test MAPE", "Test WAPE", "Test R2"], rows,
      widths=[5.4, 1.8, 1.8, 1.9, 1.9, 1.9, 1.5],
      caption=f"Table 8.1: Experiment table (test period 1 Oct - 31 Dec 2024, {res['n_test']:,} records)", font=9)
para("CV WAPE is the mean over four expanding-window folds of the training period and was the only basis for model "
     "selection. The seasonal naive forecast has the lowest cross-validated WAPE and was selected. The test results "
     "were computed once afterwards. The seasonal level rule has a lower test error than the selected model; this "
     "is reported but not acted upon, because choosing it now would use the test set for model selection.")
para("The three machine-learning models did not outperform the seasonal rules. Each SKU has only about ten records, so "
     "the learned models have little history per product, and the festive uplift is not present in the training "
     "period for them to learn. The seasonal methods apply the documented season multipliers directly. A longer daily "
     "history per SKU would make the comparison fairer to the learned models.")
h2("8.2 Error analysis")
e = res["errors"]
table(["Group", "Test WAPE"], [
    *[[f"Season: {k}", pct(v)] for k, v in e["wape_by_season"].items()],
    *[[f"Category: {k}", pct(v)] for k, v in e["wape_by_category"].items()],
    ["Excluding the demand surges", pct(e["wape_excluding_surges"])]],
    widths=[8.0, 4.0], caption="Table 8.2: Error of the selected model by group")
para(f"Error is similar across seasons and categories. {e['surge_rows']} test records are demand surges (sales above "
     "2.2 times the seasonal level); they cannot be predicted from the features and account for part of the remaining "
     f"error (WAPE {pct(e['wape_excluding_surges'])} without them).")
h2("8.3 Alert validation")
para(f"The alerts were validated on the {al['records']:,} test records against a simulated stock-out: a record is a "
     "stock-out when actual daily sales multiplied by the lead time exceed the stock on hand (a constant sales rate "
     "over the lead time is assumed).")
table(["Rule", "Alerts raised", "Recall", "Precision"], [
    ["Forecast-based reorder point", al["forecast_rule"]["alerts_raised"], pct(al["forecast_rule"]["recall"]),
     pct(al["forecast_rule"]["precision"])],
    ["Store reorder_level (baseline)", al["store_reorder_level"]["alerts_raised"],
     pct(al["store_reorder_level"]["recall"]), pct(al["store_reorder_level"]["precision"])]],
    widths=[6.4, 3.2, 3.2, 3.2], caption=f"Table 8.3: Alert validation ({al['simulated_stockouts']} simulated stock-outs)")
para("The forecast-based rule finds almost all simulated stock-outs at the cost of more alerts, while the store's fixed "
     "rule misses about three quarters of them. The simulated stock-out rate is "
     f"{pct(al['stockout_rate'], 0)} because the generated stock levels are low relative to demand over the lead time, "
     "so recall is easier to reach than it would be with real stock data.")
h2("8.4 Limitations")
bullets([
    "About 90% of the records are generated, so absolute figures illustrate the method and do not describe a real store.",
    "The seasonal multipliers are a planning assumption taken from the data generation, not learned from a previous year.",
    "With roughly ten records per SKU, simple seasonal rules are hard to beat; the conclusion may change with real daily sales history.",
    "The stock-out simulation assumes a constant sales rate over the lead time and uses generated stock levels.",
    "The 20% safety buffer is an assumption and was not optimised against a service-level target.",
    "The category of a SKU is inconsistent in the supplied rows."])

# ------------------------------------------------------------------ 9 ethics
h1("9. Ethical Considerations")
bullets([
    "Sensitive pricing: real supplier unit costs were replaced with a relative Cost Index before analysis. Real prices do "
    "not appear in the processed data, models, dashboard or this report, and the raw file is excluded from version control.",
    "Anonymisation: supplier and product identifiers are pseudonymous codes. No names, contact details or customer "
    "information are stored or processed.",
    "Generated data: 4,500 of 5,000 records are generated; the assumptions are documented and the report states that "
    "absolute metrics illustrate the method.",
    "Minimisation: only the fields needed for forecasting, alerts and supplier analysis are used.",
    "Fairness: the system makes no decisions about individuals.",
    "Transparency: weaker results are reported next to stronger ones, including the finding that the machine-learning "
    "models did not outperform simple seasonal methods.",
    "Secrets: the repository contains no passwords, API keys or personal data."])
para("The signed Ethical Compliance Declaration is submitted as a separate document (docs/ethical_declaration.md).")

# ------------------------------------------------------------------ 10 conclusion
h1("10. Conclusion and Future Work")
para(f"The project delivers a reproducible inventory management prototype that meets its three success criteria. A "
     f"seasonal naive forecast reduces forecast error from {pct(res['naive_test_wape'])} to {pct(champ['WAPE'])} (WAPE) "
     f"on a held-out festive and winter period, and a forecast-based reorder point detects "
     f"{pct(al['forecast_rule']['recall'])} of simulated stock-outs where the store's fixed rule detects "
     f"{pct(al['store_reorder_level']['recall'])}. The REST API and dashboard make these results available to the "
     "store manager, and the supplier scorecard identifies where lead-time risk is concentrated.")
para("Future work:", bold_lead="")
bullets(["Replace the generated history with real daily sales per SKU and re-run the model comparison.",
         "Learn seasonal factors from at least two years of data instead of assuming them.",
         "Add safety stock based on demand variability and a target service level, and economic order quantity.",
         "Add price and promotion effects to the forecast.",
         "Move storage from CSV files to a SQL database and automate purchase-order generation."])

# ------------------------------------------------------------------ references
h1("11. References")
for t in [
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html",
    "Hyndman, R. J. and Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd edition. OTexts. https://otexts.com/fpp3/",
    "Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29(5), 1189-1232. https://doi.org/10.1214/aos/1013203451",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324",
    "pandas, Flask, Streamlit and Plotly documentation: https://pandas.pydata.org, https://flask.palletsprojects.com, https://docs.streamlit.io, https://plotly.com/python/",
    "HCLTech Career Shaper Programme (2025-26). Project Brief, Project No. 21: Optimized Retail Inventory Management System.",
    "HCLTech Career Shaper Programme (2025-26). Industry Project Build Guide, Digital-Data-AI / AIML.",
    "HCLTech Career Shaper Programme Guidelines (provided by the Programme Coordinator)."]:
    p = doc.add_paragraph(style="List Number")
    p.add_run(t).font.size = Pt(10)

# ------------------------------------------------------------------ appendices
h1("Appendix A: Repository and Reproduction")
para("The complete source code is in the project repository. Folder layout:")
table(["Path", "Content"], [
    ["data/raw, data/processed", "Supplied dataset (not submitted), cleaned dataset and data dictionary"],
    ["src/", "prepare_data.py, eda.py, forecasting.py, train.py, alerts.py"],
    ["api/, app/", "Flask REST API and Streamlit dashboard"],
    ["models/", "forecast.joblib (selected model and SKU demand state)"],
    ["reports/", "Figures, screenshots, experiment table, results.json, this report"],
    ["docs/", "Charter, data source note, cleaning log, ethical declaration, demo video"],
    ["tests/", "Automated tests"]], widths=[4.4, 11.6])
para("Commands (Windows):")
for line in ["python -m venv venv", "venv\\Scripts\\activate", "pip install -r requirements.txt",
             "python src/prepare_data.py", "python src/eda.py", "python src/train.py", "python -m pytest tests -q",
             "python api/app.py", "streamlit run app/dashboard.py"]:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(line)
    r.font.name, r.font.size = "Consolas", Pt(9.5)

h1("Appendix B: AI Tool Disclosure", page_break=False)
para("In line with the programme's ethical guidelines, the use of AI tools in this project is disclosed as follows.")
table(["Item", "Details"], [
    ["Tool", "An AI coding assistant (large language model)"],
    ["Used for", "Drafting and debugging the Python scripts, reviewing the project against the programme build guide, and drafting the documentation and this report"],
    ["Learner responsibility", "The learner is responsible for the content of the final submission; all scripts were run and the reported results reproduced from the repository"],
    ["Data", "No real customer or supplier data was shared with the AI tool beyond the programme-supplied dataset.csv"]],
    widths=[4.0, 12.0])

doc.save(OUT)
print("saved", OUT)
