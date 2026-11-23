
# ==========================================
# 0. IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET = "CohereLabs/Global-MMLU"

# Languages to load
LANGUAGES = [
    "en",
    "vi",
    # "fr",
    # "de",
    # "ja",
]

# Select "all" or a list of splits, such as ["dev", "test"]
SPLITS = "all"

# Columns to analyze
COLUMNS = {
    "label": "cultural_sensitivity_label",
    "subject": "subject",
    "category": "subject_category"
}

# Output directories
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
IMG_DIR = os.path.join(OUTPUT_DIR, "img")

os.makedirs(IMG_DIR, exist_ok=True)

EXCEL_PATH = os.path.join(
    OUTPUT_DIR,
    "Global_MMLU_Statistics.xlsx"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_global_mmlu(dataset_name, languages, splits):
    """
    Purpose:
        Load Global-MMLU for the selected languages and combine the
        requested splits into a single DataFrame.

    Args:
        dataset_name (str): Dataset name on Hugging Face.
        languages (list): Language codes to load.
        splits (str/list): "all" or a list of specific splits.

    Returns:
        pd.DataFrame: The combined dataset with language and split columns.
    """

    dataframes = []

    for lang in languages:

        print(f"Loading language: {lang}")

        dataset = load_dataset(dataset_name, lang)

        selected_splits = (
            list(dataset.keys())
            if splits == "all"
            else splits
        )

        for split in selected_splits:

            if split not in dataset:
                print(f"Warning: {split} not found in {lang}")
                continue

            df = dataset[split].to_pandas()

            df["language"] = lang
            df["split"] = split

            dataframes.append(df)

    if not dataframes:
        raise ValueError("No dataset samples were loaded!")

    return pd.concat(dataframes, ignore_index=True)


# ==========================================
# 3. STATISTICS
# ==========================================

def get_statistics(df, column):
    """
    Purpose:
        Count each distinct value in a column and calculate its percentage.

    Args:
        df (pd.DataFrame): The input dataset.
        column (str): The column to summarize.

    Returns:
        pd.DataFrame: A table containing each value, its item count, and percentage.
    """

    values = (
        df[column]
        .fillna("UNLABELED")
        .astype(str)
        .str.strip()
        .replace("", "UNLABELED")
    )

    counts = values.value_counts()

    result = pd.DataFrame({
        "items": counts,
        "percentage": counts / counts.sum() * 100
    })

    result.index.name = column

    return result.reset_index()


def get_language_statistics(df, column):
    """
    Purpose:
        Summarize a column's distribution separately for each language.

    Args:
        df (pd.DataFrame): The input dataset.
        column (str): The column to summarize.

    Returns:
        pd.DataFrame: The summary table with an additional language column.
    """

    grouped = []

    for lang, group in df.groupby("language"):

        temp = get_statistics(group, column)
        temp.insert(0, "language", lang)

        grouped.append(temp)

    return pd.concat(grouped, ignore_index=True)


# ==========================================
# 4. OVERVIEW
# ==========================================

def create_overview(df, languages, dataset_name):
    """
    Purpose:
        Create a dataset overview with sample and language counts and the
        distribution of CS/CA labels.

    Args:
        df (pd.DataFrame): The combined dataset.
        languages (list): The selected languages.
        dataset_name (str): The dataset name.

    Returns:
        tuple: The overview table and the per-language sample-count table.
    """

    labels = (
        df["cultural_sensitivity_label"]
        .fillna("UNLABELED")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    overview = pd.DataFrame({
        "Metric": [
            "Dataset",
            "Total samples",
            "Number of languages",
            "Languages",
            "Splits",
            "Number of labels",
            "CS samples",
            "CA samples"
        ],
        "Value": [
            dataset_name,
            len(df),
            len(languages),
            ", ".join(languages),
            ", ".join(sorted(df["split"].unique())),
            labels.nunique(),
            (labels == "CS").sum(),
            (labels == "CA").sum()
        ]
    })

    language_overview = (
        df.groupby("language")
        .size()
        .reset_index(name="items")
    )

    language_overview["percentage"] = (
        language_overview["items"]
        / language_overview["items"].sum() * 100
    )

    return overview, language_overview


# ==========================================
# 5. BAR CHART
# ==========================================

def plot_bar_chart(df, column, title, filename, top_n=None):
    """
    Purpose:
        Plot a horizontal bar chart showing the count and percentage
        for each data group.

    Args:
        df (pd.DataFrame): The summary table.
        column (str): The column containing group names.
        title (str): The chart title.
        filename (str): The output image filename.
        top_n (int/None): If specified, show only the top N groups.
    """

    plot_df = df.copy()

    if top_n is not None:
        plot_df = plot_df.nlargest(top_n, "items")

    plot_df = plot_df.sort_values("items")

    fig, ax = plt.subplots(
        figsize=(12, max(5, len(plot_df) * 0.35))
    )

    bars = ax.barh(
        plot_df[column].astype(str),
        plot_df["items"]
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of items")
    ax.set_ylabel(column.capitalize())

    for bar, (_, row) in zip(bars, plot_df.iterrows()):

        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {int(row['items']):,} ({row['percentage']:.1f}%)",
            va="center",
            fontsize=8
        )

    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(IMG_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ==========================================
# 6. PIE CHART
# ==========================================

def plot_pie_chart(df, column, title, filename, top_n=None):
    """
    Purpose:
        Plot a pie chart showing each data group's share of the total samples.

    Args:
        df (pd.DataFrame): The summary table.
        column (str): The column containing group names.
        title (str): The chart title.
        filename (str): The output image filename.
        top_n (int/None): Show only the top N groups; combine the remainder
            into "Other".
    """

    plot_df = df.copy()

    if top_n is not None and len(plot_df) > top_n:

        plot_df = plot_df.nlargest(top_n, "items").copy()

        other_items = df["items"].sum() - plot_df["items"].sum()

        if other_items > 0:

            other_row = pd.DataFrame([{
                column: "Other",
                "items": other_items,
                "percentage": other_items / df["items"].sum() * 100
            }])

            plot_df = pd.concat(
                [plot_df, other_row],
                ignore_index=True
            )

    fig, ax = plt.subplots(figsize=(9, 8))

    labels = [
        f"{name} ({pct:.1f}%)"
        for name, pct in zip(
            plot_df[column],
            plot_df["percentage"]
        )
    ]

    ax.pie(
        plot_df["items"],
        labels=labels,
        startangle=90,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        pctdistance=0.75
    )

    ax.set_title(title, fontsize=15, fontweight="bold")

    plt.tight_layout()

    plt.savefig(
        os.path.join(IMG_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ==========================================
# 7. EXPORT TO EXCEL
# ==========================================

def export_excel(excel_path, data, overview,
                 language_overview, stats, language_stats):
    """
    Purpose:
        Export the dataset and summary tables to a multi-sheet Excel workbook.

    Args:
        excel_path (str): Path to the Excel workbook.
        data (pd.DataFrame): The combined raw dataset.
        overview (pd.DataFrame): The overview table.
        language_overview (pd.DataFrame): Per-language summary statistics.
        stats (dict): Overall summary tables.
        language_stats (dict): Per-language summary tables.
    """

    excel_data = data.copy()

    # Convert lists and dictionaries to strings for Excel compatibility.
    for col in excel_data.columns:

        if excel_data[col].dtype == "object":

            excel_data[col] = excel_data[col].apply(
                lambda x: str(x)
                if isinstance(x, (list, dict))
                else x
            )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        overview.to_excel(
            writer,
            sheet_name="Overview",
            index=False,
            startrow=0
        )

        language_overview.to_excel(
            writer,
            sheet_name="Overview",
            index=False,
            startrow=len(overview) + 3
        )

        excel_data.to_excel(
            writer,
            sheet_name="Raw_Data",
            index=False
        )

        # Overall statistics
        for name, df in stats.items():

            df.to_excel(
                writer,
                sheet_name=name.capitalize(),
                index=False
            )

        # Per-language statistics
        for name, df in language_stats.items():

            df.to_excel(
                writer,
                sheet_name=f"{name.capitalize()}_by_Lang",
                index=False
            )

        # Format the workbook.
        for ws in writer.sheets.values():

            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:

                cell.font = Font(
                    bold=True,
                    color="FFFFFF"
                )

                cell.fill = PatternFill(
                    "solid",
                    fgColor="244062"
                )

                cell.alignment = Alignment(
                    horizontal="center"
                )

            for col_cells in ws.columns:

                max_length = max(
                    len(str(cell.value or ""))
                    for cell in list(col_cells)[:500]
                )

                col_letter = get_column_letter(
                    col_cells[0].column
                )

                ws.column_dimensions[col_letter].width = min(
                    max_length + 3,
                    40
                )


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

# Load dataset
data = load_global_mmlu(
    DATASET,
    LANGUAGES,
    SPLITS
)

print("\nTotal samples:", len(data))

# Overall statistics
stats = {
    name: get_statistics(data, column)
    for name, column in COLUMNS.items()
}

# Per-language statistics
language_stats = {
    name: get_language_statistics(data, column)
    for name, column in COLUMNS.items()
}

# Dataset overview
overview, language_overview = create_overview(
    data,
    LANGUAGES,
    DATASET
)

# ==========================================
# 9. CREATE CHARTS
# ==========================================

# Label
plot_bar_chart(
    stats["label"],
    "cultural_sensitivity_label",
    "Global-MMLU Label Distribution",
    "label_bar.png"
)

plot_pie_chart(
    stats["label"],
    "cultural_sensitivity_label",
    "Global-MMLU Label Distribution",
    "label_pie.png"
)

# Subject
plot_bar_chart(
    stats["subject"],
    "subject",
    "Global-MMLU Subject Distribution",
    "subject_bar.png",
    top_n=25
)

plot_pie_chart(
    stats["subject"],
    "subject",
    "Global-MMLU Subject Distribution",
    "subject_pie.png",
    top_n=10
)

# Category
plot_bar_chart(
    stats["category"],
    "subject_category",
    "Global-MMLU Category Distribution",
    "category_bar.png"
)

plot_pie_chart(
    stats["category"],
    "subject_category",
    "Global-MMLU Category Distribution",
    "category_pie.png"
)

# Charts for each language
for name, column in COLUMNS.items():

    df = language_stats[name]

    for lang in LANGUAGES:

        lang_df = df[df["language"] == lang]

        plot_bar_chart(
            lang_df,
            column,
            f"{name.capitalize()} Distribution - {lang}",
            f"{name}_{lang}_bar.png",
            top_n=25
        )

        plot_pie_chart(
            lang_df,
            column,
            f"{name.capitalize()} Distribution - {lang}",
            f"{name}_{lang}_pie.png",
            top_n=10
        )


# ==========================================
# 10. EXPORT EXCEL
# ==========================================

export_excel(
    EXCEL_PATH,
    data,
    overview,
    language_overview,
    stats,
    language_stats
)

# ==========================================
# 11. COMPLETION
# ==========================================

print("\n========== EXPORT COMPLETED ==========")

print("Excel file:", EXCEL_PATH)
print("Image directory:", IMG_DIR)

print("\nGenerated files:")

for root, dirs, files in os.walk(OUTPUT_DIR):

    for file in files:
        print(os.path.join(root, file))

print("\nTotal samples:", len(data))
print("Total languages:", len(LANGUAGES))
print("Excel sheets: 8")
