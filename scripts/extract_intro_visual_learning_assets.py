import argparse
import re
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
from PIL import Image


PDF_CROP_BOXES = {
    11: {
        "page11_loss_x2_plot.png": (18, 165, 478, 426),
        "page11_loss_x3_plot.png": (492, 165, 955, 426),
        "page11_loss_x4_plot.png": (968, 165, 1428, 426),
        "page11_loss_table.png": (250, 475, 740, 650),
    },
    12: {
        "page12_psnr_x2_plot.png": (18, 132, 478, 395),
        "page12_psnr_x3_plot.png": (492, 132, 955, 395),
        "page12_psnr_x4_plot.png": (968, 132, 1428, 395),
        "page12_psnr_table.png": (100, 438, 1000, 714),
    },
    13: {
        "page13_ssim_x2_plot.png": (18, 132, 478, 401),
        "page13_ssim_x3_plot.png": (492, 132, 955, 401),
        "page13_ssim_x4_plot.png": (968, 132, 1428, 401),
        "page13_ssim_table.png": (80, 440, 1010, 712),
    },
    14: {
        "page14_realsr_psnr_x2_plot.png": (55, 84, 494, 324),
        "page14_realsr_psnr_x3_plot.png": (503, 84, 940, 324),
        "page14_realsr_psnr_x4_plot.png": (949, 84, 1385, 324),
        "page14_realsr_ssim_x2_plot.png": (55, 335, 494, 575),
        "page14_realsr_ssim_x3_plot.png": (503, 335, 940, 575),
        "page14_realsr_ssim_x4_plot.png": (949, 335, 1385, 575),
        "page14_realsr_table.png": (40, 520, 1110, 760),
    },
}


SLIDE_TABLES = {
    "loss": {
        "headers": ["Model", "Training", "Validation"],
        "rows": [
            ["X2", "0.0166", "0.015"],
            ["X3", "0.0228", "0.0225"],
            ["X4", "0.0288", "0.0264"],
        ],
    },
    "df2k_psnr": {
        "headers": [
            "Model",
            "BSD100 Max",
            "BSD100 Final",
            "Set5 Max",
            "Set5 Final",
            "Set14 Max",
            "Set14 Final",
            "Urban100 Max",
            "Urban100 Final",
        ],
        "rows": [
            ["X2", "30.77", "30.77", "35.59", "35.59", "31.52", "31.52", "27.88", "27.88"],
            ["X3", "27.94", "27.94", "31.73", "31.73", "28.38", "28.38", "N/A", "N/A"],
            ["X4", "26.79", "26.77", "30.13", "30.1", "27.01", "26.99", "24.16", "24.15"],
        ],
    },
    "df2k_ssim": {
        "headers": [
            "Model",
            "BSD100 Max",
            "BSD100 Final",
            "Set5 Max",
            "Set5 Final",
            "Set14 Max",
            "Set14 Final",
            "Urban100 Max",
            "Urban100 Final",
        ],
        "rows": [
            ["X2", "0.8925", "0.8925", "0.9538", "0.9538", "0.9096", "0.9096", "0.8884", "0.8884"],
            ["X3", "0.7945", "0.7945", "0.9047", "0.9047", "0.8254", "0.8254", "N/A", "N/A"],
            ["X4", "0.7289", "0.7278", "0.8693", "0.8689", "0.7667", "0.766", "0.7284", "0.7285"],
        ],
    },
    "realsr_psnr": {
        "headers": ["Model", "BSD100", "Set5", "Set14", "Urban100"],
        "rows": [
            ["X2", "28.87", "31.79", "29.24", "25.58"],
            ["X3", "26.07", "28.15", "26.36", "N/A"],
            ["X4", "25.29", "27.13", "25.36", "22.71"],
        ],
    },
    "realsr_ssim": {
        "headers": ["Model", "BSD100", "Set5", "Set14", "Urban100"],
        "rows": [
            ["X2", "0.8695", "0.9174", "0.8807", "0.8494"],
            ["X3", "0.7656", "0.8408", "0.7867", "N/A"],
            ["X4", "0.7141", "0.8205", "0.742", "0.7"],
        ],
    },
}


TRAIN_RE = re.compile(r"Epoch\s+(\d+),\s+Loss:\s+([0-9.]+)")
VAL_RE = re.compile(
    r"Epoch\s+(\d+),\s+Loss:\s+([0-9.]+),\s+PSNR:\s+([0-9.]+),\s+SSIM:\s+([0-9.]+)"
)


def render_page(pdf_path: Path, page_number: int, zoom: float = 2.0) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_pdf_crops(pdf_path: Path, crops_dir: Path) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages = {}

    for page_number in PDF_CROP_BOXES:
        rendered_pages[page_number] = render_page(pdf_path, page_number)

    for page_number, crop_map in PDF_CROP_BOXES.items():
        page_img = rendered_pages[page_number]
        for name, box in crop_map.items():
            crop = page_img.crop(box)
            crop.save(crops_dir / name)


def parse_train_log(path: Path):
    epochs = []
    losses = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = TRAIN_RE.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return epochs, losses


def parse_val_log(path: Path):
    epochs = []
    losses = []
    psnrs = []
    ssims = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = VAL_RE.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                psnrs.append(float(match.group(3)))
                ssims.append(float(match.group(4)))
    return epochs, losses, psnrs, ssims


def plot_loss_panels(results_dir: Path, dataset_name: str, scales, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, scale in zip(axes, [2, 3, 4]):
        train_log = results_dir / f"{scale}x" / "results" / "train_log.txt"
        val_log = results_dir / f"{scale}x" / "results" / "val_log.txt"

        if not train_log.exists() or not val_log.exists():
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"No checked-in {dataset_name} x{scale} logs",
                ha="center",
                va="center",
                fontsize=12,
            )
            continue

        train_epochs, train_losses = parse_train_log(train_log)
        val_epochs, val_losses, _, _ = parse_val_log(val_log)

        ax.plot(train_epochs, train_losses, marker="o", markersize=2, linewidth=1, label="Train Loss")
        ax.plot(val_epochs, val_losses, marker="o", markersize=3, linewidth=1, label="Validation Loss")
        ax.set_title(f"{dataset_name} x{scale}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"{dataset_name} Training and Validation Loss from Checked-In Logs", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def markdown_table(headers, rows):
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider] + row_lines)


def write_markdown(output_path: Path) -> None:
    lines = [
        "# Intro_To_Visual_Learning Extraction",
        "",
        "Source PDF: `/mntdatalora/src/LightSR/Intro_To_Visual_Learning.pdf`",
        "",
        "This file contains the tables transcribed from the slide deck plus links to the cropped PNG assets and the loss curves regenerated from checked-in logs.",
        "",
        "## Slide 11: DIV2K Training and Validation Loss",
        "",
        markdown_table(SLIDE_TABLES["loss"]["headers"], SLIDE_TABLES["loss"]["rows"]),
        "",
        "Cropped assets:",
        "- `pdf_crops/page11_loss_x2_plot.png`",
        "- `pdf_crops/page11_loss_x3_plot.png`",
        "- `pdf_crops/page11_loss_x4_plot.png`",
        "- `pdf_crops/page11_loss_table.png`",
        "",
        "Preview:",
        "![Slide 11 loss table](pdf_crops/page11_loss_table.png)",
        "",
        "Redrawn from logs:",
        "- `log_plots/df2k_train_val_loss_from_logs.png`",
        "",
        "![DF2K loss curves from logs](log_plots/df2k_train_val_loss_from_logs.png)",
        "",
        "## Slide 12: DIV2K PSNR Table",
        "",
        markdown_table(SLIDE_TABLES["df2k_psnr"]["headers"], SLIDE_TABLES["df2k_psnr"]["rows"]),
        "",
        "Cropped assets:",
        "- `pdf_crops/page12_psnr_x2_plot.png`",
        "- `pdf_crops/page12_psnr_x3_plot.png`",
        "- `pdf_crops/page12_psnr_x4_plot.png`",
        "- `pdf_crops/page12_psnr_table.png`",
        "",
        "Preview:",
        "![Slide 12 PSNR table](pdf_crops/page12_psnr_table.png)",
        "",
        "## Slide 13: DIV2K SSIM Table",
        "",
        markdown_table(SLIDE_TABLES["df2k_ssim"]["headers"], SLIDE_TABLES["df2k_ssim"]["rows"]),
        "",
        "Cropped assets:",
        "- `pdf_crops/page13_ssim_x2_plot.png`",
        "- `pdf_crops/page13_ssim_x3_plot.png`",
        "- `pdf_crops/page13_ssim_x4_plot.png`",
        "- `pdf_crops/page13_ssim_table.png`",
        "",
        "Preview:",
        "![Slide 13 SSIM table](pdf_crops/page13_ssim_table.png)",
        "",
        "## Slide 14: RealSR PSNR Table",
        "",
        markdown_table(SLIDE_TABLES["realsr_psnr"]["headers"], SLIDE_TABLES["realsr_psnr"]["rows"]),
        "",
        "## Slide 14: RealSR SSIM Table",
        "",
        markdown_table(SLIDE_TABLES["realsr_ssim"]["headers"], SLIDE_TABLES["realsr_ssim"]["rows"]),
        "",
        "Cropped assets:",
        "- `pdf_crops/page14_realsr_psnr_x2_plot.png`",
        "- `pdf_crops/page14_realsr_psnr_x3_plot.png`",
        "- `pdf_crops/page14_realsr_psnr_x4_plot.png`",
        "- `pdf_crops/page14_realsr_ssim_x2_plot.png`",
        "- `pdf_crops/page14_realsr_ssim_x3_plot.png`",
        "- `pdf_crops/page14_realsr_ssim_x4_plot.png`",
        "- `pdf_crops/page14_realsr_table.png`",
        "",
        "Preview:",
        "![Slide 14 RealSR table](pdf_crops/page14_realsr_table.png)",
        "",
        "Redrawn from logs:",
        "- `log_plots/realsr_train_val_loss_from_logs.png`",
        "",
        "![REALSR loss curves from logs](log_plots/realsr_train_val_loss_from_logs.png)",
        "",
        "## Notes",
        "",
        "- The `DF2K/4x` train and validation logs are not checked into this repository, so the x4 loss chart is available only as a PDF crop.",
        "- The `page14_realsr_psnr_x3_plot.png` crop preserves the source slide exactly. In the source PDF, that embedded chart appears upside down.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Extract tables and chart crops from Intro_To_Visual_Learning.pdf.")
    parser.add_argument(
        "--pdf",
        default="/mntdatalora/src/LightSR/Intro_To_Visual_Learning.pdf",
        help="Path to the source PDF.",
    )
    parser.add_argument(
        "--df2k-results",
        default="/mntdatalora/src/LightSR/results/DF2K",
        help="Path to the checked-in DF2K results directory.",
    )
    parser.add_argument(
        "--realsr-results",
        default="/mntdatalora/src/LightSR/results/REALSR",
        help="Path to the checked-in REALSR results directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mntdatalora/src/LightSR/external_assets/ucsd_downloads/slides/intro_to_visual_learning_extracted",
        help="Directory where PNG crops, regenerated plots, and the markdown file will be written.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "pdf_crops"
    plots_dir = output_dir / "log_plots"
    markdown_path = output_dir / "intro_to_visual_learning_extract.md"

    extract_pdf_crops(pdf_path, crops_dir)
    plot_loss_panels(Path(args.df2k_results), "DF2K", [2, 3, 4], plots_dir / "df2k_train_val_loss_from_logs.png")
    plot_loss_panels(Path(args.realsr_results), "REALSR", [2, 3, 4], plots_dir / "realsr_train_val_loss_from_logs.png")
    write_markdown(markdown_path)

    print(f"Wrote assets to: {output_dir}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
