# 🎯 Tracking Analysis Tool

A standalone application for analyzing psychophysics tracking data from blob tracking experiments with auditory feedback conditions.

---

## 📥 Download & Install

### Windows Executable (No Python Required!)

1. **Download** [`TrackingAnalysis.rar`](dist/TrackingAnalysis.rar) from this repository
2. **Extract** the RAR file to any folder
3. **Run** `TrackingAnalysis.exe`
4. Your browser will open automatically at `http://localhost:8501`

> 💡 **Tip**: Keep all files in the extracted folder together - the executable needs them to run.

---

## ✨ Features

- **Drag & Drop Upload** - Upload CSV files directly in the browser
- **Folder Path Input** - Load data from a local directory
- **Cross-correlation analysis** - Detect predictive vs reactive tracking
- **Statistical analysis** - Mann-Whitney U, Kruskal-Wallis, Spearman correlations
- **Rich visualizations** - Interactive plots, heatmaps, animated replays
- **Export results** - JSON and Markdown reports

---

## 📊 Research Questions

This tool helps answer:

1. Does auditory feedback improve tracking accuracy? (dynamic vs static)
2. Can observers discriminate blobs with different SD sizes? (21, 31, 34 arcmin)
3. Is tracking predictive or reactive? (cross-correlation lag analysis)
4. Are there individual differences in tracking ability?

---

## 📁 Data Format

### File Naming Convention
```
Participant_XXXX_Tracking_blob_experiment_XXarcmin_vX_dynamic.csv
Participant_XXXX_Tracking_blob_experiment_XXarcmin_vX_static.csv
```

### Required CSV Columns

| Column | Description |
|--------|-------------|
| `Frame` | Frame number (0-indexed) |
| `Target_X` | Target X position (pixels) |
| `Target_Y` | Target Y position (pixels) |
| `Mouse_X` | Mouse/response X position (pixels) |
| `Mouse_Y` | Mouse/response Y position (pixels) |

---

## 📖 Usage Guide

1. **Load Data** - Drag & drop CSV files or enter folder path
2. **Run Analysis** - Click ▶️ Run Analysis (uses default settings)
3. **View Results** - Explore metrics, visualizations, and statistics
4. **Export** - Download results as JSON or Markdown

---

## 📝 License

MIT License

## 👥 Authors

Psychophysics and Reaction Time Methods - Group Project

---

<p align="center">
  Built with ❤️ using <a href="https://streamlit.io/">Streamlit</a>
</p>
