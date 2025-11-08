# NEET Question Extraction Pipeline

A robust ETL (Extract, Transform, Load) pipeline for extracting NEET multiple-choice questions from PDF files, handling incomplete extractions, and validating data quality.

## 📋 Overview

This pipeline processes 47 PDF files containing ~200 NEET questions each (9,400 total) and produces a clean, validated dataset ready for further use. The pipeline includes smart detection of image-based questions, incomplete extractions, and automatic reprocessing of problematic questions.

## 🔄 Pipeline Architecture

```
Raw PDFs (47 files)
    ↓
[1] Extract Questions → Test_Extraction_Raw.jsonl (9,400 questions)
    ↓
[2] Check Incomplete → extraction_report.json (identifies ~573 incomplete)
    ↓
[3] Reprocess → Test_Extraction_Fixed.jsonl (fixes ~559, 97.6% success)
    ↓
[4] Validate → Test_Extraction_Fixed_valid.jsonl (8,852 valid questions)
```

## 📁 Scripts

### Core Pipeline (4 scripts)
1. **`process_questions_simple.py`** - Extract questions from PDFs
2. **`check_extraction_incomplete.py`** - Detect incomplete extractions
3. **`reprocess_incomplete.py`** - Fix incomplete questions
4. **`validate_extraction.py`** - Filter valid questions

### Utilities (2 scripts)
- **`extract_missing_questions.py`** - Compare OLD vs NEW extractions
- **`split_jsonl.py`** - Split JSONL into chunks

## 🚀 Quick Start

### Run Complete Pipeline (4 Steps)

**Step 1: Extract** (9,400 questions)
```cmd
.venv\Scripts\python.exe "aneeta_v2\scripts\Data Extraction\process_questions_simple.py" --input "Raw Data\QuestionBank" --output "aneeta_v2\Processed Data\Test_Extraction_Raw.jsonl"
```

**Step 2: Check** (~573 incomplete found)
```cmd
.venv\Scripts\python.exe "aneeta_v2\scripts\Data Extraction\check_extraction_incomplete.py" "aneeta_v2\Processed Data\Test_Extraction_Raw.jsonl" --report "aneeta_v2\Processed Data\extraction_report.json"
```

**Step 3: Reprocess** (559 fixed, 97.6% success)
```cmd
.venv\Scripts\python.exe "aneeta_v2\scripts\Data Extraction\reprocess_incomplete.py" --pdf-dir "Raw Data\QuestionBank" --extraction "aneeta_v2\Processed Data\Test_Extraction_Raw.jsonl" --report "aneeta_v2\Processed Data\extraction_report.json" --output "aneeta_v2\Processed Data\Test_Extraction_Fixed.jsonl"
```

**Step 4: Validate** (8,852 valid, 548 filtered)
```cmd
.venv\Scripts\python.exe "aneeta_v2\scripts\Data Extraction\validate_extraction.py" "aneeta_v2\Processed Data\Test_Extraction_Fixed.jsonl" --output-dir "aneeta_v2\Processed Data"
```

---

### Compare OLD vs NEW Extractions

```cmd
.venv\Scripts\python.exe "aneeta_v2\scripts\Data Extraction\extract_missing_questions.py"
```
**Output**: `New_Questions_Added.jsonl` - 977 new valid questions added to dataset

## 📊 Results (November 4, 2025)

| Stage | Count | Rate |
|-------|-------|------|
| Raw Extraction | 9,400 | 47 PDFs × 200 questions |
| Incomplete Found | 573 | 6.1% |
| Successfully Fixed | 559 | 97.6% fix rate |
| Final Valid | 8,852 | 94.2% valid |
| Filtered Out | 548 | 5.8% (image-based) |
| **New vs OLD** | **+977** | **12.4% growth** |

## 🔧 Key Features

### Parser Fixes
1. **No-space options**: `\s*` handles `(2)More` without space
2. **Unmarked option 3**: Detects missing `(3)` marker between `(2)` and `(4)`

### Validation Logic
- Image-based detection (metadata flag, keywords, empty options)
- Incomplete extraction filtering
- Missing field validation (text, answer, options)

## 📝 Output Format

**JSONL**: One JSON object per line
```json
{
  "id": "Question_Paper_1_p2_q13",
  "question_number": "13",
  "page": 2,
  "source": "Question_Paper_1",
  "type": "mcq",
  "extracted_text": "What is the SI unit of force?",
  "options": [
    {"label": "1", "text": "Newton"},
    {"label": "2", "text": "Joule"},
    {"label": "3", "text": "Watt"},
    {"label": "4", "text": "Pascal"}
  ],
  "answer": "1",
  "explanation": "Force is measured in Newtons (N).",
  "metadata": {
    "option_count": 4,
    "is_image_based": false,
    "has_incomplete_extraction": false,
    "reprocessed": false
  }
}
```

## 🛠️ Requirements

**Python**: 3.13+  
**Dependency**: PyMuPDF (fitz)

```cmd
.venv\Scripts\pip install PyMuPDF
```

## 📂 Files

```
scripts/Data Extraction/
├── process_questions_simple.py          # Step 1: Extract
├── check_extraction_incomplete.py       # Step 2: Check
├── reprocess_incomplete.py              # Step 3: Reprocess
├── validate_extraction.py               # Step 4: Validate
├── extract_missing_questions.py         # Compare extractions
├── split_jsonl.py                       # Split utility
└── README.md

Processed Data/
├── Test_Extraction_Raw.jsonl            # 9,400 raw
├── extraction_report.json               # 573 incomplete
├── Test_Extraction_Fixed.jsonl          # 9,400 fixed
├── Test_Extraction_Fixed_valid.jsonl    # 8,852 valid ✅
├── Test_Extraction_Fixed_invalid_and_image.jsonl  # 548 filtered
└── New_Questions_Added.jsonl            # 977 new questions
```

## 💡 Best Practices

1. Run complete pipeline (don't skip steps)
2. Check `extraction_report.json` for patterns
3. Verify reprocessing fixes >95% of incomplete
4. Keep intermediate files for debugging

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Low extraction count | Verify 47 PDFs in `Raw Data/QuestionBank` |
| High incomplete (>10%) | Check PDF quality/format changes |
| Reprocessing fails | Verify `--pdf-dir` path matches report |
| Too many filtered | Adjust validation thresholds (lines 52-71) |

## ⚡ Performance

**Total time**: ~3-5 minutes for complete pipeline
- Extract: 2-3 min
- Check: 5 sec
- Reprocess: 30-60 sec
- Validate: 10 sec

---

**Version**: 2.0 | **Last Updated**: November 4, 2025 | **Pipeline**: Extract → Check → Reprocess → Validate
