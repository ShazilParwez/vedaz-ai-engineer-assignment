# Vedaz AI Engineer Assignment

## Project Overview

The goal of this project is to build an automated, robust pipeline to generate, validate, and evaluate synthetic chat data for **Vedaz**, an AI Astrologer. This synthetic dataset is designed for fine-tuning large language models to ensure they act compassionately, honestly, and strictly adhere to Vedaz's safety principles (e.g., no death/illness predictions, no guarantees of success).

The project contains three major components:
- **Dataset Checker**: Validates structure, detects duplicates, and filters unsafe conversations.
- **Synthetic Data Generator**: Uses the Gemini API to autonomously produce diverse, realistic chat data.
- **Response Evaluator**: Uses an LLM-as-a-Judge approach to score the quality of the astrological advice.

---

## Architecture

```
User Topics
      │
      ▼
generator.py
      │
      ▼
checker.py
      │
      ▼
Generated Dataset
      │
      ▼
evaluator.py
      │
      ▼
evaluation.csv
```

**Workflow:**
1. **Generation:** `generator.py` queries the Gemini API with specific topics and personas to create synthetic user-astrologer conversations.
2. **Validation:** The generated conversations are immediately passed into the `checker.py` logic to ensure structural and safety compliance before being saved to the dataset.
3. **Checking & Splitting:** `checker.py` is run on the full dataset to detect exact and semantic duplicates using Sentence Transformers, subsequently splitting the clean data into `train.jsonl` and `test.jsonl`.
4. **Evaluation:** `evaluator.py` takes the test questions, queries the Gemini model as the AI Astrologer, and then uses a secondary LLM Judge prompt to score the responses across multiple dimensions.

---

## Features

### checker.py
- **Structure validation**: Ensures conversations strictly follow the `system` -> `user` -> `assistant` JSON format.
- **Dataset statistics**: Computes minimum, maximum, and average turn/word counts.
- **Regex safety detection**: Flags severe policy violations (e.g., guaranteed money, fear-selling).
- **Exact duplicate detection**: Uses MD5 hashing to catch identical conversations.
- **Semantic duplicate detection**: Uses Sentence Transformers and cosine similarity to catch near-duplicates.
- **Train/Test split**: Randomly segments valid data for fine-tuning and evaluation.

### generator.py
- **Gemini API integration**: Interfaces with Google's `google-genai` SDK for synthetic data creation.
- **JSON validation**: Strictly enforces valid JSON outputs from the LLM.
- **Retry logic**: Implements exponential backoff for API limits and malformed outputs.
- **Schema validation**: Guarantees the output structure matches the fine-tuning requirements.
- **Safety validation**: Immediately drops generations that fail the Vedaz safety policies.

### evaluator.py
- **LLM-as-a-Judge**: Evaluates model performance autonomously without human bias.
- **Safety scoring**: Rates responses (1-5) on adherence to safety policies.
- **Honesty scoring**: Rates responses on astrological truthfulness.
- **Helpfulness scoring**: Rates the practical utility of the advice.
- **Warmth scoring**: Rates the empathy and compassion of the AI.
- **Astrology limitation scoring**: Rates how naturally the AI mentions the limits of astrology.
- **CSV report generation**: Exports detailed scores and reasoning for every evaluation.

---

## Installation

Ensure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

You must also set your Gemini API key in your environment variables:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**Mac/Linux:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## Usage

### 1. Generate Synthetic Data
Generates realistic conversations on a specific topic.
```bash
python generator.py --topic "career delay" --language Hindi --count 10 --output generated_chats.jsonl
```

### 2. Check and Validate Dataset
Cleans the dataset, removes duplicates, and splits it into train/test sets.
```bash
python checker.py --input generated_chats.jsonl --sim_threshold 0.85
```

### 3. Evaluate Responses
Runs the LLM Judge against the test questions and exports scores.
```bash
python evaluator.py --input test.jsonl --output evaluation.csv
```

---

## Project Structure

- `checker.py`: Core logic for validating data structure, catching safety violations, and removing duplicates.
- `generator.py`: Script to dynamically prompt the Gemini API to build synthetic datasets.
- `evaluator.py`: Script to autonomously evaluate and score the AI Astrologer's responses using an LLM Judge.
- `generated_chats.jsonl`: The raw output dataset containing all generated conversations.
- `checker_report.txt`: A detailed report showing statistics, duplication counts, and validation errors.
- `flagged_chats.json`: Conversations that were caught and rejected by the safety filters.
- `train.jsonl`: Cleaned, validated dataset ready for model fine-tuning (80% split).
- `test.jsonl`: Hold-out set used for evaluating the final model (20% split).
- `evaluation.csv`: The final evaluation report containing 1-5 metric scores and AI Judge reasoning.
- `requirements.txt`: Python package dependencies.
- `README.md`: Project documentation.

---

## Assumptions

- The dataset follows the JSONL format, where each line is a valid JSON object.
- A valid Gemini API key is available in the environment variables.
- Users provide meaningful, distinct topics to the generator to ensure dataset diversity.

---

## Limitations

- **Regex Context Blindness:** Regex heuristics in `checker.py` may flag safe phrases if they coincidentally contain restricted keywords, failing to understand the broader context.
- **Semantic Similarity Thresholds:** A static similarity threshold (e.g., 0.85) may require tuning based on the specific astrological topic to prevent over-filtering or under-filtering.
- **LLM Judge Inconsistencies:** The LLM-as-a-Judge may not always be perfectly consistent in its 1-5 scoring across large datasets.
- **Human Oversight:** Generated conversations and evaluations still strongly benefit from a final human-in-the-loop review.

---

## Future Improvements

- **Better Multilingual Support:** Expand the safety regex and prompt logic to comprehensively support diverse regional dialects.
- **Toxicity Detection:** Integrate a dedicated moderation model (e.g., Perspective API) alongside regex.
- **Human-in-the-loop Review:** Add a simple UI or CLI tool for human annotators to approve/reject generated data.
- **Better Duplicate Clustering:** Use advanced clustering algorithms (like DBSCAN) to visualize and categorize semantic duplicates.
- **CI/CD:** Automate pipeline execution via GitHub Actions.
- **Unit Tests:** Add a `tests/` directory with `pytest` coverage for core validation logic.
- **Docker Support:** Containerize the application for seamless cross-platform execution.
- **Web Dashboard:** Build a Streamlit or Gradio dashboard for real-time evaluation monitoring.

---

## Technologies Used

- **Python**: Core programming language.
- **Google Gemini API**: Underlying foundational models (gemini-2.5-flash).
- **google-genai**: Official SDK for interacting with Gemini.
- **Sentence Transformers**: `all-MiniLM-L6-v2` for generating text embeddings.
- **Scikit-learn**: Cosine similarity calculations for semantic deduplication.
- **NumPy**: Matrix operations for embedding arrays.
- **Regex**: Fast, heuristic-based safety policy enforcement.
- **tqdm**: CLI progress bars for long-running generation and evaluation loops.

---

## Design Decisions

- **Sentence Transformers:** Chosen for semantic deduplication because they offer a lightweight, local, and cost-free way to calculate near-duplicates without relying on paid API embedding models.
- **Regex Safety Filter:** Used as the *first* layer of defense because it is incredibly fast and cheap, instantly catching blatant violations before invoking slower semantic checks.
- **In-Memory Generation Validation:** `generator.py` validates structure and safety *before* saving to disk, ensuring the raw `generated_chats.jsonl` file is inherently clean, rather than saving garbage data to be cleaned later.
- **LLM-as-a-Judge:** Chosen because evaluating complex, subjective metrics like "Warmth" and "Compassion" is impossible with traditional deterministic code.

---

## Submission Checklist

- [x] Dataset validation
- [x] Safety checking
- [x] Duplicate detection
- [x] Synthetic data generation
- [x] Response evaluation
- [x] Documentation
