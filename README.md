# 📰 Hard Truth: News Comparison with LLMs

##  Setup Instructions

###  1. Install Python 3.10 (Required)

> This project is **not compatible with Python 3.13**.

Download Python 3.10 from:  
[https://www.python.org/downloads/release/python-31013/](https://www.python.org/downloads/release/python-31013/)

Make sure to:
-  Check **"Add Python to PATH"** during installation

---

###  2. Clone and Navigate to the Project

```bash
git clone <your-repo-url>
cd hard_truth
```

### 3. Run the Setup Script
py -3.10 setup_env.py

### 4. Activate the Virtual Environment
Windows: .\venv\Scripts\activate

macOS/Linux: source venv/bin/activate

### 5. Run the Project
python main.py

### Output Files
| File                            | Description                          |
| ------------------------------- | ------------------------------------ |
| `outputs/articles.csv`          | Raw fetched articles                 |
| `outputs/embedded_articles.csv` | Articles with embeddings             |
| `outputs/ground_truth.txt`      | Commonly extracted entities          |
| `outputs/articles_scored.csv`   | Articles scored against ground truth |
