# Fake News Detection — Lightweight Pipeline

**One-line:** End-to-end fake-news classifier (preprocessing → Word2Vec → RandomForest).
**Author:** Kashyap Milind Trivedi

## What it is

A compact, reproducible pipeline for binary fake-news classification. Fast preprocessing (HTML/URL/emoji removal, stopwords, stemming), Word2Vec document vectors, and a RandomForest baseline — ready for experimentation or quick demos.

## Quick start

```bash
# create venv & install
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# run
python /mnt/data/fake_news_detetction.py
```

## Requirements (suggested)

```
numpy pandas scikit-learn gensim nltk beautifulsoup4
```

*(Download NLTK stopwords on first run: `nltk.download('stopwords')`.)*

## Dataset

Download from Kaggle and place CSV in `data/` (or update path in script):
[https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

## Files

* `/mnt/data/fake_news_detetction.py` — main preprocessing + training script.

## Output

Console logs + evaluation (accuracy, precision/recall/F1). Models are **not** persisted by default (add `joblib.dump` to save them).

## Next steps (optional)

* Persist models & add FastAPI inference.
* Swap averaged Word2Vec for Sentence-BERT for better accuracy.
* Add CV / hyperparameter tuning and explainability (SHAP/LIME).

## License & Contact

MIT License. Open issues/PRs or contact me (Kashyap Milind Trivedi) for collaboration.

---

