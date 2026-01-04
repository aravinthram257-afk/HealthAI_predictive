import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
from data_loader import load_reviews
def run_sentiment_ml_training():
    print("Starting TF-IDF + Logistic Regression sentiment training...")
    df = load_reviews()
    df = df[df["rating"] != "Ratings"]
    df["review_text"] = df["review_text"].astype(str)
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_num"] = df["rating_num"].fillna(df["rating_num"].median())
    df["label"] = (df["rating_num"] >= 4).astype(int)
    texts = df["review_text"].tolist()
    labels = df["label"].tolist()
    print("Label counts:", labels.count(0), "neg,", labels.count(1), "pos")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    # Logistic Regression classifier
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    print(f"ML Sentiment - Accuracy : {acc:.3f}")
    print(f"ML Sentiment - Precision: {prec:.3f}")
    print(f"ML Sentiment - Recall   : {rec:.3f}")
    print(f"ML Sentiment - F1-score : {f1:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    # Save artifacts
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    vec_path = models_dir / "sentiment_tfidf.joblib"
    clf_path = models_dir / "sentiment_logreg.joblib"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(clf, clf_path)
    print(f"Saved TF-IDF vectorizer to {vec_path}")
    print(f"Saved LogisticRegression model to {clf_path}")
    return vectorizer, clf
if __name__ == "__main__":
    run_sentiment_ml_training()
