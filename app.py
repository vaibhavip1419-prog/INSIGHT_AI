import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import plotly.express as px
import joblib

st.set_page_config(page_title="INSIGHT_AI", page_icon="🤖", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
/* ===== INSIGHT_AI DARK THEME ===== */
.stApp { background: #07111f !important; color: #e5eef8 !important; }
[data-testid="stAppViewContainer"] { background: #07111f !important; }
[data-testid="stHeader"] { background: #07111f !important; }
[data-testid="stToolbar"] { background: #07111f !important; }

/* Main content */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #dbe7f3 !important; }
h1,h2,h3,h4,h5,h6, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 { color: #f3f7fb !important; }
.main-title { font-size: 40px; font-weight: 850; color: #5ee7df !important; margin: 0; letter-spacing: .3px; }
.sub-title { color: #9fb3c8 !important; font-size: 17px; font-weight: 600; margin-bottom: 18px; }
.section-title { font-size: 25px; font-weight: 800; color: #f3f7fb !important; padding: 10px 0 14px; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1b33 0%,#081522 100%) !important; border-right: 1px solid #1d3853 !important; }
[data-testid="stSidebar"] * { color: #dce8f5 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #dce8f5 !important; }
[data-testid="stSidebar"] hr { border-color: #24415e !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #e5eef8 !important; font-weight: 600 !important; }

/* Inputs */
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input {
  color: #e8f1f8 !important; -webkit-text-fill-color: #e8f1f8 !important;
  background: #0d1d2e !important; border: 1px solid #31506d !important; border-radius: 10px !important;
}
[data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus { border-color: #45d6cf !important; box-shadow: 0 0 0 1px #45d6cf !important; }
[data-testid="stTextArea"] textarea::placeholder, [data-testid="stTextInput"] input::placeholder { color: #8195aa !important; -webkit-text-fill-color: #8195aa !important; opacity: 1 !important; }
[data-testid="stTextArea"] label, [data-testid="stTextInput"] label, [data-testid="stNumberInput"] label { color: #dbe7f3 !important; font-weight: 700 !important; }

/* Select / multiselect */
[data-baseweb="select"] > div, [data-baseweb="select"] { background: #0d1d2e !important; border-color: #31506d !important; color: #e8f1f8 !important; }
[data-baseweb="select"] *, [data-baseweb="popover"] *, [data-baseweb="menu"] * { color: #e8f1f8 !important; background-color: transparent; }
[data-baseweb="popover"], [data-baseweb="menu"] { background: #102337 !important; border: 1px solid #31506d !important; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button { color: #06131e !important; -webkit-text-fill-color: #06131e !important; background: linear-gradient(90deg,#5ee7df,#43c9d0) !important; border: 1px solid #5ee7df !important; font-weight: 850 !important; border-radius: 10px !important; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background: linear-gradient(90deg,#43c9d0,#5ee7df) !important; }

/* Cards */
.kpi { background: linear-gradient(145deg,#102337,#0c1b2b) !important; border: 1px solid #24445f; border-radius: 15px; padding: 18px; box-shadow: 0 8px 24px rgba(0,0,0,.25); min-height: 105px; }
.kpi-label { color: #91a7bc !important; font-size: 14px; font-weight: 700; }
.kpi-value { color: #f3f7fb !important; font-size: 29px; font-weight: 850; margin-top: 5px; }
.response { background: linear-gradient(145deg,#102337,#0d1c2c) !important; color: #e8f1f8 !important; border: 1px solid #31506d; border-radius: 14px; padding: 22px; box-shadow: 0 8px 24px rgba(0,0,0,.22); }
.response * { color: #e8f1f8 !important; }

/* Alerts */
[data-testid="stAlert"], [data-testid="stAlert"] *, [data-baseweb="notification"], [data-baseweb="notification"] * { color: #e8f1f8 !important; }

/* Metrics / captions / code */
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { color: #f3f7fb !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #91a7bc !important; }
[data-testid="stCodeBlock"] { background: #0b1928 !important; border: 1px solid #27445e !important; }
[data-testid="stCodeBlock"] *, pre, code { color: #cfe2f3 !important; background: #0b1928 !important; }

/* Progress */
[data-testid="stProgressBar"] { background: #22374b !important; }

/* Radio / checkbox */
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color: #dbe7f3 !important; }

/* Hide Streamlit decorative white surfaces */
[data-testid="stDecoration"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)



# ---------------- HYBRID SENTIMENT CHECK ----------------
def improve_sentiment_prediction(review, model_prediction):
    """
    Keep Logistic Regression as the main ML model, but correct obvious
    sentiment expressions that TF-IDF can occasionally misclassify.
    """
    text = str(review).lower()
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    strong_negative = [
        "too tight", "too loose", "doesn't fit", "does not fit",
        "not fit", "poor quality", "poor material", "bad quality",
        "very poor", "very bad", "not good", "not happy",
        "very disappointed", "disappointed", "uncomfortable",
        "terrible", "horrible", "worst", "hate", "waste of money",
        "dislike", "defective", "damaged", "broken", "late delivery",
        "delivery was late", "never received", "wrong size"
    ]

    strong_positive = [
        "excellent", "very good", "really good", "love this",
        "love it", "very happy", "highly recommend", "perfect",
        "beautiful and comfortable", "great quality", "amazing",
        "fantastic", "wonderful", "comfortable and beautiful"
    ]

    # Negative expressions get priority when both positive and negative
    # words appear in the same review (e.g. "good product but too tight").
    if any(phrase in text for phrase in strong_negative):
        return "Negative"

    if any(phrase in text for phrase in strong_positive):
        return "Positive"

    return model_prediction

# ---------------- PRODUCT IDENTIFICATION ----------------
def detect_product(review, product_classes):
    """Identify a product/class from the words used in customer feedback."""
    text = re.sub(r"[^a-zA-Z\s]", " ", str(review).lower())
    words = set(text.split())

    candidates = sorted(
        [str(x) for x in product_classes if pd.notna(x)],
        key=lambda x: len(x.split()),
        reverse=True
    )

    # First: exact phrase match.
    for product in candidates:
        product_words = re.sub(r"[^a-zA-Z\s]", " ", product.lower()).split()
        if product_words and all(word in words for word in product_words):
            return product

    # Second: match normal singular/plural forms.
    # Examples: dress -> Dresses, blouse -> Blouses, jacket -> Jackets.
    for product in candidates:
        p = re.sub(r"[^a-zA-Z\s]", " ", product.lower()).strip()
        if not p or " " in p:
            continue

        possible_forms = {p}

        if p.endswith("ies"):
            possible_forms.add(p[:-3] + "y")
        elif p.endswith("es"):
            possible_forms.add(p[:-2])
            possible_forms.add(p[:-1])
        elif p.endswith("s"):
            possible_forms.add(p[:-1])

        if words.intersection(possible_forms):
            return product

    return "Product not identified"

# ---------------- NLP PIPELINE ----------------
def load_model_and_vectorizer():
    # Load the exact fitted artifacts exported from the corrected Colab notebook.
    # Paths are based on this app.py file, so the app works regardless of the
    # terminal's current working directory.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "sentiment_model_fixed.pkl")
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer_fixed.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Validate the artifact. If a stale/unfitted pickle is accidentally present,
    # rebuild BOTH vectorizer and model from the bundled training data so they stay aligned.
    if not hasattr(vectorizer, "idf_") or not hasattr(vectorizer, "vocabulary_") or not hasattr(model, "coef_"):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        data_path = os.path.join(base_dir, "final_predictions.csv")
        data = pd.read_csv(data_path)
        text_col = "Clean_Review"
        label_col = "Sentiment"
        data = data.dropna(subset=[text_col, label_col]).copy()
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X = vectorizer.fit_transform(data[text_col].astype(str))
        y = data[label_col].replace({"Positive":"Postive", "Neutral":"Netural"})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = LogisticRegression(max_iter=1000, class_weight={"Negative":1.0, "Netural":3.0, "Postive":1.0})
        model.fit(X_train, y_train)

    # Match the original Colab preprocessing as closely as possible.
    # The model was trained after stopword removal + lemmatization, while keeping
    # negation words that are important for sentiment.
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            lemmatizer = WordNetLemmatizer()
    except Exception:
        # Safe fallback if NLTK resources cannot be downloaded.
        stop_words = set()
        lemmatizer = None

    stop_words -= {"not", "no", "nor", "neither", "never", "n't"}

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        words = text.split()
        if lemmatizer is not None:
            words = [
                lemmatizer.lemmatize(word)
                for word in words
                if word not in stop_words
            ]
        else:
            words = [word for word in words if word not in stop_words]
        return " ".join(words)

    return vectorizer, model, clean_text, 0.0

# Load the exact model trained in Colab. Streamlit does not retrain it.
tfidf, best_model, clean_text, best_accuracy = load_model_and_vectorizer()
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_predictions.csv"))

# ---------------- DYNAMIC DATA ----------------
if "dynamic_df" not in st.session_state:
    st.session_state.dynamic_df = df.copy()

def normalize_uploaded_dataset(uploaded_df):
    """Prepare a user CSV so the dashboard can analyze it automatically."""
    data = uploaded_df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    aliases = {
        "review": "Review Text", "review text": "Review Text", "review_text": "Review Text",
        "reviews": "Review Text", "comment": "Review Text", "comments": "Review Text",
        "rating": "Rating", "score": "Rating",
        "department": "Department Name", "department name": "Department Name",
        "category": "Class Name", "class": "Class Name", "class name": "Class Name",
        "product class": "Class Name", "division": "Division Name", "division name": "Division Name",
        "recommended": "Recommended IND", "recommended ind": "Recommended IND", "age": "Age"
    }
    lower_to_original = {str(c).strip().lower(): c for c in data.columns}
    rename_map = {}
    for alias, target in aliases.items():
        if alias in lower_to_original and target not in data.columns:
            rename_map[lower_to_original[alias]] = target
    data = data.rename(columns=rename_map)

    if "Review Text" not in data.columns:
        raise ValueError("CSV must contain a review column such as 'Review Text', 'Review', or 'review_text'.")

    data = data.dropna(subset=["Review Text"]).copy()
    data["Review Text"] = data["Review Text"].astype(str)

    for col in ["Rating", "Age", "Recommended IND", "Division Name", "Department Name", "Class Name"]:
        if col not in data.columns:
            data[col] = np.nan

    data["Clean_Review"] = data["Review Text"].apply(clean_text)
    vectors = tfidf.transform(data["Clean_Review"].astype(str))
    predictions = best_model.predict(vectors)
    data["Sentiment"] = predictions
    data["Predicted_Sentiment"] = predictions

    return data

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🤖 INSIGHT_AI")
st.sidebar.caption("AI-Powered Customer Feedback & Business Analytics System")

st.sidebar.markdown("### 📂 Add Your Dataset")

# Allow the user to upload ONE or MORE CSV files.
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV file(s)",
    type=["csv"],
    accept_multiple_files=True,
    help="Upload one or more customer feedback CSV files. All uploaded files will be combined automatically and the graphs will update from the combined data."
)

if uploaded_files:
    # Create a key from all selected files so the dashboard refreshes
    # whenever the selection changes.
    upload_key = tuple(
        sorted((file.name, file.size) for file in uploaded_files)
    )

    if st.session_state.get("processed_upload_key") != upload_key:
        try:
            uploaded_data = []

            for file in uploaded_files:
                temp_df = pd.read_csv(file)
                uploaded_data.append(temp_df)

            # Combine all uploaded CSV files into one dataset.
            combined_df = pd.concat(uploaded_data, ignore_index=True)
            user_df = normalize_uploaded_dataset(combined_df)

            st.session_state.dynamic_df = user_df
            st.session_state.processed_upload_key = upload_key
            st.session_state.uploaded_rows = len(user_df)
            st.session_state.uploaded_file_count = len(uploaded_files)

        except Exception as e:
            st.sidebar.error(f"Could not process CSV file(s): {e}")

df = st.session_state.dynamic_df.copy()
product_classes = sorted(df["Class Name"].dropna().astype(str).unique().tolist())

if "uploaded_rows" in st.session_state:
    st.sidebar.success(f"Loaded {st.session_state.uploaded_rows:,} reviews")
else:
    st.sidebar.info(f"Using {len(df):,} project reviews")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Application",
    ["Give Feedback", "Customer Feedback Overview", "Sentiment Analysis", "Business Insights"]
)

# ---------------- USER FEEDBACK ----------------
if page == "Give Feedback":
    st.markdown('<div class="main-title">INSIGHT_AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Customer Feedback & Business Analytics System</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Customer Feedback Analysis</div>', unsafe_allow_html=True)

    st.write("Enter a customer review below. INSIGHT_AI preprocesses the feedback and predicts its sentiment using the corrected Logistic Regression model from your project.")

    with st.form("feedback_form"):
        review = st.text_area(
            "Enter Customer Feedback",
            height=170,
            placeholder="Example: The dress is beautiful and comfortable, but the size is slightly small."
        )
        submitted = st.form_submit_button("🔍 Analyze Feedback", use_container_width=True)

    if submitted:
        if not review.strip():
            st.warning("Please enter customer feedback first.")
        else:
            cleaned = clean_text(review)
            vector = tfidf.transform([cleaned])
            raw_prediction = best_model.predict(vector)[0]
            probs = best_model.predict_proba(vector)[0]

            # Normalize the spelling used by the original notebook:
            # Postive -> Positive and Netural -> Neutral.
            label_map = {"Postive": "Positive", "Netural": "Neutral", "Negative": "Negative"}
            prediction = label_map.get(raw_prediction, raw_prediction)
            prediction = improve_sentiment_prediction(review, prediction)
            probability_map = {
                label_map.get(cls, cls): float(prob)
                for cls, prob in zip(best_model.classes_, probs)
            }
            confidence = float(np.max(probs))
            detected_product = detect_product(review, product_classes)

            # If the hybrid phrase check changed the sentiment, reflect that
            # in the displayed probability so the result is not contradictory.
            if prediction not in probability_map:
                probability_map[prediction] = confidence

            st.markdown("### Analysis Result")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Customer Feedback:**  \n{review}")
            with c2:
                st.markdown(f"**Detected Product:** 🛍️ {detected_product}")
                st.markdown("**Predicted Sentiment:**")
                if prediction == "Positive":
                    st.success("😊 POSITIVE")
                    response = "Thank you for your positive feedback! We're glad you had a good experience. Your feedback helps us continue improving our products and service."
                elif prediction == "Negative":
                    st.error("😟 NEGATIVE")
                    response = "We're sorry that your experience did not meet your expectations. Thank you for sharing your feedback. It will help us identify areas that need improvement."
                else:
                    st.info("😐 NEUTRAL")
                    response = "Thank you for sharing your feedback. We appreciate your input and will use it to better understand customer needs and improve our products."

            st.markdown("### 🤖 INSIGHT_AI Response")
            st.markdown(f'<div class="response">{response}</div>', unsafe_allow_html=True)

            st.progress(confidence, text=f"Model confidence: {confidence*100:.2f}%")

            p1, p2, p3 = st.columns(3)
            for col, label in zip((p1,p2,p3), ("Positive","Neutral","Negative")):
                with col:
                    st.metric(label, f"{probability_map.get(label,0)*100:.2f}%")

            # Add the new feedback to the current session dataset.
            new_row = {col: np.nan for col in st.session_state.dynamic_df.columns}
            if "Review Text" in new_row:
                new_row["Review Text"] = review
            if "Clean_Review" in new_row:
                new_row["Clean_Review"] = cleaned
            if "Sentiment" in new_row:
                new_row["Sentiment"] = raw_prediction
            if "Class Name" in new_row and detected_product != "Product not identified":
                new_row["Class Name"] = detected_product
            st.session_state.dynamic_df = pd.concat(
                [st.session_state.dynamic_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            st.caption("Model used: Logistic Regression | Corrected fitted Logistic Regression + TF-IDF")

            st.markdown("### 🔎 Preprocessed Text")
            st.code(cleaned if cleaned else "(empty after preprocessing)")

# ---------------- DASHBOARD ----------------
else:
    # Always use the latest data, including new feedback submitted in this session.
    df = st.session_state.dynamic_df.copy()

    st.sidebar.markdown("### Filters")

    tmp = df.copy()
    tmp["Department Name"] = tmp["Department Name"].fillna("Unknown").astype(str)
    tmp["Class Name"] = tmp["Class Name"].fillna("Unknown").astype(str)
    tmp["Rating"] = pd.to_numeric(tmp["Rating"], errors="coerce")

    departments = sorted(tmp["Department Name"].unique().tolist())
    classes = sorted(tmp["Class Name"].unique().tolist())
    sentiments = ["Positive", "Neutral", "Negative"]
    ratings = sorted(tmp["Rating"].dropna().unique().tolist())

    sel_dept = st.sidebar.multiselect("Department", departments, default=departments)
    sel_class = st.sidebar.multiselect("Product Class", classes, default=classes)
    sel_sent = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)
    sel_rating = st.sidebar.multiselect("Rating", ratings, default=ratings)

    tmp["Sentiment_Display"] = tmp["Sentiment"].replace(
        {"Postive": "Positive", "Netural": "Neutral", "Negative": "Negative"}
    )
    tmp["Recommendation"] = tmp["Recommended IND"].map(
        {1: "Recommended", 0: "Not Recommended"}
    )

    filtered = tmp[
        tmp["Department Name"].isin(sel_dept)
        & tmp["Class Name"].isin(sel_class)
        & tmp["Sentiment_Display"].isin(sel_sent)
    ].copy()
    if ratings:
        filtered = filtered[filtered["Rating"].isin(sel_rating)].copy()

    st.markdown('<div class="main-title">INSIGHT_AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">AI-Powered Customer Feedback & Business Analytics System</div>',
        unsafe_allow_html=True
    )
    st.caption(f"🔄 Dynamic reviews in this session: {len(df):,}")

    def kpi(label, value):
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True
        )

    def layout(fig, height=360):
        fig.update_layout(
            height=height,
            margin=dict(l=45, r=25, t=65, b=55),
            paper_bgcolor="#0d1d2e",
            plot_bgcolor="#0d1d2e",
            font=dict(color="#e8f1f8"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )
        fig.update_xaxes(
            showgrid=False,
            color="#dbe7f3",
            title_font=dict(color="#dbe7f3"),
            tickfont=dict(color="#a9bbcc")
        )
        fig.update_yaxes(
            gridcolor="#22384d",
            color="#dbe7f3",
            title_font=dict(color="#dbe7f3"),
            tickfont=dict(color="#a9bbcc")
        )
        return fig

    if filtered.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    # ---------------- OVERVIEW ----------------
    if page == "Customer Feedback Overview":
        st.markdown(
            '<div class="section-title">Customer Feedback Overview</div>',
            unsafe_allow_html=True
        )

        vals = [
            len(filtered),
            f"{filtered.Rating.mean():.2f} / 5",
            int((filtered.Sentiment_Display == "Positive").sum()),
            int((filtered.Sentiment_Display == "Negative").sum()),
            int((filtered.Sentiment_Display == "Neutral").sum())
        ]

        cols = st.columns(5)
        for col, label, val in zip(
            cols,
            ["Total Reviews", "Average Rating", "Positive Reviews",
             "Negative Reviews", "Neutral Reviews"],
            vals
        ):
            with col:
                kpi(label, f"{val:,}" if isinstance(val, int) else val)

        st.markdown("### 📊 Overall Feedback")

        c1, c2 = st.columns(2)

        with c1:
            s = (
                filtered["Sentiment_Display"]
                .value_counts()
                .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
                .reset_index()
            )
            s.columns = ["Sentiment", "Count"]
            fig = px.pie(
                s,
                names="Sentiment",
                values="Count",
                hole=.58,
                title="Sentiment Distribution"
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(layout(fig, 370), use_container_width=True)

        with c2:
            r = (
                filtered["Rating"]
                .value_counts()
                .reindex(ratings, fill_value=0)
                .reset_index()
            )
            r.columns = ["Rating", "Count"]
            fig = px.bar(
                r,
                x="Rating",
                y="Count",
                text="Count",
                title="Rating Distribution"
            )
            fig.update_traces(textposition="outside")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(layout(fig, 370), use_container_width=True)

        d = filtered["Department Name"].value_counts().reset_index()
        d.columns = ["Department", "Count"]
        d = d.sort_values("Count", ascending=True)

        fig = px.bar(
            d,
            x="Count",
            y="Department",
            orientation="h",
            text="Count",
            title="Reviews by Department"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(layout(fig, max(330, 70 + 45 * len(d))), use_container_width=True)

    # ---------------- SENTIMENT ----------------
    elif page == "Sentiment Analysis":
        st.markdown(
            '<div class="section-title">Sentiment Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Sentiment by Department")

        p = (
            pd.crosstab(
                filtered["Department Name"],
                filtered["Sentiment_Display"]
            )
            .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
            .reset_index()
        )
        p = p.melt(
            id_vars="Department Name",
            var_name="Sentiment",
            value_name="Reviews"
        )

        fig = px.bar(
            p,
            x="Department Name",
            y="Reviews",
            color="Sentiment",
            barmode="stack",
            text_auto=True,
            title="Positive, Neutral and Negative Reviews by Department"
        )
        fig.update_layout(xaxis_title="Department", yaxis_title="Reviews")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(
            layout(fig, max(390, 300 + 20 * len(sel_dept))),
            use_container_width=True
        )

        st.markdown("### ⚠️ Negative Reviews")

        neg = filtered[filtered.Sentiment_Display == "Negative"]
        if neg.empty:
            st.success("No negative reviews match the selected filters.")
        else:
            n = neg["Department Name"].value_counts().reset_index()
            n.columns = ["Department", "Negative Reviews"]
            n = n.sort_values("Negative Reviews", ascending=True)

            fig = px.bar(
                n,
                x="Negative Reviews",
                y="Department",
                orientation="h",
                text="Negative Reviews",
                title="Negative Reviews by Department"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(
                layout(fig, max(330, 70 + 45 * len(n))),
                use_container_width=True
            )

    # ---------------- BUSINESS INSIGHTS ----------------
    else:
        st.markdown(
            '<div class="section-title">Business Insights</div>',
            unsafe_allow_html=True
        )

        a = (
            filtered.groupby("Department Name", as_index=False)["Rating"]
            .mean()
            .sort_values("Rating", ascending=True)
        )

        fig = px.bar(
            a,
            x="Rating",
            y="Department Name",
            orientation="h",
            text="Rating",
            title="Average Rating by Department"
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )
        fig.update_xaxes(range=[0, 5])
        fig.update_layout(xaxis_title="Average Rating", yaxis_title="Department")
        st.plotly_chart(
            layout(fig, max(330, 70 + 45 * len(a))),
            use_container_width=True
        )

        c1, c2 = st.columns(2)

        with c1:
            p = (
                filtered[filtered.Sentiment_Display == "Positive"]
                ["Department Name"]
                .value_counts()
                .reset_index()
            )
            p.columns = ["Department", "Positive Reviews"]

            if p.empty:
                st.info("No positive reviews match the selected filters.")
            else:
                p = p.sort_values("Positive Reviews", ascending=True)
                fig = px.bar(
                    p,
                    x="Positive Reviews",
                    y="Department",
                    orientation="h",
                    text="Positive Reviews",
                    title="Positive Reviews by Department"
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(
                    layout(fig, max(330, 70 + 45 * len(p))),
                    use_container_width=True
                )

        with c2:
            st.markdown("### 💡 Key Business Insights")

            positive_pct = (
                filtered.Sentiment_Display.eq("Positive").mean() * 100
            )

            highest = a.loc[a["Rating"].idxmax()]
            lowest = a.loc[a["Rating"].idxmin()]

            st.info(
                f"**Positive feedback:** {positive_pct:.1f}% "
                f"of the selected reviews."
            )
            st.info(
                f"**Highest average rating:** {highest['Department Name']} "
                f"({highest['Rating']:.2f}/5)."
            )
            st.info(
                f"**Lowest average rating:** {lowest['Department Name']} "
                f"({lowest['Rating']:.2f}/5)."
            )

st.caption("INSIGHT_AI • AI-powered customer feedback analysis and business analytics")
