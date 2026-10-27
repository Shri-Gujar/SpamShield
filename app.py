import os
import pickle
import html
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SpamShield | SMS Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "spam_model_improved.pkl"
DATASET_PATH = "SMSSpamCollection"

# --------------------------------------------------
# LAVENDER THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F8F7FF;
        color: #292344;
    }

    [data-testid="stSidebar"] {
        background: #EEE9FF;
        border-right: 1px solid #DDD4FF;
    }

    [data-testid="stSidebar"] * {
        color: #302451;
    }

    .hero {
        background: linear-gradient(120deg, #6D4AFF, #A78BFA);
        color: white;
        padding: 30px;
        border-radius: 22px;
        margin-bottom: 24px;
        box-shadow: 0 8px 25px rgba(109, 74, 255, 0.16);
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: #F4F0FF;
        margin-top: 10px;
        margin-bottom: 0;
        font-size: 15px;
    }

    .metric-card {
        background: white;
        border: 1px solid #E6E0FA;
        border-radius: 17px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 4px 15px rgba(60, 40, 100, 0.04);
    }

    .metric-label {
        color: #756B91;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #5036B9;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #8A80A2;
        font-size: 12px;
        margin-top: 4px;
    }

    .result-card {
        padding: 24px;
        border-radius: 18px;
        margin: 15px 0;
        border: 1px solid #E6E0FA;
        background: white;
    }

    .safe-card {
        border-left: 6px solid #16A34A;
    }

    .spam-card {
        border-left: 6px solid #DC2626;
    }

    .section-title {
        color: #39266F;
        font-weight: 800;
        font-size: 22px;
        margin-bottom: 8px;
    }

    .muted {
        color: #756B91;
        font-size: 13px;
    }

    div.stButton > button {
        background: #7048F5;
        color: white;
        border: none;
        border-radius: 11px;
        padding: 10px 20px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #5935D8;
        color: white;
        border: none;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #292344 !important;
        background: #FFFFFF !important;
        -webkit-text-fill-color: #292344 !important;
    }

    div[data-baseweb="select"] * {
        color: #292344 !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }

    hr {
        border-color: #E6E0FA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


model = load_model()

if model is None:
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Keep spam_model_improved.pkl in the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def predict_message(message):
    """Return prediction and estimated spam probability."""

    prediction = str(model.predict([message])[0]).lower()

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([message])[0]
        classes = [str(label).lower() for label in model.classes_]

        if "spam" in classes:
            spam_index = classes.index("spam")
            probability = float(probabilities[spam_index]) * 100

    return prediction, probability


def explain_message(message):
    """
    Show influential words when the saved model is a compatible
    TF-IDF + Logistic Regression pipeline.
    """

    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        if not hasattr(classifier, "coef_"):
            return []

        vector = vectorizer.transform([message])
        feature_names = vectorizer.get_feature_names_out()

        coefficients = classifier.coef_[0]
        classes = [str(label).lower() for label in classifier.classes_]

        # Logistic Regression coefficients normally point toward
        # classes_[1]. Adjust direction if spam is classes_[0].
        if classes.index("spam") == 0:
            coefficients = -coefficients

        contributions = vector.multiply(coefficients).toarray()[0]
        present_indices = vector.nonzero()[1]

        ranked = sorted(
            [
                (feature_names[index], float(contributions[index]))
                for index in present_indices
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        return ranked[:8]

    except (KeyError, AttributeError, ValueError, IndexError):
        return []


@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None

    try:
        data = pd.read_csv(
            DATASET_PATH,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8",
            on_bad_lines="skip",
        )

        data = data.dropna(subset=["label", "text"])
        data["label"] = data["label"].astype(str).str.strip().str.lower()
        data["text"] = data["text"].astype(str)

        data = data[data["label"].isin(["ham", "spam"])]

        if data.empty:
            return None

        return data

    except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
        return None


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-note">{html.escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def record_prediction(message, prediction, probability):
    st.session_state.history.insert(
        0,
        {
            "Time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "Message": message,
            "Prediction": prediction.upper(),
            "Spam probability (%)": (
                round(probability, 2)
                if probability is not None
                else None
            ),
        },
    )

    # Keep the latest 100 predictions.
    st.session_state.history = st.session_state.history[:100]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🛡️ SpamShield")
    st.caption("SMS Security Intelligence")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔎 Smart Detection",
            "📂 Batch Analysis",
            "📊 Model Insights",
        ],
    )

    st.divider()

    st.markdown("### System Status")
    st.success("Model loaded successfully")

    st.markdown(
        f"**Model file:** `{MODEL_PATH}`"
    )

    st.caption("Machine learning powered SMS classification")

    st.divider()

    st.caption("SpamShield • Portfolio Project")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SpamShield</h1>
        <p>
            Intelligent SMS spam detection, message analysis,
            batch classification and model performance insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# PAGE 1: DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Security Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Monitor your recent predictions and explore the capabilities "
        "of your SMS spam detection model."
    )

    history_df = pd.DataFrame(st.session_state.history)

    total_predictions = len(history_df)

    spam_count = (
        int((history_df["Prediction"] == "SPAM").sum())
        if total_predictions
        else 0
    )

    ham_count = (
        int((history_df["Prediction"] == "HAM").sum())
        if total_predictions
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Messages Analyzed", total_predictions, "Current session")

    with col2:
        metric_card("Spam Detected", spam_count, "Flagged messages")

    with col3:
        metric_card("Legitimate Messages", ham_count, "Classified as ham")

    with col4:
        spam_rate = (
            f"{spam_count / total_predictions * 100:.1f}%"
            if total_predictions
            else "—"
        )
        metric_card("Spam Share", spam_rate, "Of session predictions")

    st.markdown("---")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Recent Activity")

        if not history_df.empty:
            st.dataframe(
                history_df.head(8),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No predictions yet. Open Smart Detection and analyze "
                "your first message."
            )

    with right:
        st.markdown("### Prediction Distribution")

        if total_predictions:
            distribution = pd.DataFrame(
                {
                    "Category": ["Spam", "Legitimate"],
                    "Count": [spam_count, ham_count],
                }
            )

            fig = px.pie(
                distribution,
                names="Category",
                values="Count",
                hole=0.62,
                color="Category",
                color_discrete_map={
                    "Spam": "#E87979",
                    "Legitimate": "#8B79F7",
                },
            )

            fig.update_layout(
                margin=dict(t=15, b=15, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Your chart will appear after the first prediction.")

    st.markdown("---")

    st.markdown("### What SpamShield Can Do")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 🔎 Smart Detection")
        st.write(
            "Classify an individual SMS and view its estimated spam probability."
        )

    with c2:
        st.markdown("#### 📂 Batch Analysis")
        st.write(
            "Upload a CSV file, classify multiple messages and export results."
        )

    with c3:
        st.markdown("#### 📊 Model Insights")
        st.write(
            "Inspect model performance, confusion matrix and evaluation metrics."
        )

    st.caption(
        "Note: session statistics describe only messages analyzed in this "
        "session. They are not live network or telecom security statistics."
    )

# ==================================================
# PAGE 2: SMART DETECTION
# ==================================================

elif page == "🔎 Smart Detection":

    st.markdown(
        '<div class="section-title">Smart Message Detection</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter an SMS message to classify it as spam or legitimate."
    )

    message = st.text_area(
        "Message content",
        height=170,
        placeholder=(
            "Example: Congratulations! You have won a prize. "
            "Click the link to claim it."
        ),
        key="single_message",
    )

    analyze = st.button(
        "🔍 Analyze Message",
        use_container_width=True,
    )

    if analyze:
        if not message.strip():
            st.warning("Please enter a message first.")

        else:
            prediction, probability = predict_message(message.strip())

            st.session_state.latest_result = {
                "message": message.strip(),
                "prediction": prediction,
                "probability": probability,
            }

            record_prediction(
                message.strip(),
                prediction,
                probability,
            )

    result = st.session_state.latest_result

    if result is not None:
        prediction = result["prediction"]
        probability = result["probability"]

        if prediction == "spam":
            st.markdown(
                """
                <div class="result-card spam-card">
                    <h2>🚨 Spam Detected</h2>
                    <p>
                        The model classified this message as spam.
                        Be careful with suspicious links, requests for money,
                        passwords or personal information.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card safe-card">
                    <h2>✅ Classified as Legitimate (Ham)</h2>
                    <p>
                        The model classified this message as legitimate.
                        This does not guarantee that the message is safe.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)

        with col1:
            metric_card(
                "Predicted Category",
                prediction.upper(),
                "Machine learning output",
            )

        with col2:
            metric_card(
                "Estimated Spam Probability",
                (
                    f"{probability:.2f}%"
                    if probability is not None
                    else "Unavailable"
                ),
                "Model estimate, not a guarantee",
            )

        if probability is not None:
            st.markdown("### Spam Probability")
            st.progress(min(max(probability / 100, 0.0), 1.0))
            st.caption(
                "A high estimated probability indicates stronger model "
                "support for the spam class. Predictions can still be wrong."
            )

        st.markdown("---")
        st.markdown("### Message Analysis")

        st.write("**Analyzed message:**")
        st.code(result["message"], language=None)

        influential_terms = explain_message(result["message"])

        if influential_terms:
            st.markdown("#### Influential Words")

            st.write(
                "These words have the strongest individual contributions "
                "among the words present in this message. They are model "
                "signals, not proof that a message is malicious."
            )

            term_df = pd.DataFrame(
                influential_terms,
                columns=["Word", "Spam-direction contribution"],
            )

            term_df["Direction"] = term_df[
                "Spam-direction contribution"
            ].apply(
                lambda value: (
                    "Spam signal" if value > 0 else "Ham signal"
                )
            )

            st.dataframe(
                term_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Word-level explanation is unavailable for this model format."
            )

        st.caption(
            "Privacy note: this demo processes text locally in the running "
            "app. If deployed publicly, review hosting, logging and data "
            "handling before entering private messages."
        )

# ==================================================
# PAGE 3: BATCH ANALYSIS
# ==================================================

elif page == "📂 Batch Analysis":

    st.markdown(
        '<div class="section-title">Batch SMS Analyzer</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a CSV containing SMS messages to classify multiple rows "
        "in one operation."
    )

    st.info(
        "Your CSV should contain a column with message text, such as "
        "'text', 'message' or 'sms'. The app will let you choose the column."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Use a CSV with a header row and a column containing SMS text.",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.warning("The uploaded CSV is empty.")

            elif len(uploaded_df.columns) == 0:
                st.warning("No columns were found in the CSV.")

            else:
                st.markdown("### Preview")
                st.dataframe(
                    uploaded_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                columns = uploaded_df.columns.tolist()

                preferred = [
                    column
                    for column in columns
                    if str(column).strip().lower()
                    in ["text", "message", "sms", "message_text", "content"]
                ]

                default_index = (
                    columns.index(preferred[0]) if preferred else 0
                )

                text_column = st.selectbox(
                    "Select the message-text column",
                    options=columns,
                    index=default_index,
                )

                if st.button(
                    "🚀 Analyze All Messages",
                    use_container_width=True,
                ):
                    messages = uploaded_df[text_column].fillna("").astype(str)

                    results = []
                    valid_count = 0

                    progress = st.progress(0)
                    status = st.empty()

                    total = len(messages)

                    for index, msg in enumerate(messages):
                        if msg.strip():
                            prediction, probability = predict_message(msg)

                            results.append(
                                {
                                    "Prediction": prediction.upper(),
                                    "Spam probability (%)": (
                                        round(probability, 2)
                                        if probability is not None
                                        else None
                                    ),
                                }
                            )

                            valid_count += 1

                        else:
                            results.append(
                                {
                                    "Prediction": "EMPTY",
                                    "Spam probability (%)": None,
                                }
                            )

                        progress.progress((index + 1) / total)
                        status.write(
                            f"Analyzed {index + 1} of {total} rows..."
                        )

                    results_df = pd.DataFrame(results)

                    output_df = uploaded_df.copy()
                    output_df["SpamShield Prediction"] = results_df[
                        "Prediction"
                    ]
                    output_df["SpamShield Spam Probability (%)"] = results_df[
                        "Spam probability (%)"
                    ]

                    spam_total = int(
                        (output_df["SpamShield Prediction"] == "SPAM").sum()
                    )

                    ham_total = int(
                        (output_df["SpamShield Prediction"] == "HAM").sum()
                    )

                    st.session_state["batch_results"] = output_df

                    # Store non-empty messages in session history.
                    for msg, row in zip(messages, results):
                        if msg.strip() and row["Prediction"] in ["SPAM", "HAM"]:
                            record_prediction(
                                msg,
                                row["Prediction"].lower(),
                                row["Spam probability (%)"],
                            )

                    status.success("Batch analysis completed.")

                    a, b, c = st.columns(3)

                    with a:
                        metric_card("Total Rows", total)

                    with b:
                        metric_card("Spam Messages", spam_total)

                    with c:
                        metric_card("Legitimate Messages", ham_total)

                    st.markdown("### Classification Results")

                    st.dataframe(
                        output_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇️ Download Results as CSV",
                        data=output_df.to_csv(index=False).encode("utf-8"),
                        file_name="spamshield_analysis.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.caption(
                        f"Non-empty messages analyzed: {valid_count}. "
                        "Blank rows are marked EMPTY."
                    )

        except (
            UnicodeError,
            pd.errors.ParserError,
            ValueError,
            OSError,
        ) as error:
            st.error(f"Could not read this CSV file: {error}")

    elif "batch_results" in st.session_state:
        st.markdown("### Previous Batch Results")
        previous_results = st.session_state["batch_results"]

        st.dataframe(
            previous_results,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Previous Results",
            data=previous_results.to_csv(index=False).encode("utf-8"),
            file_name="spamshield_analysis.csv",
            mime="text/csv",
        )

# ==================================================
# PAGE 4: MODEL INSIGHTS
# ==================================================

elif page == "📊 Model Insights":

    st.markdown(
        '<div class="section-title">Model Performance & Insights</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Evaluate the saved model against the held-out test split of the "
        "SMS Spam Collection dataset."
    )

    st.warning(
        "These results are meaningful only if this is the same dataset and "
        "split used to train the saved model. Do not use training-set scores "
        "as evidence of generalization."
    )

    dataset = load_dataset()

    if dataset is None:
        st.error(
            f"Dataset file '{DATASET_PATH}' was not found or could not be "
            "read. Keep the original SMSSpamCollection file in the same "
            "folder as app.py."
        )

    else:
        st.markdown("### Dataset Overview")

        total_rows = len(dataset)
        spam_rows = int((dataset["label"] == "spam").sum())
        ham_rows = int((dataset["label"] == "ham").sum())

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Total Messages", total_rows, "Loaded dataset")

        with c2:
            metric_card("Spam Examples", spam_rows, "Labelled spam")

        with c3:
            metric_card("Ham Examples", ham_rows, "Labelled legitimate")

        split_size = st.slider(
            "Test-set size",
            min_value=10,
            max_value=30,
            value=20,
            step=5,
            help=(
                "Use 20% to match the original 80/20 split if your "
                "training script used test_size=0.2."
            ),
        )

        if st.button(
            "📈 Evaluate Saved Model",
            use_container_width=True,
        ):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset["text"],
                    dataset["label"],
                    test_size=split_size / 100,
                    random_state=42,
                    stratify=dataset["label"],
                )

                predictions = model.predict(X_test)
                predictions = [
                    str(value).lower() for value in predictions
                ]

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(
                    y_test,
                    predictions,
                    pos_label="spam",
                    zero_division=0,
                )
                recall = recall_score(
                    y_test,
                    predictions,
                    pos_label="spam",
                    zero_division=0,
                )
                f1 = f1_score(
                    y_test,
                    predictions,
                    pos_label="spam",
                    zero_division=0,
                )

                cm = confusion_matrix(
                    y_test,
                    predictions,
                    labels=["ham", "spam"],
                )

                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    metric_card("Accuracy", f"{accuracy * 100:.2f}%")

                with m2:
                    metric_card("Spam Precision", f"{precision * 100:.2f}%")

                with m3:
                    metric_card("Spam Recall", f"{recall * 100:.2f}%")

                with m4:
                    metric_card("Spam F1 Score", f"{f1 * 100:.2f}%")

                st.markdown("---")
                st.markdown("### Confusion Matrix")

                matrix_df = pd.DataFrame(
                    cm,
                    index=["Actual Ham", "Actual Spam"],
                    columns=["Predicted Ham", "Predicted Spam"],
                )

                fig = px.imshow(
                    matrix_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Purples",
                    labels={
                        "x": "Predicted label",
                        "y": "Actual label",
                        "color": "Messages",
                    },
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Classification Report")

                report = classification_report(
                    y_test,
                    predictions,
                    labels=["ham", "spam"],
                    target_names=["Ham", "Spam"],
                    output_dict=True,
                    zero_division=0,
                )

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True,
                )

                st.markdown("### Understanding the Metrics")

                st.markdown(
                    """
                    - **Accuracy:** percentage of all test messages classified correctly.
                    - **Precision:** of messages predicted as spam, how many were actually spam.
                    - **Recall:** of actual spam messages, how many the model detected.
                    - **F1 score:** a combined measure of spam precision and recall.
                    - **Confusion matrix:** shows correct and incorrect predictions for each class.
                    """
                )

                st.caption(
                    "This evaluation uses the selected test split and "
                    "random_state=42. Confirm that these settings match your "
                    "training script before comparing results with your README."
                )

            except (ValueError, AttributeError, KeyError) as error:
                st.error(
                    "Evaluation failed. Check that the saved model was "
                    "trained for the same dataset and label format."
                )
                st.code(str(error))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#887DA5; padding:10px;">
        <b>SpamShield</b> · SMS Spam Detection System<br>
        <span style="font-size:12px;">
            Machine learning classification is not a guarantee of message safety.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)