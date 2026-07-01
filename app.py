import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import time


st.set_page_config(page_title="🕵️ AI Powered Fake Job Posting Detection System", page_icon="favicon.png", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0A0612;
        background-image:
            radial-gradient(ellipse at 15% 20%, rgba(139, 92, 246, 0.12) 0%, transparent 55%),
            radial-gradient(ellipse at 85% 75%, rgba(192, 132, 252, 0.10) 0%, transparent 55%),
            radial-gradient(ellipse at 50% 50%, rgba(109, 40, 217, 0.06) 0%, transparent 60%),
            linear-gradient(180deg, #0A0612 0%, #100820 50%, #150D2E 100%);
        color: #EDE9FE;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E0A1E 0%, #150D2E 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    section[data-testid="stSidebar"] * {
        color: #C4B5FD !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #A78BFA !important;
    }

    /* Main title */
    h1 {
        color: #A78BFA !important;
        letter-spacing: -0.5px;
        text-shadow: 0 0 30px rgba(167, 139, 250, 0.4);
    }
    h2, h3 { color: #C4B5FD !important; }

    /* Text inputs & text areas */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #150D2E !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 10px !important;
        color: #EDE9FE !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
    }

    /* Labels */
    .stTextInput label, .stTextArea label {
        color: #C4B5FD !important;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.4px;
    }

    /* Primary buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6D28D9 0%, #8B5CF6 60%, #A78BFA 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        padding: 0.5rem 1.4rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        box-shadow: 0 4px 20px rgba(109, 40, 217, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 32px rgba(139, 92, 246, 0.55) !important;
        filter: brightness(1.12);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Download button */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid rgba(139, 92, 246, 0.5) !important;
        color: #A78BFA !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: rgba(139, 92, 246, 0.12) !important;
        border-color: #8B5CF6 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(139, 92, 246, 0.07) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 10px !important;
        color: #C4B5FD !important;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background: rgba(15, 10, 30, 0.75) !important;
        border: 1px solid rgba(139, 92, 246, 0.12) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 10px !important;
        border-left-width: 4px !important;
    }

    /* Spinner */
    .stSpinner > div { color: #A78BFA !important; }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6D28D9, #8B5CF6, #C4B5FD) !important;
        border-radius: 99px;
    }

    /* Horizontal rule */
    hr { border-color: rgba(139, 92, 246, 0.18) !important; }

    /* Caption */
    .stCaption { color: #C4B5FD !important; }

    /* Info box */
    div[data-testid="stInfo"] {
        background: rgba(109, 40, 217, 0.12) !important;
        border-left: 4px solid #7C3AED !important;
        border-radius: 8px !important;
        color: #C4B5FD !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0A0612; }
    ::-webkit-scrollbar-thumb { background: #6D28D9; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #8B5CF6; }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('model/xgb_final_model.joblib')
#st.write(model.classes_)
explainer = shap.TreeExplainer(model)
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')

st.sidebar.title("About the Project")
st.sidebar.info("""
Detect whether a job posting is real or fake using a trained NLP model.  
Developed by **Krishna Kailas Paralkar** using TF-IDF + XGBoost.
""")
st.sidebar.markdown("---")
st.sidebar.write("📊 Model: XGBoost")
st.sidebar.write("🔤 Text Vectorizer: TF-IDF")
st.sidebar.write("🔎 Explainability: LIME & SHAP (explained in GitHub repo)")

# Main title
st.title("🕵️ AI Powered Fake Job Posting Detection System")
st.markdown("Analyze job postings using Machine Learning and predict whether they are Legitimate or Fraudulent")

# Divider
st.markdown("---")
st.subheader("✍️ Enter Job Details Below")

    
    
if "example" not in st.session_state:
    st.session_state.example = False

def set_example_inputs():
    st.session_state.title = "Data Scientist"
    st.session_state.description = "We are looking for a Data Scientist to analyze large amounts of raw information to find patterns that will help improve our company."
    st.session_state.requirements = "Experience in Python, SQL, machine learning, data visualization. Familiarity with cloud platforms is a plus."

st.button("🎯 Try with Example Job", on_click=set_example_inputs)

    
title = st.text_input("Job Title", key="title", placeholder="e.g., Data Scientist")
description = st.text_area("Job Description", key="description", height=150, placeholder="Describe the job in detail...")
requirements = st.text_area("Job Requirements", key="requirements", height=150, placeholder="List the job requirements...")

col1, col2 = st.columns([5,6])
col1, col_spacer, col2 =st.columns([1,2,1])

with col2:
    def clear_inputs():
        st.session_state.title = ""
        st.session_state.description = ""
        st.session_state.requirements = ""
    st.button("🧹Clear Fields", on_click=clear_inputs)


# Predict Button
with col1:
    if st.button("🫆 Predict"):
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
        if not title or not description or not requirements:
            st.warning("⚠️ Please fill in all the fields before predicting.")
        else:
            text = title + " " + description + " " + requirements
            text_vector = vectorizer.transform([text])
            prediction = model.predict(text_vector)[0]
            probability = model.predict_proba(text_vector)[0][int(prediction)]
            
        if prediction == 0:
            st.success(f"✅ This job posting looks **Real** (Confidence: {probability:.2f})")
        else:
            st.error(f"❌ This job posting looks **Fake** (Confidence: {probability:.2f})")
            
        result_text = f"Prediction: {'Real' if prediction == 0 else 'Fake'}\nConfidence: {probability:.2f}"
        st.download_button("📄 Download Prediction", result_text, file_name="prediction.txt")

       
# ✅ SHAP Waterfall Plot
        shap_values = explainer(text_vector)
        dense_vec = text_vector.toarray()[0]
        feature_names = vectorizer.get_feature_names_out()

        input_features = {
            name: dense_vec[i]
            for i, name in enumerate(feature_names)
            if dense_vec[i] > 0
        }

        shap_exp = shap.Explanation(
            values=shap_values.values[0][dense_vec > 0],
            base_values=shap_values.base_values[0],
            data=dense_vec[dense_vec > 0],
            feature_names=list(input_features.keys())
        )

        st.subheader("🔍 Why did the model make this prediction?")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        st.pyplot(fig)
            
# Extra details after prediction
        st.markdown(f"🔢 **Words in input**: {len(text.split())}")
        st.progress(int(probability* 100))
        st.caption(f"Model confidence: {probability:.2%}")

        if probability > 0.75:
            st.info("🧠 The model is quite confident in this prediction.")
        elif probability < 0.55:
            st.warning("🤔 The model isn't very confident. Review manually.")

            st.markdown("---")
with st.expander("📊 SHAP Global Explanation (Beeswarm Plot)"):
    st.write("The plot below shows how each feature impacts the model output globally.")
    st.image("visuals/shap_beeswarm_plot.png", caption="SHAP Summary Plot (Beeswarm)", use_column_width=True)
            
# How it works
with st.expander("ℹ️ How this works"):
    st.write("""
    This tool uses a trained XGBoost model that analyzes job title, description, and requirements using TF-IDF features.
    It was trained on a real-world job postings dataset and predicts whether a job post is fake or real.
    """)

# Footer
st.markdown("---")
st.markdown("Developed by Krishna Kailas Paralkar | [Linkedin](https://www.linkedin.com/in/krishna-paralkar-0497003bb/) | [GitHub Repo](https://github.com/KrishnaParalkar-5034/AI_Powered_Fake_Job_Detection.git)")