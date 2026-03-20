# app.py
import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

st.set_page_config(layout="wide", page_title="Calgary Heritage Buildings")

# custom CSS
st.markdown("""
    <style>
        /* global font size */
        html, body, [class*="css"] {
            font-size: 20px !important;
        }
        
        /* input fields */
        input, textarea, select {
            font-size: 20px !important;
        }
        
        /* selectbox and dropdowns */
        div[data-baseweb="select"] {
            font-size: 20px !important;
        }
        
        div[data-baseweb="input"] {
            font-size: 20px !important;
        }
        
        /* slider */
        div[data-testid="stSlider"] {
            font-size: 20px !important;
        }
        
        /* labels */
        label, p {
            font-size: 20px !important;
        }

        /* table */
        td, th {
            font-size: 20px !important;
        }

        /* metric */
        div[data-testid="metric-container"] > div {
            font-size: 20px !important;
        }
    </style>
""", unsafe_allow_html=True)
# load model and results
@st.cache_data
def load_data():
    results = pd.read_csv("results-basic.csv").sort_values("probability", ascending=False)
    return results

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

results = load_data()
model = load_model()

st.title("Calgary Heritage Building Designator")

df = pd.read_csv("trimmed-data-1.csv")

tab1, tab2 = st.tabs(["Rankings", "Predict New Building"])
# get unique values from your dataset
communities = sorted(df["Community"].unique())
architectural_styles = sorted(df["Architectural Style"].unique())
original_uses = sorted(df["Original Use"].unique())
resource_types = sorted(df["Resource Type"].unique())
development_eras = sorted(df["Development Era"].unique())

print(resource_types)


with tab1:
    st.subheader("Predicted Likelihood of Designation Based on Similarity to Previous Designations")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        n = st.slider("Show top N buildings", 5, len(results), 20, 5)
    
    top_n = results[["Name", "Community", "Year of Construction", "probability"]].head(n).reset_index(drop=True)
    top_n.index += 1
    top_n["probability"] = top_n["probability"].apply(lambda x: f"{x:.1%}")
    top_n.columns = ["Building Name", "Community", "Year of Construction", "Designation Likelihood"]
    st.dataframe(top_n, width='stretch')

with tab2:
    st.subheader("Predict Designation Likelihood")
    
    col1, col2 = st.columns(2)
    
    with col1:
        community = st.selectbox("Community", communities)
        architectural_style = st.selectbox("Architectural Style", architectural_styles)
        original_use = st.selectbox("Original Use", original_uses)
        resource_type = st.selectbox("Resource Type", resource_types)
    
    with col2:
        development_era = st.selectbox("Development Era", development_eras)
        ward = st.selectbox("Ward", sorted(df["Ward"].unique()))
        provincial = st.selectbox("Provincial Designation", [False, True])

    if st.button("Predict"):
        new_building = pd.DataFrame([{
            "Community": community,
            "Ward": ward,
            "Resource Type": resource_type,
            "Development Era": development_era,
            "Architectural Style": architectural_style,
            "Original Use": original_use,
            "Provincial": provincial,
        }])
        new_building = pd.get_dummies(new_building)
        new_building = new_building.reindex(columns=model.get_booster().feature_names, fill_value=0)
        prob = model.predict_proba(new_building)[0][1]
        st.metric("Designation Likelihood", f"{prob:.1%}")