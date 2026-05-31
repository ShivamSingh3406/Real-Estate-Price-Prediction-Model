import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Load model and feature columns ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('real_estate_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_columns():
    with open('feature_columns.pkl', 'rb') as f:
        cols = pickle.load(f)
    return cols

model = load_model()
feature_cols = load_columns()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#1f4e79;'>🏠 Real Estate Price Predictor</h1>
    <p style='text-align:center; color:#555; font-size:16px;'>
        Ames Housing Dataset · Random Forest Model · R² ≈ 0.919
    </p>
    <hr>
""", unsafe_allow_html=True)

st.markdown("### Enter Property Details")
st.markdown("Fill in the key features below. All other features will use dataset defaults.")

# ── Input form ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏗️ Structure & Size**")
    overall_qual    = st.slider("Overall Quality (1-10)", 1, 10, 5)
    overall_cond    = st.slider("Overall Condition (1-10)", 1, 10, 5)
    gr_liv_area     = st.number_input("Above Ground Living Area (sq ft)", 400, 6000, 1500)
    total_bsmt_sf   = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    year_built      = st.number_input("Year Built", 1872, 2010, 1990)
    year_remod      = st.number_input("Year Remodelled", 1950, 2010, 2000)

with col2:
    st.markdown("**🛏️ Rooms & Bathrooms**")
    bedroom_abvgr   = st.slider("Bedrooms Above Ground", 0, 8, 3)
    full_bath       = st.slider("Full Bathrooms", 0, 4, 2)
    half_bath       = st.slider("Half Bathrooms", 0, 2, 0)
    kitchen_qual    = st.selectbox("Kitchen Quality", [5,4,3,2,1],
                                   format_func=lambda x: {5:"Excellent",4:"Good",3:"Typical",2:"Fair",1:"Poor"}[x])
    totrms_abvgrd   = st.slider("Total Rooms Above Ground", 2, 14, 6)
    fireplaces      = st.slider("Fireplaces", 0, 4, 1)

with col3:
    st.markdown("**🚗 Garage & Extras**")
    garage_cars     = st.slider("Garage Capacity (cars)", 0, 5, 2)
    garage_area     = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
    lot_area        = st.number_input("Lot Area (sq ft)", 1300, 200000, 9000)
    exter_qual      = st.selectbox("Exterior Quality", [5,4,3,2,1],
                                   format_func=lambda x: {5:"Excellent",4:"Good",3:"Typical",2:"Fair",1:"Poor"}[x])
    bsmt_qual       = st.selectbox("Basement Quality", [6,5,4,3,2,1],
                                   format_func=lambda x: {6:"Excellent",5:"Good",4:"Typical",3:"Fair",2:"Poor",1:"None"}[x])
    neighborhood    = st.selectbox("Neighborhood", [
        "NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt",
        "Gilbert","Sawyer","NWAmes","SawyerW","BrkSide","Crawfor",
        "Mitchel","NoRidge","Timber","IDOTRR","ClearCr","StoneBr","Others"
    ])

# ── Build input dataframe ───────────────────────────────────────────────────────
def build_input():
    # Start with zeros for all features
    data = {col: 0 for col in feature_cols}

    # Numerical — direct assignments
    data['Overall Qual']      = overall_qual
    data['Overall Cond']      = overall_cond
    data['Gr Liv Area_log1p'] = np.log1p(gr_liv_area)
    data['Year Built']        = year_built
    data['Year Remod/Add']    = year_remod
    data['Bedroom AbvGr']     = bedroom_abvgr
    data['Full Bath']         = full_bath
    data['Half Bath']         = half_bath
    data['Kitchen Qual']      = kitchen_qual
    data['TotRms AbvGrd']     = totrms_abvgrd
    data['Fireplaces']        = fireplaces
    data['Garage Cars']       = garage_cars
    data['Garage Area']       = garage_area
    data['Lot Area']          = lot_area
    data['Exter Qual']        = exter_qual
    data['Bsmt Qual']         = bsmt_qual

    # Transformed features
    data['Lot Frontage_transformed'] = np.sqrt(lot_area) * 0.5  # approximate
    data['BsmtFin SF 1_yeojohnson']  = np.log1p(total_bsmt_sf * 0.6)
    data['Bsmt Unf SF']              = total_bsmt_sf * 0.4

    # Defaults for ordinal features
    data['Lot Shape']      = 4   # Regular
    data['Land Slope']     = 3   # Gentle
    data['Exter Cond']     = 3   # Typical
    data['Bsmt Cond']      = 4   # Typical
    data['Bsmt Exposure']  = 2   # No
    data['BsmtFin Type 1'] = 2   # Unfinished
    data['BsmtFin Type 2'] = 1   # None
    data['Heating QC']     = 3   # Typical
    data['Central Air']    = 1   # Yes
    data['Functional']     = 8   # Typical
    data['Garage Finish']  = 2   # Unfinished
    data['Garage Qual']    = 4   # Typical
    data['Garage Cond']    = 4   # Typical
    data['Paved Drive']    = 3   # Paved
    data['Fireplace Qu']   = 4   # Typical
    data['Street']         = 1   # Paved
    data['Utilities']      = 1   # All public
    data['Condition 2']    = 1   # Normal
    data['Roof Matl']      = 1   # CompShg
    data['Heating']        = 1   # GasA

    # Defaults — numerical misc
    data['Bsmt Full Bath']  = 0
    data['Bsmt Half Bath']  = 0
    data['Kitchen AbvGr']   = 1
    data['Garage Yr Blt']   = year_built
    data['Misc Val']        = 0
    data['Mo Sold']         = 6
    data['Yr Sold']         = 2008
    data['Low Qual Fin SF'] = 0

    # One-hot: MS Zoning → RL (most common)
    if 'MS Zoning_RL' in data: data['MS Zoning_RL'] = 1

    # One-hot: Land Contour → Lvl
    if 'Land Contour_Lvl' in data: data['Land Contour_Lvl'] = 1

    # One-hot: Lot Config → Inside
    if 'Lot Config_Inside' in data: data['Lot Config_Inside'] = 1

    # One-hot: Neighborhood
    nb_col = f'Neighborhood_{neighborhood}'
    if nb_col in data: data[nb_col] = 1

    # One-hot: Condition 1 → Norm
    if 'Condition 1_Norm' in data: data['Condition 1_Norm'] = 1

    # One-hot: Bldg Type → 1Fam (default = 0 for all dummies = 1Fam)
    # One-hot: House Style → 1Story
    if 'House Style_1Story' in data: data['House Style_1Story'] = 1

    # One-hot: Roof Style → Gable
    if 'Roof Style_Gable' in data: data['Roof Style_Gable'] = 1

    # One-hot: Exterior 1st → VinylSd
    if 'Exterior 1st_VinylSd' in data: data['Exterior 1st_VinylSd'] = 1

    # One-hot: Foundation → PConc
    if 'Foundation_PConc' in data: data['Foundation_PConc'] = 1

    # One-hot: Electrical → SBrkr
    if 'Electrical_SBrkr' in data: data['Electrical_SBrkr'] = 1

    # One-hot: Garage Type → Attchd (default = 0 for all dummies)
    # One-hot: Sale Type → WD
    if 'Sale Type_WD' in data: data['Sale Type_WD'] = 1

    # One-hot: Sale Condition → Normal
    if 'Sale Condition_Normal' in data: data['Sale Condition_Normal'] = 1

    return pd.DataFrame([data])[feature_cols]

# ── Predict button ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predict_btn = st.button("🔮 Predict Sale Price", use_container_width=True, type="primary")

if predict_btn:
    input_df = build_input()
    prediction = model.predict(input_df)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #1f4e79, #2e86c1);
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        '>
            <p style='color:#a8d8f0; font-size:18px; margin:0;'>Estimated Sale Price</p>
            <h1 style='color:white; font-size:56px; margin:10px 0;'>
                ${prediction:,.0f}
            </h1>
            <p style='color:#a8d8f0; font-size:14px; margin:0;'>
                Model: Random Forest · R² ≈ 0.919 · RMSE ≈ $25,244
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence range
    margin = 25244
    st.info(f"📊 **Prediction Range:** ${prediction - margin:,.0f} — ${prediction + margin:,.0f}  (±1 RMSE)")

    # Key inputs summary
    with st.expander("📋 Input Summary"):
        summary = {
            "Overall Quality": overall_qual,
            "Overall Condition": overall_cond,
            "Living Area (sq ft)": gr_liv_area,
            "Year Built": year_built,
            "Bedrooms": bedroom_abvgr,
            "Full Baths": full_bath,
            "Garage Cars": garage_cars,
            "Neighborhood": neighborhood,
            "Exterior Quality": {5:"Excellent",4:"Good",3:"Typical",2:"Fair",1:"Poor"}[exter_qual],
            "Kitchen Quality": {5:"Excellent",4:"Good",3:"Typical",2:"Fair",1:"Poor"}[kitchen_qual],
        }
        st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center; color:#aaa; font-size:13px;'>
        Built by <b>Shivam Singh</b> · M.Sc. Data Science & AI, BITS Pilani · 
        <a href='https://github.com/ShivamSingh3406' target='_blank'>GitHub</a>
    </p>
""", unsafe_allow_html=True)
