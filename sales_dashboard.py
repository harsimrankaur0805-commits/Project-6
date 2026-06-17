import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Rossmann Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

.kpi-card{
    padding:20px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 5px 15px rgba(0,0,0,0.2);
}

h1,h2,h3{
    color:#1f2937;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():

    train = pd.read_csv("train.csv")
    store = pd.read_csv("store.csv")

    df = pd.merge(train, store, on="Store")

    df["Date"] = pd.to_datetime(df["Date"])

    return df

df = load_data()

# -----------------------------------
# LOAD MODEL
# -----------------------------------

try:
    model = joblib.load("random_forest_model.pkl")
    st.write("Model Loaded Successfully")
except Exception as e:
    st.write("Error:", e)
    model = None

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🎛 Dashboard Filters")


store_type = st.sidebar.multiselect(
    "Store Type",
    options=df["StoreType"].dropna().unique(),
    default=df["StoreType"].dropna().unique()
)

assortment = st.sidebar.multiselect(
    "Assortment",
    options=df["Assortment"].dropna().unique(),
    default=df["Assortment"].dropna().unique()
)
# DATE FILTER
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(
        df["Date"].min(),
        df["Date"].max()
    )
)


filtered_df = df[
    (df["StoreType"].isin(store_type))
    &
    (df["Assortment"].isin(assortment))
]

# -----------------------------------
# TITLE
# -----------------------------------
# Apply Date Filter
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    filtered_df = filtered_df[
        (filtered_df["Date"] >= start_date)
        &
        (filtered_df["Date"] <= end_date)
    ]

st.title("📈 Rossmann Sales Forecasting Dashboard")
st.success("Dashboard Loaded Successfully")

# -----------------------------------
# KPI SECTION
# -----------------------------------

total_sales = filtered_df["Sales"].sum()
avg_sales = filtered_df["Sales"].mean()
total_customers = filtered_df["Customers"].sum()
total_stores = filtered_df["Store"].nunique()

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='kpi-card'
    style='background:linear-gradient(135deg,#667eea,#764ba2)'>
    <h4>💰 Total Sales</h4>
    <h2>{total_sales:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='kpi-card'
    style='background:linear-gradient(135deg,#11998e,#38ef7d)'>
    <h4>📊 Avg Sales</h4>
    <h2>{avg_sales:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='kpi-card'
    style='background:linear-gradient(135deg,#f7971e,#ffd200)'>
    <h4>👥 Customers</h4>
    <h2>{total_customers:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='kpi-card'
    style='background:linear-gradient(135deg,#ff416c,#ff4b2b)'>
    <h4>🏪 Stores</h4>
    <h2>{total_stores}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------
# MONTHLY SALES TREND
# -----------------------------------

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .set_index("Date")
    .resample("ME")["Sales"]
    .sum()
    .reset_index()
)

fig = px.area(
    monthly_sales,
    x="Date",
    y="Sales",
    title="Monthly Revenue Trend",
    color_discrete_sequence=["#667eea"]
)

fig.update_layout(height=500)

st.plotly_chart(fig, width="stretch")
# -----------------------------------
# STORE TYPE + ASSORTMENT
# -----------------------------------

col1,col2 = st.columns(2)

with col1:

    st.subheader("🏪 Sales by Store Type")

    sales_store = (
        filtered_df
        .groupby("StoreType")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        sales_store,
        x="StoreType",
        y="Sales",
        color="Sales",
        color_continuous_scale="Turbo"
    )

    st.plotly_chart(fig, width="stretch")

with col2:

    st.subheader("📦 Assortment Analysis")

    assortment_sales = (
        filtered_df
        .groupby("Assortment")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        assortment_sales,
        names="Assortment",
        values="Sales",
        hole=0.5
    )

    st.plotly_chart(fig, width="stretch")


# -----------------------------------
# TOP STORES
# -----------------------------------

st.subheader("🏆 Top Revenue Stores")

top_stores = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

ffig = px.bar(
    top_stores,
    x="Sales",
    y="Store",
    orientation="h",
    text="Sales",
    color="Sales",
    color_continuous_scale="Viridis",
    template="plotly_white"
)

fig.update_layout(
    height=500,
    yaxis=dict(categoryorder="total ascending")
)

fig.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)

st.plotly_chart(fig, width="stretch")

# -----------------------------------
# COMPETITION IMPACT
# -----------------------------------

if "CompetitionDistance" in filtered_df.columns:

    st.subheader("🎯 Competition Distance Impact")

    sample_df = filtered_df.sample(
        min(3000, len(filtered_df))
    )

    fig = px.scatter(
        sample_df,
        x="CompetitionDistance",
        y="Sales",
        color="StoreType",
        opacity=0.6
    )

    st.plotly_chart(fig, width="stretch")

# -----------------------------------
# HEATMAP
# -----------------------------------

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include="number")

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r"
)

fig.update_layout(height=800)

st.plotly_chart(fig, width="stretch")





