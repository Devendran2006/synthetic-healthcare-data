import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Synthetic Healthcare Data Generation",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    real = pd.read_csv("data/ctgan_ready.csv")
    synthetic = pd.read_csv("data/synthetic_healthcare.csv")
    return real, synthetic

real_df, synthetic_df = load_data()

# --------------------------------------------------
# GLOBAL FILTERS
# --------------------------------------------------

st.sidebar.header("🔍 Global Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=real_df["Gender"].unique(),
    default=real_df["Gender"].unique()
)

disease_filter = st.sidebar.multiselect(
    "Medical Condition",
    options=real_df["Medical Condition"].unique(),
    default=real_df["Medical Condition"].unique()
)

admission_filter = st.sidebar.multiselect(
    "Admission Type",
    options=real_df["Admission Type"].unique(),
    default=real_df["Admission Type"].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    int(real_df["Age"].min()),
    int(real_df["Age"].max()),
    (
        int(real_df["Age"].min()),
        int(real_df["Age"].max())
    )
)

real_df = real_df[
    (real_df["Gender"].isin(gender_filter)) &
    (real_df["Medical Condition"].isin(disease_filter)) &
    (real_df["Admission Type"].isin(admission_filter)) &
    (real_df["Age"].between(age_range[0], age_range[1]))
]

synthetic_df = synthetic_df[
    (synthetic_df["Gender"].isin(gender_filter)) &
    (synthetic_df["Medical Condition"].isin(disease_filter)) &
    (synthetic_df["Admission Type"].isin(admission_filter)) &
    (synthetic_df["Age"].between(age_range[0], age_range[1]))
]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏥 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Executive Summary",
        "MLOps Monitor",
        "Model Quality",
        "Data Analysis",
        "Real vs Synthetic",
        "CTGAN Results",
        "CRM Analytics",
        "Clinical Insights",
        "Synthetic Intelligence",
        "Dataset Preview",
        "Download Data",
        "Future Scope"
    ]
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("🏥 Privacy-Preserving Synthetic Healthcare Data Generation")

    st.markdown("""
    This project generates synthetic healthcare data using CTGAN
    while preserving patient privacy through anonymization.
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Real Records", len(real_df))

    with col2:
        st.metric("Synthetic Records", len(synthetic_df))

    with col3:
        st.metric("Real Accuracy", "29.58%")

    with col4:
        st.metric("Synthetic Accuracy", "28.97%")

    st.divider()

    st.subheader("Project Workflow")

    st.code("""
Healthcare Dataset
↓
PII Detection
↓
Anonymization
↓
Preprocessing
↓
CTGAN Training
↓
Synthetic Data Generation
↓
Evaluation
↓
ML Validation
""")

    st.divider()

    st.subheader("Utility Retention")

    st.metric(
        "Utility Retention",
        "97.94%"
    )

    st.success(
        "Synthetic data preserves most of the predictive power of the original dataset."
    )

    st.divider()

    st.subheader("Project Architecture")

    st.code("""
Raw Healthcare Data
        ↓
PII Detection
        ↓
Anonymization
        ↓
Preprocessing
        ↓
CTGAN Training
        ↓
Synthetic Dataset
        ↓
Evaluation
        ↓
Dashboard Analytics
""")

elif page == "Executive Summary":

    st.title("📊 Executive Summary")

    total_patients = len(real_df)

    total_diseases = real_df[
        "Medical Condition"
    ].nunique()

    avg_age = round(
        real_df["Age"].mean(),
        1
    )

    avg_bill = round(
        real_df["Billing Amount"].mean(),
        2
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Patients",
            f"{total_patients:,}"
        )

    with col2:
        st.metric(
            "Unique Diseases",
            total_diseases
        )

    with col3:
        st.metric(
            "Average Age",
            avg_age
        )

    with col4:
        st.metric(
            "Average Billing",
            f"${avg_bill:,.0f}"
        )
        st.divider()

    metrics_df = pd.DataFrame({
        "Metric": [
            "Quality Score",
            "Privacy Score",
            "Correlation"
        ],
        "Score": [
            83.87,
            100,
            97.44
        ]
    })

    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Score",
        text="Score",
        title="Model Performance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )    
    st.divider()

    top_disease = (
        real_df[
            "Medical Condition"
        ]
        .value_counts()
        .idxmax()
    )

    highest_cost = (
        real_df.groupby(
            "Medical Condition"
        )["Billing Amount"]
        .mean()
        .idxmax()
    )

    highest_risk = (
        real_df.groupby(
            "Medical Condition"
        )["Length_of_Stay"]
        .mean()
        .idxmax()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success(
            f"Most Common Disease\n\n{top_disease}"
        )

    with col2:
        st.warning(
            f"Highest Cost Disease\n\n{highest_cost}"
        )

    with col3:
        st.error(
            f"Highest Risk Disease\n\n{highest_risk}"
        )
# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

elif page == "Executive Summary":

    st.title("📊 Executive Summary")

    total_revenue = real_df["Billing Amount"].sum()

    top_disease = (
        real_df["Medical Condition"]
        .value_counts()
        .idxmax()
    )

    top_insurance = (
        real_df["Insurance Provider"]
        .value_counts()
        .idxmax()
    )

    avg_stay = round(
        real_df["Length_of_Stay"].mean(),
        2
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Patients",
            len(real_df)
        )

    with col2:
        st.metric(
            "Revenue",
            f"${total_revenue:,.0f}"
        )

    with col3:
        st.metric(
            "Top Disease",
            top_disease
        )

    with col4:
        st.metric(
            "Top Insurance",
            top_insurance
        )

    with col5:
        st.metric(
            "Average Stay",
            avg_stay
        )

    st.divider()

    disease_count = (
        real_df["Medical Condition"]
        .value_counts()
        .reset_index()
    )

    disease_count.columns = [
        "Disease",
        "Patients"
    ]

    fig = px.bar(
        disease_count,
        x="Disease",
        y="Patients",
        title="Patient Distribution by Disease"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    insurance_count = (
        real_df["Insurance Provider"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    insurance_count.columns = [
        "Insurance Provider",
        "Patients"
    ]

    fig = px.pie(
        insurance_count,
        names="Insurance Provider",
        values="Patients",
        title="Top Insurance Providers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# MLOPS MONITOR
# --------------------------------------------------

elif page == "MLOps Monitor":

    st.title("⚙️ MLOps Monitoring Dashboard")

    metrics = pd.read_csv(
        "metrics/latest_metrics.csv"
    )

    latest = metrics.iloc[0]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Quality Score",
            latest["Quality Score"]
        )

        st.metric(
            "Privacy Score",
            latest["Privacy Score"]
        )

    with col2:

        st.metric(
            "Correlation Similarity",
            latest["Correlation Similarity"]
        )

        st.metric(
            "Last Training",
            str(
                latest["Timestamp"]
            )
        )

    st.divider()

    st.subheader(
        "Current Model"
    )

    st.success(
        "CTGAN Model Registered Successfully"
    )
    history = pd.read_csv(
        "metrics/training_history.csv"
    )

    st.divider()

    st.subheader(
        "Training History"
    )

    import plotly.express as px

    fig = px.line(
        history,
        x="Timestamp",
        y="Quality Score",
        markers=True,
        title="Quality Score Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )   
# --------------------------------------------------
# MODEL QUALITY
# --------------------------------------------------

elif page == "Model Quality":

    st.title("📈 Model Quality Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Quality Score", "83.87%")
        st.progress(0.8387)

    with col2:
        st.metric("Privacy Score", "100%")
        st.progress(1.0)

    with col3:
        st.metric("Correlation Similarity", "97.44%")
        st.progress(0.9744)

    with col4:
        st.metric("Utility Retention", "97.94%")
        st.progress(0.9794)

    st.divider()

    quality_df = pd.DataFrame({
        "Metric": [
            "Quality",
            "Privacy",
            "Correlation",
            "Utility"
        ],
        "Score": [
            83.87,
            100,
            97.44,
            97.94
        ]
    })

    fig = px.bar(
        quality_df,
        x="Metric",
        y="Score",
        text="Score",
        title="Model Performance Metrics"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# REAL VS SYNTHETIC
# --------------------------------------------------

elif page == "Real vs Synthetic":

    st.title("⚖️ Real vs Synthetic Comparison")

    st.subheader("Age Distribution")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            real_df,
            x="Age",
            nbins=20,
            title="Real Data Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            synthetic_df,
            x="Age",
            nbins=20,
            title="Synthetic Data Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Gender Comparison")

    col1, col2 = st.columns(2)

    real_gender = (
        real_df["Gender"]
        .value_counts()
        .reset_index()
    )

    real_gender.columns = [
        "Gender",
        "Count"
    ]

    synthetic_gender = (
        synthetic_df["Gender"]
        .value_counts()
        .reset_index()
    )

    synthetic_gender.columns = [
        "Gender",
        "Count"
    ]

    with col1:

        fig = px.pie(
            real_gender,
            names="Gender",
            values="Count",
            title="Real Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            synthetic_gender,
            names="Gender",
            values="Count",
            title="Synthetic Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Disease Comparison")

    col1, col2 = st.columns(2)

    real_condition = (
        real_df["Medical Condition"]
        .value_counts()
        .reset_index()
    )

    real_condition.columns = [
        "Disease",
        "Count"
    ]

    synthetic_condition = (
        synthetic_df["Medical Condition"]
        .value_counts()
        .reset_index()
    )

    synthetic_condition.columns = [
        "Disease",
        "Count"
    ]

    with col1:

        fig = px.bar(
            real_condition,
            x="Disease",
            y="Count",
            title="Real Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            synthetic_condition,
            x="Disease",
            y="Count",
            title="Synthetic Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# --------------------------------------------------
# DATA ANALYSIS
# --------------------------------------------------

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    st.subheader("Medical Condition Distribution")

    col1, col2 = st.columns(2)

    real_condition = (
        real_df["Medical Condition"]
        .value_counts()
        .reset_index()
    )

    real_condition.columns = [
        "Condition",
        "Count"
    ]

    synthetic_condition = (
        synthetic_df["Medical Condition"]
        .value_counts()
        .reset_index()
    )

    synthetic_condition.columns = [
        "Condition",
        "Count"
    ]

    with col1:

        fig = px.bar(
            real_condition,
            x="Condition",
            y="Count",
            title="Real Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            synthetic_condition,
            x="Condition",
            y="Count",
            title="Synthetic Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Gender Distribution")

    col1, col2 = st.columns(2)

    real_gender = (
        real_df["Gender"]
        .value_counts()
        .reset_index()
    )

    real_gender.columns = [
        "Gender",
        "Count"
    ]

    synthetic_gender = (
        synthetic_df["Gender"]
        .value_counts()
        .reset_index()
    )

    synthetic_gender.columns = [
        "Gender",
        "Count"
    ]

    with col1:

        fig = px.pie(
            real_gender,
            names="Gender",
            values="Count",
            title="Real Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            synthetic_gender,
            names="Gender",
            values="Count",
            title="Synthetic Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Age Distribution")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            real_df,
            x="Age",
            nbins=20,
            title="Real Dataset Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            synthetic_df,
            x="Age",
            nbins=20,
            title="Synthetic Dataset Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
# --------------------------------------------------
# CTGAN RESULTS
# --------------------------------------------------

elif page == "CTGAN Results":

    st.title("🤖 CTGAN Results")

    comparison = pd.DataFrame({
        "Dataset": [
            "Real Data",
            "Synthetic Data"
        ],
        "Accuracy": [
            29.58,
            28.97
        ]
    })

    fig = px.bar(
        comparison,
        x="Dataset",
        y="Accuracy",
        text="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Quality Score",
            "83.87%"
        )

    with col2:
        st.metric(
            "Privacy Score",
            "100%"
        )

    with col3:
        st.metric(
            "Correlation Similarity",
            "97.44%"
        )

    with col4:
        st.metric(
            "Utility Retention",
            "97.94%"
        )

    st.divider()

    metrics_df = pd.DataFrame({
        "Metric": [
            "Quality",
            "Privacy",
            "Correlation",
            "Utility"
        ],
        "Score": [
            83.87,
            100,
            97.44,
            97.94
        ]
    })

    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Score",
        text="Score",
        title="CTGAN Evaluation Metrics"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# CRM ANALYTICS
# --------------------------------------------------

elif page == "CRM Analytics":

    st.title("👥 CRM Analytics Dashboard")

    total_patients = len(real_df)

    total_revenue = real_df[
        "Billing Amount"
    ].sum()

    avg_stay = round(
        real_df[
            "Length_of_Stay"
        ].mean(),
        2
    )

    total_diseases = real_df[
        "Medical Condition"
    ].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Patients",
            f"{total_patients:,}"
        )

    with col2:
        st.metric(
            "Revenue",
            f"${total_revenue:,.0f}"
        )

    with col3:
        st.metric(
            "Unique Diseases",
            total_diseases
        )

    with col4:
        st.metric(
            "Average Stay",
            avg_stay
        )

    st.divider()

    st.subheader(
        "🏥 Disease Distribution"
    )

    disease_df = (
        real_df[
            "Medical Condition"
        ]
        .value_counts()
        .reset_index()
    )

    disease_df.columns = [
        "Disease",
        "Patients"
    ]

    fig = px.bar(
        disease_df,
        x="Disease",
        y="Patients",
        text="Patients",
        title="Patients by Disease"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "💰 Revenue by Disease"
    )

    revenue_df = (
        real_df
        .groupby(
            "Medical Condition"
        )[
            "Billing Amount"
        ]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        revenue_df,
        x="Medical Condition",
        y="Billing Amount",
        title="Revenue Contribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "🏢 Insurance Provider Analysis"
    )

    insurance_df = (
        real_df[
            "Insurance Provider"
        ]
        .value_counts()
        .head(10)
        .reset_index()
    )

    insurance_df.columns = [
        "Provider",
        "Patients"
    ]

    fig = px.pie(
        insurance_df,
        names="Provider",
        values="Patients",
        title="Top Insurance Providers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "📈 Average Stay by Disease"
    )

    stay_df = (
        real_df
        .groupby(
            "Medical Condition"
        )[
            "Length_of_Stay"
        ]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        stay_df,
        x="Medical Condition",
        y="Length_of_Stay",
        title="Average Hospital Stay"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "👨‍⚕️ Admission Type Analysis"
    )

    admission_df = (
        real_df[
            "Admission Type"
        ]
        .value_counts()
        .reset_index()
    )

    admission_df.columns = [
        "Admission Type",
        "Count"
    ]

    fig = px.pie(
        admission_df,
        names="Admission Type",
        values="Count",
        title="Admission Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# --------------------------------------------------
# CLINICAL INSIGHTS
# --------------------------------------------------

elif page == "Clinical Insights":

    st.title("🩺 Clinical Insights Dashboard")

    filtered = real_df.copy()

    col1, col2, col3 = st.columns(3)

    with col1:

        top_disease = (
            filtered["Medical Condition"]
            .value_counts()
            .idxmax()
        )

        st.metric(
            "Most Common Disease",
            top_disease
        )

    with col2:

        avg_age = round(
            filtered["Age"].mean(),
            1
        )

        st.metric(
            "Average Patient Age",
            avg_age
        )

    with col3:

        avg_stay = round(
            filtered["Length_of_Stay"].mean(),
            1
        )

        st.metric(
            "Average Stay",
            avg_stay
        )

    st.divider()

    st.subheader(
        "Disease Distribution"
    )

    disease_df = (
        filtered["Medical Condition"]
        .value_counts()
        .reset_index()
    )

    disease_df.columns = [
        "Disease",
        "Count"
    ]

    fig = px.bar(
        disease_df,
        x="Disease",
        y="Count",
        text="Count",
        title="Disease Frequency"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Blood Type Analysis"
    )

    blood_df = (
        filtered["Blood Type"]
        .value_counts()
        .reset_index()
    )

    blood_df.columns = [
        "Blood Type",
        "Count"
    ]

    fig = px.pie(
        blood_df,
        names="Blood Type",
        values="Count",
        title="Blood Type Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Test Result Analysis"
    )

    test_df = (
        filtered["Test Results"]
        .value_counts()
        .reset_index()
    )

    test_df.columns = [
        "Result",
        "Count"
    ]

    fig = px.bar(
        test_df,
        x="Result",
        y="Count",
        text="Count",
        title="Test Results Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "AI Generated Clinical Insights"
    )

    most_common = (
        filtered["Medical Condition"]
        .value_counts()
        .idxmax()
    )

    highest_age = int(
        filtered["Age"].max()
    )

    lowest_age = int(
        filtered["Age"].min()
    )

    st.success(
        f"""
        • Most frequent disease: {most_common}

        • Age range observed: {lowest_age} - {highest_age}

        • CTGAN preserved major healthcare patterns.

        • Synthetic dataset suitable for model training,
          dashboard testing and research.
        """
    )
# --------------------------------------------------
# SYNTHETIC DATA INTELLIGENCE
# --------------------------------------------------

elif page == "Synthetic Intelligence":

    st.title("🧠 Synthetic Healthcare Intelligence")

    st.markdown(
        """
        AI-driven analysis generated from
        synthetic healthcare records.
        """
    )

    st.divider()

    # Disease Selector

    disease = st.selectbox(
        "Select Disease",
        sorted(
            synthetic_df[
                "Medical Condition"
            ].unique()
        )
    )

    filtered = synthetic_df[
        synthetic_df[
            "Medical Condition"
        ] == disease
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Patients",
            len(filtered)
        )

    with col2:

        st.metric(
            "Average Age",
            round(
                filtered["Age"].mean(),
                1
            )
        )

    with col3:

        st.metric(
            "Average Stay",
            round(
                filtered[
                    "Length_of_Stay"
                ].mean(),
                1
            )
        )

    st.divider()
    st.subheader(
        "Gender Distribution"
    )

    gender_df = (
        filtered["Gender"]
        .value_counts()
        .reset_index()
    )

    gender_df.columns = [
        "Gender",
        "Count"
    ]

    fig = px.pie(
        gender_df,
        names="Gender",
        values="Count",
        title=f"{disease} Gender Split"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Age Distribution"
    )

    fig = px.histogram(
        filtered,
        x="Age",
        nbins=20,
        title=f"{disease} Age Pattern"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.divider()

    st.subheader(
        "Length Of Stay Pattern"
    )

    fig = px.box(
        filtered,
        y="Length_of_Stay",
        title=f"{disease} Hospital Stay"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Billing Analysis"
    )

    fig = px.histogram(
        filtered,
        x="Billing Amount",
        nbins=25,
        title=f"{disease} Billing Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.divider()

    st.subheader(
        "AI Generated Insights"
    )

    avg_age = round(
        filtered["Age"].mean(),
        1
    )

    avg_stay = round(
        filtered[
            "Length_of_Stay"
        ].mean(),
        1
    )

    avg_bill = round(
        filtered[
            "Billing Amount"
        ].mean(),
        2
    )

    st.success(
        f"""
        Disease: {disease}

        Average Age:
        {avg_age}

        Average Length Of Stay:
        {avg_stay} days

        Average Billing:
        ${avg_bill}

        This synthetic pattern can be used
        for healthcare analytics,
        model development,
        dashboard testing,
        and AI experimentation.
        """
    )
elif page == "Synthetic Intelligence":

    st.title("🧠 Synthetic Healthcare Intelligence")

    # Disease Selector

    # KPI Cards

    # Gender Distribution

    # Age Distribution

    # Length Of Stay Pattern

    # Billing Analysis

    # AI Generated Insights

    st.success(
        f"""
        Disease: {disease}
        Average Age: {avg_age}
        Average Length Of Stay: {avg_stay}
        Average Billing: ${avg_bill}
        """
    )

    # ===========================
    # PASTE NEW CODE HERE
    # ===========================

    st.divider()

    st.subheader("🏆 Top Risk Diseases")

    risk_df = (
        synthetic_df.groupby(
            "Medical Condition"
        )["Length_of_Stay"]
        .mean()
        .reset_index()
        .sort_values(
            "Length_of_Stay",
            ascending=False
        )
    )

    fig = px.bar(
        risk_df,
        x="Medical Condition",
        y="Length_of_Stay",
        title="Average Hospital Stay by Disease"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Highest Cost Disease

    st.divider()

    st.subheader("💰 Highest Cost Diseases")

    # ... continue remaining code ...

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

elif page == "Dataset Preview":

    st.title("📋 Dataset Preview")

    tab1, tab2 = st.tabs(
        [
            "Real Dataset",
            "Synthetic Dataset"
        ]
    )

    with tab1:

        st.subheader(
            "Real Dataset Sample"
        )

        st.dataframe(
            real_df.head(100),
            use_container_width=True
        )

    with tab2:

        st.subheader(
            "Synthetic Dataset Sample"
        )

        st.dataframe(
            synthetic_df.head(100),
            use_container_width=True
        )

# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

elif page == "Download Data":

    st.title("📥 Download Center")

    st.subheader(
        "Synthetic Dataset"
    )

    csv = synthetic_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Synthetic Dataset",
        data=csv,
        file_name="synthetic_healthcare.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader(
        "Project Metrics"
    )

    metrics_df = pd.DataFrame({
        "Metric": [
            "Quality Score",
            "Privacy Score",
            "Correlation Similarity",
            "Utility Retention"
        ],
        "Value": [
            "83.87%",
            "100%",
            "97.44%",
            "97.94%"
        ]
    })

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

# --------------------------------------------------
# FUTURE SCOPE
# --------------------------------------------------

elif page == "Future Scope":

    st.title("🚀 Future Roadmap")

    st.markdown("""
### Phase 1 - Completed

✅ PII Detection

✅ Data Anonymization

✅ CTGAN Training

✅ Synthetic Data Generation

✅ Quality Evaluation

✅ Privacy Evaluation

✅ Correlation Evaluation

✅ Streamlit Dashboard

✅ GitHub CI/CD

---

### Phase 2

🔹 MIMIC-IV Clinical Integration

🔹 ECG Synthetic Data Generation

🔹 MRI Synthetic Data Generation

🔹 X-Ray Synthetic Data Generation

🔹 SDMetrics Advanced Reports

🔹 Automated Retraining Pipeline

---

### Phase 3

🔹 AI Disease Risk Prediction

🔹 AI Clinical Pattern Discovery

🔹 Healthcare CRM Intelligence

🔹 Synthetic Digital Patient Framework

🔹 MLOps Automation

🔹 Docker Deployment

🔹 AWS Deployment

🔹 Azure Deployment

🔹 Kubernetes Deployment
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Synthetic Healthcare Data Generation using CTGAN | Privacy Preserving AI Project"
)