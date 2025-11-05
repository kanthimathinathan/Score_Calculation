import streamlit as st
import pandas as pd
import openpyxl
from io import BytesIO
import re
from difflib import SequenceMatcher
import os

# Configure page
st.set_page_config(
    page_title="Resolution Scoring & Categorization",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Resolution Scoring & Categorization System")
st.markdown("Upload an Excel file to score resolutions and classify them into categories")

# Sidebar for API key (optional)
with st.sidebar:
    st.header("⚙️ Configuration")
    use_llm = st.checkbox("Use LLM (OpenAI)", value=False)
    api_key = None
    if use_llm:
        api_key = st.text_input("OpenAI API Key", type="password")
        st.caption("Leave empty to use local fallback algorithm")

# Initialize session state
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'df_data' not in st.session_state:
    st.session_state.df_data = None
if 'df_categories' not in st.session_state:
    st.session_state.df_categories = None
if 'processed' not in st.session_state:
    st.session_state.processed = False


# Function to calculate similarity score
def calculate_similarity(summary, resolution):
    """Calculate similarity between summary and resolution using multiple metrics"""
    if pd.isna(summary) or pd.isna(resolution):
        return 1

    summary = str(summary).lower()
    resolution = str(resolution).lower()

    # Metric 1: Sequence matching
    seq_score = SequenceMatcher(None, summary, resolution).ratio()

    # Metric 2: Keyword overlap
    summary_words = set(re.findall(r'\w+', summary))
    resolution_words = set(re.findall(r'\w+', resolution))

    if len(summary_words) == 0:
        keyword_score = 0
    else:
        overlap = len(summary_words.intersection(resolution_words))
        keyword_score = overlap / len(summary_words)

    # Metric 3: Length ratio (penalize too short or too long resolutions)
    len_ratio = min(len(resolution), len(summary)) / max(len(resolution), len(summary), 1)

    # Combined score
    combined_score = (seq_score * 0.3 + keyword_score * 0.5 + len_ratio * 0.2)

    # Scale to 1-10 with some logic
    # If resolution is too short relative to problem, reduce score
    if len(resolution) < len(summary) * 0.2:
        combined_score *= 0.7

    # Convert to 1-10 scale
    score = max(1, min(10, int(combined_score * 9) + 1))

    return score


# Function to find best category match
def find_best_category(summary, resolution, categories_df):
    """Find best matching category and subcategory"""
    if categories_df is None or len(categories_df) == 0:
        return "Unknown", "Unknown"

    text = f"{summary} {resolution}".lower()

    best_match = None
    best_score = 0

    for _, row in categories_df.iterrows():
        category = str(row['Category']).lower()
        subcategory = str(row['SubCategory']).lower()

        # Calculate match score
        cat_words = set(re.findall(r'\w+', category))
        subcat_words = set(re.findall(r'\w+', subcategory))
        text_words = set(re.findall(r'\w+', text))

        # Score based on word overlap
        cat_overlap = len(cat_words.intersection(text_words))
        subcat_overlap = len(subcat_words.intersection(text_words))

        total_score = cat_overlap * 2 + subcat_overlap  # Weight category higher

        if total_score > best_score:
            best_score = total_score
            best_match = (row['Category'], row['SubCategory'])

    if best_match:
        return best_match
    else:
        # Return first category if no match found
        return categories_df.iloc[0]['Category'], categories_df.iloc[0]['SubCategory']


# Function to process with LLM
def process_with_llm(row, categories_list, api_key):
    """Process a single row using OpenAI API"""
    try:
        import openai
        openai.api_key = api_key

        categories_text = "\n".join([f"- {cat}: {subcat}" for cat, subcat in categories_list])

        prompt = f"""You are an expert at analyzing banking support tickets. 

Problem Summary: {row['SUMMARY_NOTE']}

Resolution Applied: {row['RESOLUTION_NOTE']}

Task: Provide the following in JSON format:
1. Score (1-10): How well does the resolution address the problem? Consider completeness, relevance, and appropriateness.
2. Category: Best matching category from the list below
3. SubCategory: Best matching subcategory paired with the category

Available Categories:
{categories_text}

Respond ONLY with valid JSON in this exact format:
{{"score": <number>, "category": "<category>", "subcategory": "<subcategory>"}}"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )

        result = response.choices[0].message.content.strip()

        # Parse JSON response
        import json
        data = json.loads(result)

        return data['score'], data['category'], data['subcategory']

    except Exception as e:
        st.warning(f"LLM failed, using fallback: {str(e)}")
        return None, None, None


# Function to process data
def process_data(df_data, df_categories, use_llm=False, api_key=None):
    """Process the data and add Score, Category, and SubCategory columns"""

    results = []

    # Prepare categories list for LLM
    categories_list = []
    if df_categories is not None:
        categories_list = list(zip(df_categories['Category'], df_categories['SubCategory']))

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df_data.iterrows():
        status_text.text(f"Processing row {idx + 1} of {len(df_data)}...")

        score = None
        category = None
        subcategory = None

        # Try LLM if enabled
        if use_llm and api_key:
            score, category, subcategory = process_with_llm(row, categories_list, api_key)

        # Fallback to local algorithm
        if score is None:
            score = calculate_similarity(row['SUMMARY_NOTE'], row['RESOLUTION_NOTE'])
            category, subcategory = find_best_category(
                row['SUMMARY_NOTE'],
                row['RESOLUTION_NOTE'],
                df_categories
            )

        results.append({
            'Score': score,
            'Category': category,
            'SubCategory': subcategory
        })

        progress_bar.progress((idx + 1) / len(df_data))

    status_text.text("Processing complete!")
    progress_bar.empty()
    status_text.empty()

    # Add results to dataframe
    results_df = pd.DataFrame(results)
    df_processed = pd.concat([df_data, results_df], axis=1)

    return df_processed


# Function to create downloadable Excel
def create_excel_download(df_data, df_categories):
    """Create Excel file with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_data.to_excel(writer, sheet_name='Data', index=False)
        if df_categories is not None:
            df_categories.to_excel(writer, sheet_name='Categories', index=False)

    output.seek(0)
    return output


# Main UI
st.markdown("---")

# File upload section
uploaded_file = st.file_uploader(
    "📁 Upload Excel File (with 'Data' and 'Categories' tabs)",
    type=['xlsx', 'xls'],
    help="Excel file should have two tabs: 'Data' (with LOB, SOR_CASE_NBR, SUMMARY_NOTE, RESOLUTION_NOTE) and 'Categories' (with Category, SubCategory)"
)

if uploaded_file is not None and not st.session_state.uploaded:
    try:
        # Read the Excel file
        df_data = pd.read_excel(uploaded_file, sheet_name='Data')
        df_categories = pd.read_excel(uploaded_file, sheet_name='Categories')

        # Validate required columns
        required_data_cols = ['LOB', 'SOR_CASE_NBR', 'SUMMARY_NOTE', 'RESOLUTION_NOTE']
        required_cat_cols = ['Category', 'SubCategory']

        missing_data_cols = [col for col in required_data_cols if col not in df_data.columns]
        missing_cat_cols = [col for col in required_cat_cols if col not in df_categories.columns]

        if missing_data_cols:
            st.error(f"❌ Missing columns in Data tab: {', '.join(missing_data_cols)}")
        elif missing_cat_cols:
            st.error(f"❌ Missing columns in Categories tab: {', '.join(missing_cat_cols)}")
        else:
            st.session_state.df_data = df_data
            st.session_state.df_categories = df_categories
            st.session_state.uploaded = True
            st.success(
                f"✅ File uploaded successfully! Found {len(df_data)} records and {len(df_categories)} categories.")

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Show data preview if uploaded
if st.session_state.uploaded:
    st.markdown("### 📋 Data Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Tab (first 5 rows)**")
        st.dataframe(st.session_state.df_data.head(), use_container_width=True)

    with col2:
        st.markdown("**Categories Tab**")
        st.dataframe(st.session_state.df_categories, use_container_width=True)

    st.markdown("---")

    # Generate button
    if st.button("🚀 Generate Scores & Categories", type="primary", use_container_width=True):
        with st.spinner("Processing data... This may take a moment."):

            # Check if LLM should be used
            use_llm_processing = use_llm and api_key and len(api_key) > 0

            if use_llm_processing:
                st.info("🤖 Using OpenAI LLM for analysis...")
            else:
                st.info("💻 Using local algorithm for analysis...")

            # Process the data
            processed_df = process_data(
                st.session_state.df_data,
                st.session_state.df_categories,
                use_llm=use_llm_processing,
                api_key=api_key
            )

            st.session_state.df_data = processed_df
            st.session_state.processed = True

            st.success("✅ Processing complete!")

    # Show processed data and download
    if st.session_state.processed:
        st.markdown("### 📊 Processed Results")
        st.dataframe(st.session_state.df_data, use_container_width=True)

        # Statistics
        st.markdown("### 📈 Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_score = st.session_state.df_data['Score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}/10")

        with col2:
            top_category = st.session_state.df_data['Category'].mode()[0] if len(
                st.session_state.df_data) > 0 else "N/A"
            st.metric("Most Common Category", top_category)

        with col3:
            total_records = len(st.session_state.df_data)
            st.metric("Total Records", total_records)

        # Download button
        st.markdown("---")
        excel_file = create_excel_download(st.session_state.df_data, st.session_state.df_categories)

        st.download_button(
            label="📥 Download Processed Excel File",
            data=excel_file,
            file_name="processed_resolutions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Resolution Scoring & Categorization System v1.0</p>
    <p style='font-size: 0.8em;'>Upload Excel → Generate Scores → Download Results</p>
</div>
""", unsafe_allow_html=True)