# 1. IMPORTS & CONFIGURATION
import streamlit as st
import pandas as pd
import openpyxl
from io import BytesIO
import re
from difflib import SequenceMatcher

# 2. SESSION STATE MANAGEMENT
st.session_state.uploaded       # File upload status
st.session_state.df_data        # Data DataFrame
st.session_state.df_categories  # Categories DataFrame
st.session_state.processed      # Processing status

# 3. SCORING ENGINE
calculate_similarity()          # Local algorithm (returns score + rationale)
process_with_llm()             # LLM integration (returns score + rationale)

# 4. CATEGORIZATION ENGINE
find_best_category()           # Category matching

# 5. BATCH PROCESSOR
process_data()                 # Main processing loop

# 6. EXPORT ENGINE
create_excel_download()        # Excel generation

# 7. UI COMPONENTS
- File uploader
- Data preview tables
- Generate button
- Progress bar
- Statistics metrics
- Download button
