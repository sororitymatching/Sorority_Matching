import streamlit as st
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURATION ---
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
SHEET_NAME = "Party Excuses for Recruitment - Sorority Members"
# 🔒 SET A PASSWORD FOR ACCESS
ADMIN_PASSWORD = "password" 

# --- AUTHENTICATION (Reused from your main app) ---
def get_data_from_sheet():
    """Connects to Google Sheets and returns a dataframe."""
    try:
        # Load credentials from the SAME secrets file
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        client = gspread.authorize(creds)
        
        # Open the spreadsheet and get all values
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_records()
        
        # Convert to a Pandas DataFrame for easy viewing
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return pd.DataFrame()

# --- MAIN APP LAYOUT ---
st.set_page_config(page_title="Admin Dashboard", layout="wide")

st.title("📊 Recruitment Excuses Dashboard")

# 🔒 PASSWORD PROTECTION
# This checks if the password has been entered correctly
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password_input = st.text_input("Enter Admin Password:", type="password")
    if password_input == ADMIN_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password_input:
        st.error("Incorrect password.")
else:
    # --- IF LOGGED IN, SHOW DATA ---
    st.success("Logged in as Admin")
    
    if st.button("Refresh Data"):
        st.rerun()

    # Load data
    df = get_data_from_sheet()

    if not df.empty:
        # SEARCH FILTER
        search_term = st.text_input("🔍 Search by Name:")
        
        if search_term:
            # Filter the dataframe based on the name column
            # Note: We use the column name exactly as it appears in the Sheet
            df = df[df["Choose your name:"].str.contains(search_term, case=False, na=False)]

        # METRICS
        total_excuses = len(df)
        st.metric("Total Excuses Submitted", total_excuses)

        # DISPLAY THE TABLE
        st.dataframe(
            df, 
            use_container_width=True,
            hide_index=True
        )

        # DOWNLOAD BUTTON
        # Allows you to download the current view as a CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='recruitment_excuses.csv',
            mime='text/csv',
        )
    else:
        st.info("No excuses have been submitted yet.")
