import streamlit as st
import gspread
import pandas as pd

# --- CONFIGURATION ---
SHEET_NAME = "Party Excuses (Test)"
ADMIN_PASSWORD = "password" 

# --- AUTHENTICATION (Modern Method) ---
def get_data_from_sheet():
    """Connects to Google Sheets and returns a dataframe."""
    try:
        # 1. Connect using the modern gspread method (no oauth2client needed)
        # This automatically uses the scopes needed for Drive & Sheets
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        
        # 2. Open the spreadsheet
        sheet = gc.open(SHEET_NAME).sheet1
        
        # 3. Get ALL values (raw list of lists) instead of records
        # This prevents errors with duplicate headers or empty columns
        data = sheet.get_all_values()
        
        # 4. Check if data exists
        if not data:
            return pd.DataFrame()

        # 5. Convert to Pandas DataFrame
        # The first row (data[0]) is the header
        # The rest (data[1:]) is the actual data
        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return pd.DataFrame()

# --- MAIN APP LAYOUT ---
st.set_page_config(page_title="Admin Dashboard", layout="wide")

st.title("📊 Recruitment Excuses Dashboard")

# 🔒 PASSWORD PROTECTION
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
            # We use 'Choose your name:' because that matches your CSV header exactly
            # If the column name changes in the sheet, update it here!
            df = df[df["Choose your name:"].str.contains(search_term, case=False, na=False)]

        # METRICS
        st.metric("Total Excuses Submitted", len(df))

        # DISPLAY THE TABLE
        st.dataframe(
            df, 
            use_container_width=True,
            hide_index=True
        )

        # DOWNLOAD BUTTON
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='recruitment_excuses.csv',
            mime='text/csv',
        )
    else:
        st.info("No data found. The sheet might be empty or the connection failed.")
