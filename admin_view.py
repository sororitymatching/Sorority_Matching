import streamlit as st
import gspread
import pandas as pd

# --- CONFIGURATION ---
SHEET_NAME = "Party Excuses (Test)"
ADMIN_PASSWORD = "password" 

# Define scopes to ensure we have permission to write to the sheet
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# --- AUTHENTICATION FUNCTIONS ---

def get_data_from_sheet():
    """Connects to Google Sheets and returns a dataframe."""
    try:
        # Connect using the secrets and explicit scopes
        creds_dict = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(creds_dict, scopes=SCOPES)
        
        # Open the spreadsheet
        sheet = gc.open(SHEET_NAME).sheet1
        
        # Get ALL values (raw list of lists)
        data = sheet.get_all_values()
        
        # Check if data exists
        if not data:
            return pd.DataFrame()

        # Convert to Pandas DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return pd.DataFrame()

def update_party_count(new_count):
    """Updates the number of parties in the Config tab."""
    try:
        # Connect using the secrets and explicit scopes
        creds_dict = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(creds_dict, scopes=SCOPES)
        
        # Open the specific 'Config' tab
        # Note: You MUST create a tab named 'Config' in your sheet for this to work
        config_sheet = gc.open(SHEET_NAME).worksheet("Config")
        
        # Update cell B1
        config_sheet.update_acell('B1', new_count)
        return True
    except Exception as e:
        st.error(f"Error updating config: {e}")
        return False

def update_roster(names_list):
    """Updates the roster in the Config tab, Column D."""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(creds_dict, scopes=SCOPES)
        config_sheet = gc.open(SHEET_NAME).worksheet("Config")
        
        # 1. Clear existing names in Column D (from row 2 down)
        # We assume D1 is the header "Roster"
        config_sheet.batch_clear(["D2:D1000"])
        
        # 2. Format data for upload (must be a list of lists: [['Name 1'], ['Name 2']])
        # We sort them alphabetically for you automatically
        names_list.sort()
        formatted_names = [[name] for name in names_list if name.strip()]
        
        # 3. Update Column D starting at D2
        if formatted_names:
            config_sheet.update(range_name='D2', values=formatted_names)
            
        return True
    except Exception as e:
        st.error(f"Error updating roster: {e}")
        return False

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
    # --- IF LOGGED IN ---
    
    # 1. Header and Refresh Button
    col1, col2 = st.columns([3,1])
    with col1:
        st.success("Logged in as Admin")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    # 2. SETTINGS SECTION (New Feature)
    st.markdown("### ⚙️ Event Settings")
    with st.form("settings_form"):
        st.markdown("Set the number of parties users can choose from.")
        
        new_party_count = st.number_input("Number of Parties (Rounds):", min_value=1, max_value=100, value=1)
        save_settings = st.form_submit_button("Update Party Count")
        
        if save_settings:
            if update_party_count(new_party_count):
                st.toast(f"✅ Updated! Users will now see {new_party_count} parties.")
            
    st.divider() # distinct line to separate settings from data

    # 3. Load and Display Data
    df = get_data_from_sheet()

    if not df.empty:
        # SEARCH FILTER
        search_term = st.text_input("🔍 Search by Name:")
        
        if search_term:
            # We use 'Choose your name:' because that matches your CSV header exactly
            col_name = "Choose your name:"
            if col_name in df.columns:
                df = df[df[col_name].str.contains(search_term, case=False, na=False)]
            else:
                st.warning(f"Column '{col_name}' not found. Showing all data.")

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
