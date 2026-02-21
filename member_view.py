import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- CONFIGURATION ---
SHEET_NAME = "OverallMatchingInformation"

# --- AUTHENTICATION & CONNECTION (Fixed Caching) ---

@st.cache_resource
def get_gspread_client():
    """
    Cached resource to prevent re-authenticating on every run.
    This prevents Google API Quota errors.
    """
    try:
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        # Load credentials from secrets
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"❌ API Connection Error: {e}")
        return None

def get_sheet(worksheet_name):
    client = get_gspread_client()
    if not client: return None
    try:
        return client.open(SHEET_NAME).worksheet(worksheet_name)
    except Exception as e:
        st.error(f"❌ Worksheet '{worksheet_name}' not found. Please create it in Google Sheets.")
        return None

# --- DATA FETCHING ---

@st.cache_data(ttl=60) 
def get_party_options():
    try:
        sheet = get_sheet("Config")
        if not sheet: return ["Party 1", "Party 2", "Party 3", "Party 4"]
        
        # Safely get B1
        val = sheet.acell('B1').value
        if val and val.isdigit():
            num_parties = int(val)
        else:
            num_parties = 4 # Default
            
        options = [f"Party {i+1}" for i in range(num_parties)]
        return options
    except:
        return ["Party 1", "Party 2", "Party 3", "Party 4"]

@st.cache_data(ttl=300)
def get_roster():
    try:
        sheet = get_sheet("Config")
        if not sheet: return []
        
        names = sheet.col_values(4) # Column D
        # Remove header if it exists
        if names and "Roster" in names[0]: 
            names = names[1:]
            
        return sorted([n for n in names if n.strip()])
    except:
        return []

# --- MAIN APP LAYOUT ---
st.title("Sorority Recruitment Portal")

# Load data
roster = get_roster()
party_options = get_party_options()

if not roster:
    st.warning("⚠️ Roster is empty. Please ask an Admin to upload the roster in the settings tab.")

tab1, tab2 = st.tabs(["Create Bump Team", "Party Excuses"])

# ==========================
# TAB 1: BUMP TEAM CREATION
# ==========================
with tab1:
    st.header("Bump Team Creation")
    st.markdown("Select your partners and RGL to form a bump team.")

    with st.form(key='bump_form'):
        # 1. Creator Name
        creator_name = st.selectbox("Choose your name:", [""] + roster, key="bump_creator")
        
        # 2. Partners
        partner_options = [n for n in roster if n != creator_name] if creator_name else roster
        partners = st.multiselect("Choose your bump partner(s):", partner_options)
        
        # 3. RGL
        rgl = st.selectbox("Choose your Bump Group Leader (RGL):", ["None"] + roster)

        submit_bump = st.form_submit_button(label='Submit Bump Team')

    if submit_bump:
        if not creator_name or not partners:
            st.error("⚠️ Please fill in your name and at least one partner.")
        else:
            sheet = get_sheet("Bump Teams")
            if sheet:
                try:
                    # Calculate ID based on existing rows
                    # Assuming Row 1 is headers. If sheet is empty, ID is 1.
                    existing_data = sheet.get_all_values()
                    next_id = len(existing_data) if existing_data else 1
                    
                    # If headers exist (len > 0), the first data row is ID 1
                    # If headers don't exist, we might overwrite. 
                    # Safer to just use len(existing_data) as the ID (assuming header row exists)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    partners_str = ", ".join(partners)
                    rgl_val = "" if rgl == "None" else rgl
                    
                    # Columns: [Timestamp, Creator, Partners, RGL, ID, Ranking]
                    # Admin panel expects ID at index 4 (Column E) and Ranking at index 5 (Column F)
                    row_data = [timestamp, creator_name, partners_str, rgl_val, next_id, ""] 
                    
                    sheet.append_row(row_data)
                    
                    st.success(f"✅ Bump Team #{next_id} submitted successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving data: {e}")

# ==========================
# TAB 2: PARTY EXCUSES
# ==========================
with tab2:
    st.header("Recruitment Party Excuse Form")
    st.markdown("Please fill out this form if you cannot attend a specific party.")

    with st.form(key='excuse_form'):
        name = st.selectbox("Choose your name:", [""] + roster, key="excuse_name")
        parties = st.multiselect("Choose the party/parties you are unable to attend:", party_options)
        submit_excuse = st.form_submit_button(label='Submit Excuse')

    if submit_excuse:
        if not name or not parties:
            st.warning("⚠️ Please fill in both your name and the parties.")
        else:
            sheet = get_sheet("Party Excuses")
            if sheet:
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    parties_str = ", ".join(parties)
                    sheet.append_row([timestamp, name, parties_str])
                    st.success(f"✅ Excuse recorded for {name}!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving excuse: {e}")
