import streamlit as st
import gspread
from datetime import datetime

# --- CONFIGURATION ---
SHEET_NAME = "OverallMatchingInformation"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# --- AUTHENTICATION HELPERS ---

def get_connection():
    creds_dict = dict(st.secrets["gcp_service_account"])
    return gspread.service_account_from_dict(creds_dict, scopes=SCOPES)

def get_sheet(worksheet_name):
    try:
        gc = get_connection()
        return gc.open(SHEET_NAME).worksheet(worksheet_name)
    except Exception as e:
        st.error(f"Error: Worksheet '{worksheet_name}' not found. {e}")
        return None

@st.cache_data(ttl=60) 
def get_party_options():
    try:
        gc = get_connection()
        config_sheet = gc.open(SHEET_NAME).worksheet("Config")
        num_parties = int(config_sheet.acell('B1').value)
        options = [f"Party {i+1}" for i in range(num_parties)]
        return options
    except:
        return ["Party 1", "Party 2", "Party 3", "Party 4"]

@st.cache_data(ttl=300)
def get_roster():
    try:
        gc = get_connection()
        config_sheet = gc.open(SHEET_NAME).worksheet("Config")
        names = config_sheet.col_values(4) # Column D
        if names and names[0] == "Roster":
            names = names[1:]
        return [n for n in names if n.strip()]
    except:
        return []

# --- MAIN APP LAYOUT ---
st.title("Sorority Recruitment Portal")

# FETCH DATA ONCE AT THE TOP
roster = get_roster()
party_options = get_party_options()

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
        
        # 2. Partners (Filter out the creator so they can't pick themselves)
        partner_options = [n for n in roster if n != creator_name] if creator_name else roster
        partners = st.multiselect("Choose your bump partner(s):", partner_options)
        
        # 3. RGL
        rgl = st.selectbox("Choose your Bump Group Leader (RGL):", ["None"] + roster)
        
        # 4. Details
        ranking = st.number_input("Bump Team Ranking:", min_value=1, value=1)

        submit_bump = st.form_submit_button(label='Submit Bump Team')

    if submit_bump:
        if not creator_name or not partners:
            st.error("⚠️ Please fill in your name and at least one partner.")
        else:
            sheet = get_sheet("Bump Teams")
            if sheet:
                all_records = sheet.get_all_values()
                next_id = len(all_records) # Auto-increment ID
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                partners_str = ", ".join(partners)
                rgl_val = "" if rgl == "None" else rgl
                
                row_data = [timestamp, creator_name, partners_str, rgl_val, next_id, ranking]
                sheet.append_row(row_data)
                
                st.success(f"✅ Bump Team #{next_id} created successfully!")
                st.balloons()

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
            sheet = get_sheet("Party Excuses") # Ensure this tab exists in Sheets!
            if sheet:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                parties_str = ", ".join(parties)
                sheet.append_row([timestamp, name, parties_str])
                st.success(f"✅ Excuse recorded for {name}!")
                st.balloons()
