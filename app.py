import streamlit as st
import gspread
from datetime import datetime

# --- CONFIGURATION ---
# Replace this with your actual Google Sheet name
SHEET_NAME = "OverallMatchingInformation"

# Define scopes (permissions) for the connection
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# --- AUTHENTICATION HELPERS ---

def get_connection():
    """Establishes the connection to Google Sheets."""
    # Load credentials from Streamlit secrets and convert to dict
    creds_dict = dict(st.secrets["gcp_service_account"])
    # Connect using modern gspread method with explicit scopes
    return gspread.service_account_from_dict(creds_dict, scopes=SCOPES)

def get_google_sheet():
    """Connects to the specific sheet for saving data."""
    try:
        gc = get_connection()
        # Open the spreadsheet and get the first tab (for responses)
        return gc.open(SHEET_NAME).sheet1
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

# --- FUNCTION TO GET CONFIG ---
# We use @st.cache_data so it doesn't call Google Sheets every single second
@st.cache_data(ttl=60) 
def get_party_options():
    try:
        gc = get_connection()
        
        # Open Config tab
        config_sheet = gc.open(SHEET_NAME).worksheet("Config")
        
        # Get value from B1
        num_parties = int(config_sheet.acell('B1').value)
        
        # Generate the list dynamically
        # If num_parties is 3, this creates: ["Party 1", "Party 2", "Party 3"]
        options = [f"Party {i+1}" for i in range(num_parties)]
        
        return options
    except Exception as e:
        # Fallback if connection fails (defaults to 4 parties)
        return ["Party 1", "Party 2", "Party 3", "Party 4"]

@st.cache_data(ttl=300) # Cache for 5 minutes so it's fast
def get_roster():
    try:
        gc = get_connection()
        config_sheet = gc.open(SHEET_NAME).worksheet("Config")
        
        # Get all values from Column D (assuming D1 is header, so start at D2)
        # get_col(4) gets the 4th column (D)
        names = config_sheet.col_values(4)
        
        # Remove the header if it exists (e.g., if D1 says "Roster")
        if names and names[0] == "Roster":
            names = names[1:]
            
        # Add the empty option at the start
        final_list = [""] + [n for n in names if n.strip()]
        return final_list
        
    except Exception as e:
        # Fallback to a small list if connection fails, so app doesn't crash
        return ["", "Error loading roster..."]

# --- MAIN APP LAYOUT ---
# --- MAIN APP LAYOUT ---
st.title("Sorority Recruitment Portal")
tab1, tab2 = st.tabs(["Create Bump Team", "Party Excuses"])

with tab1:
    st.header("Bump Team Creation")
    st.markdown("Select your partners and RGL to form a bump team.")

    with st.form(key='bump_form'):
        # 1. Creator Name
        creator_name = st.selectbox("Choose your name:", [""] + roster, key="bump_creator")
        
        # 2. Partners
        partner_options = roster if not creator_name else [n for n in roster if n != creator_name]
        partners = st.multiselect("Choose your bump partner(s):", partner_options)
        
        # 3. RGL
        rgl = st.selectbox("Choose your Bump Group Leader (RGL):", ["None"] + roster)
        
        # 4. Details
        # We removed Team ID input here because it is now automatic
        ranking = st.number_input("Bump Team Ranking:", min_value=1, value=1)

        submit_bump = st.form_submit_button(label='Submit Bump Team')

    if submit_bump:
        if not creator_name or not partners:
            st.error("⚠️ Please fill in your name and at least one partner.")
        else:
            sheet = get_sheet("Bump Teams")
            if sheet:
                # --- AUTOMATIC TEAM ID LOGIC ---
                # 1. Get all current values in the sheet
                all_records = sheet.get_all_values()
                
                # 2. Calculate the next ID
                # If the sheet only has a header (length 1), the first team is 1.
                # If the sheet has 5 teams (length 6), the next team is 6.
                next_id = len(all_records) 
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                partners_str = ", ".join(partners)
                rgl_val = "" if rgl == "None" else rgl
                
                # Columns: Timestamp, Creator, Partners, RGL, Team ID, Ranking
                row_data = [timestamp, creator_name, partners_str, rgl_val, next_id, ranking]
                
                sheet.append_row(row_data)
                st.success(f"✅ Bump Team #{next_id} created successfully!")
                st.balloons()

with tab1:
    st.header("Recruitment Party Excuse Form")
    st.markdown("Please fill out this form if you cannot attend a specific party.")

    with st.form(key='excuse_form'):
        roster = get_roster()
        name = st.selectbox("Choose your name:", [""] + roster)
        
        party_options = get_party_options() 
        parties = st.multiselect("Choose the party/parties you are unable to attend:", party_options)
        
        submit_excuse = st.form_submit_button(label='Submit Excuse')

    if submit_excuse:
        if not name or not parties:
            st.warning("⚠️ Please fill in both your name and the parties.")
        else:
            sheet = get_sheet("Party Excuses") # Assuming excuses go to the first sheet
            if sheet:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                parties_str = ", ".join(parties)
                sheet.append_row([timestamp, name, parties_str])
                st.success(f"✅ Excuse recorded for {name}!")
                st.balloons()
