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
st.title("🎉 Recruitment Party Excuse Form")
st.markdown("Please fill out this form if you cannot attend a specific party.")

# Create the form
with st.form(key='excuse_form'):
    # 1. Get list of members from Google Sheets
    roster = get_roster()
    
    # 2. Input: Name (Dropdown)
    name = st.selectbox("Choose your name:", roster)

    # 3. Input: Party Selection (Dynamic)
    # This fetches the number of parties from your Google Sheet 'Config' tab
    party_options = get_party_options() 
    parties = st.multiselect("Choose the party/parties you are unable to attend:", party_options)
    
    # Submit Button
    submit_button = st.form_submit_button(label='Submit Excuse')

# --- SUBMISSION LOGIC ---
if submit_button:
    # Validation: Ensure name is not empty ("") and parties are selected
    if name == "" or not parties:
        st.warning("⚠️ Please select your name and the parties you are missing.")
    else:
        sheet = get_google_sheet()
        if sheet:
            # Format the data exactly like your CSV example
            # [Timestamp, Name, Parties]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parties_str = ", ".join(parties) # Combines multiple selections into one string
            
            row_data = [timestamp, name, parties_str]
            
            # Append the row to Google Sheets
            sheet.append_row(row_data)
            
            st.success(f"✅ Thank you, {name}! Your excuse for {parties_str} has been recorded.")
            st.balloons()
