import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- CONFIGURATION ---
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# Replace this with your actual Google Sheet name
SHEET_NAME = "Party Excuses (Test)" 

# --- AUTHENTICATION FUNCTION ---
def get_google_sheet():
    """Connects to Google Sheets using secrets."""
    try:
        # Load credentials from Streamlit secrets
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        client = gspread.authorize(creds)
        
        # Open the spreadsheet
        sheet = client.open(SHEET_NAME).sheet1  # Opens the first tab
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

# --- MAIN APP LAYOUT ---
st.title("🎉 Recruitment Party Excuse Form")
st.markdown("Please fill out this form if you cannot attend a specific party.")

# Create the form
with st.form(key='excuse_form'):
    # Input: Name
    name = st.text_input("Choose your name:")

    # Input: Party Selection (Multi-select allows picking multiple parties)
    party_options = ["Party 1", "Party 2", "Party 3", "Party 4", "Preference Round"]
    parties = st.multiselect("Choose the party/parties you are unable to attend:", party_options)
    
    # Submit Button
    submit_button = st.form_submit_button(label='Submit Excuse')

# --- SUBMISSION LOGIC ---
if submit_button:
    if not name or not parties:
        st.warning("⚠️ Please fill in both your name and the parties you are missing.")
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
