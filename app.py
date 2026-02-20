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
st.title("Recruitment Party Excuse Form")
st.markdown("Please fill out this form if you cannot attend a specific party.")

# Create the form
with st.form(key='excuse_form'):
    # 1. Create your list of members
    # (You can copy/paste your full roster here)
    # 1. Create your list of members
    roster = [
        "",
        "Hailey Abbott", "Sophie Adams", "Caroline Allen", "Emily Anderson", "Jessica Armstrong",
        "Hannah Bailey", "Olivia Baker", "Riley Barnes", "Lucy Bell", "Maya Bennett",
        "Chloe Black", "Savannah Brooks", "Charlotte Brown", "Morgan Bryant", "Jordan Butler",
        "Emma Campbell", "Avery Carter", "Mia Clark", "Taylor Coleman", "Harper Collins",
        "Zoe Cook", "Ella Cooper", "Payton Cox", "Sofia Cruz", "Madison Davis",
        "Camila Diaz", "Samantha Edwards", "Lily Evans", "Ruby Fisher", "Elena Flores",
        "Audrey Foster", "Isabella Garcia", "Kendall Gibson", "Ariana Gomez", "Reese Graham",
        "Piper Gray", "Abigail Green", "Quinn Griffin", "Grace Hall", "Sydney Hamilton",
        "Elizabeth Harris", "Layla Harrison", "Kennedy Hayes", "Alexa Henderson", "Victoria Hernandez",
        "Scarlett Hill", "Stella Howard", "Violet Hughes", "Avery Jackson", "Clara James",
        "Allison Jenkins", "Sophia Johnson", "Ava Jones", "Brooklyn Kelly", "Nora King",
        "Aria Lee", "Addison Lewis", "Paige Long", "Gianna Lopez", "Eleanor Martin",
        "Luna Martinez", "Kinsley Mason", "Bailey Matthews", "Sadie McDonald", "Evelyn Miller",
        "Penelopy Mitchell", "Amelia Moore", "Mackenzie Morgan", "Hazel Morris", "Lillian Murphy",
        "Reagan Myers", "Willow Nelson", "Christine Nguyen", "Valeria Ortiz", "Skylar Parker",
        "Priya Patel", "Cadence Patterson", "Gabriella Perez", "Jasmine Perry", "Aubrey Peterson",
        "Madelyn Phillips", "Vivian Powell", "Hayden Price", "Natalia Ramirez", "Lydia Reed",
        "Andrea Reyes", "Brooke Reynolds", "Katherine Richardson", "Ximena Rivera", "Anna Roberts",
        "Mila Robinson", "Camila Rodriguez", "Bella Rogers", "Gabrielle Ross", "Summer Russell",
        "Sara Sanchez", "Faith Sanders", "Aurora Scott", "Reagan Simmons", "Olivia Smith",
        "Presely Snyder", "Natalie Stewart", "Courtney Sullivan", "Layla Taylor", "Zoey Thomas",
        "Leah Thompson", "Valentina Torres", "Ellie Turner", "Harper Walker", "Lauren Ward",
        "Alexis Washington", "Ashley Watson", "Emerson Webb", "Hadley Wells", "Jade West",
        "Addison White", "Emma Williams", "Abigail Wilson", "Emery Wood", "Delaney Woods",
        "Riley Wright", "Alice Young", "Tatum Zimmer", "Sloan Ackerman", "Teagan Brady",
        "Rory Callahan", "Fiona Doherty", "Blair Ellis", "Maeve Gallagher", "Keira Harrington"
    ]
    
    # 2. Input: Name (Dropdown)
    name = st.selectbox("Choose your name:", roster)

    # Input: Party Selection (Multi-select allows picking multiple parties)
    party_options = ["Party 1", "Party 2", "Party 3", "Party 4", "Party 5", "Party 6", "Party 7", "Party 8", "Party 9", "Party 10", "Party 11", "Party 12", "Party 13", "Party 14", "Party 15", "Party 16", "Party 17", "Party 18", "Party 19", "Party 20", "Party 21", "Party 22", "Party 23", "Party 24", "Party 25", "Party 26", "Party 27", "Party 28", "Party 29", "Party 30", "Party 31", "Party 32", "Party 33", "Party 34", "Party 35", "Party 36", "Party 37"]
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
