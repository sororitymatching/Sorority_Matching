import streamlit as st
import gspread
from datetime import datetime

# --- CONFIGURATION ---
# Replace this with your actual Google Sheet name
SHEET_NAME = "Party Excuses (Test)"

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
        
        # Always add Preference Round at the end
        options.append("Preference Round")
        
        return options
    except Exception as e:
        # Fallback if connection fails (defaults to 4 parties)
        return ["Party 1", "Party 2", "Party 3", "Party 4", "Preference Round"]

# --- MAIN APP LAYOUT ---
st.title("🎉 Recruitment Party Excuse Form")
st.markdown("Please fill out this form if you cannot attend a specific party.")

# Create the form
with st.form(key='excuse_form'):
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
