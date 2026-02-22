import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="PNM Recruitment Form", layout="wide")

# --- Google Sheets Connection Function ---
def get_google_sheet_connection():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
        # Load credentials from Streamlit secrets
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scope
        )
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

def get_worksheet():
    client = get_google_sheet_connection()
    if client:
        try:
            return client.open("OverallMatchingInformation").worksheet("PNM Information")
        except Exception as e:
            st.error(f"Error opening worksheet: {e}")
            return None
    return None

# --- Helper: Find Row by Email ---
def find_pnm_by_email(sheet, email):
    """
    Searches for an email in Column 7 (G) and returns the row index and values.
    """
    try:
        # Get all values to search locally (faster)
        all_values = sheet.get_all_values()
        
        # Email is the 7th item in the list (index 6) based on the form structure below
        EMAIL_COL_INDEX = 6 
        
        for i, row in enumerate(all_values):
            # Check if row has enough columns and matches email (case-insensitive)
            if len(row) > EMAIL_COL_INDEX and row[EMAIL_COL_INDEX].strip().lower() == email.strip().lower():
                return i + 1, row  # Return 1-based row index and the data
        return None, None
    except Exception as e:
        st.error(f"Search error: {e}")
        return None, None

# --- Main Form UI ---
st.title("Initial Recruitment Interest Form - PNM")
st.markdown("Please enter your **Penn State Email** to begin. If you are already in our system, your information will load automatically.")

# 1. THE LOOKUP TRIGGER
# We move Email to the top so we can use it to fetch data before rendering the form
search_email = st.text_input("Enter your PSU Email (e.g., xyz123@psu.edu):").strip().lower()

# 2. INITIALIZE DEFAULTS
defaults = {
    "name": "", "major": "", "minor": "", "hometown": "",
    "year": "Freshman", "hs_name": "", "hs_gpa": 0.0, "college_gpa": 0.0,
    "honors": "No", "credits": 0, "hs_inv": "", "college_inv": "",
    "res_hall": "", "res_loc": "East", "campus_addr": "", "home_addr": "",
    "rushed": "No", "allergies": "No", "hear_about": "", "video": "", "hobbies": ""
}

existing_row_index = None
existing_pnm_id = None
sheet = get_worksheet()

# 3. FETCH EXISTING DATA
if search_email and sheet:
    found_idx, row_vals = find_pnm_by_email(sheet, search_email)
    if found_idx:
        existing_row_index = found_idx
        st.info(f"Welcome back! We found your record. You are editing existing data.")
        
        # MAP ROW VALUES TO DEFAULTS
        # Row Structure based on save logic:
        # [0:Timestamp, 1:Name, 2:Major, 3:Minor, 4:Hometown, 5:Year, 6:Email, 
        #  7:HSName, 8:HSGPA, 9:ColGPA, 10:Honors, 11:HSInv, 12:ColInv, 13:ResHall, 
        #  14:ResLoc, 15:Credits, 16:Rushed, 17:Allergies, 18:HomeAddr, 19:CampAddr, 
        #  20:Hear, 21:Video, 22:Hobbies, 23:ID, 24:Rank]
        
        try:
            if len(row_vals) > 1: defaults["name"] = row_vals[1]
            if len(row_vals) > 2: defaults["major"] = row_vals[2]
            if len(row_vals) > 3: defaults["minor"] = row_vals[3]
            if len(row_vals) > 4: defaults["hometown"] = row_vals[4]
            if len(row_vals) > 5: defaults["year"] = row_vals[5]
            # Email is at 6
            if len(row_vals) > 7: defaults["hs_name"] = row_vals[7]
            if len(row_vals) > 8: defaults["hs_gpa"] = float(row_vals[8]) if row_vals[8] else 0.0
            if len(row_vals) > 9: defaults["college_gpa"] = float(row_vals[9]) if row_vals[9] else 0.0
            if len(row_vals) > 10: defaults["honors"] = row_vals[10]
            if len(row_vals) > 11: defaults["hs_inv"] = row_vals[11]
            if len(row_vals) > 12: defaults["college_inv"] = row_vals[12]
            if len(row_vals) > 13: defaults["res_hall"] = row_vals[13]
            if len(row_vals) > 14: defaults["res_loc"] = row_vals[14]
            if len(row_vals) > 15: defaults["credits"] = int(row_vals[15]) if row_vals[15] else 0
            if len(row_vals) > 16: defaults["rushed"] = row_vals[16]
            if len(row_vals) > 17: defaults["allergies"] = row_vals[17]
            if len(row_vals) > 18: defaults["home_addr"] = row_vals[18]
            if len(row_vals) > 19: defaults["campus_addr"] = row_vals[19]
            if len(row_vals) > 20: defaults["hear_about"] = row_vals[20]
            if len(row_vals) > 21: defaults["video"] = row_vals[21]
            if len(row_vals) > 22: defaults["hobbies"] = row_vals[22]
            if len(row_vals) > 23: existing_pnm_id = row_vals[23]
        except ValueError:
            pass # Handle cases where conversion to float/int fails gracefully

# 4. THE FORM
with st.form(key='pnm_form'):
    
    # Section 1: Personal Info
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Enter your name:", value=defaults["name"])
        major = st.text_input('Enter your major or "Undecided":', value=defaults["major"])
        minor = st.text_input("Enter your minor or leave blank:", value=defaults["minor"])
        hometown = st.text_input("Enter your hometown (City, State):", value=defaults["hometown"])
        
    with col2:
        # Ensure default is in list
        yr_opts = ["Freshman", "Sophomore", "Junior", "Senior"]
        yr_idx = yr_opts.index(defaults["year"]) if defaults["year"] in yr_opts else 0
        year = st.selectbox("Pick your year in school:", yr_opts, index=yr_idx)
        
        # Display Email as disabled or editable (using the search_email value)
        st.text_input("Your Email:", value=search_email, disabled=True)

    st.markdown("---")

    # Section 2: Academic History
    st.subheader("Academic History")
    col3, col4 = st.columns(2)
    
    with col3:
        hs_name = st.text_input("Enter the name of your high school:", value=defaults["hs_name"])
        hs_gpa = st.number_input("Enter your high school GPA:", min_value=0.0, max_value=5.0, step=0.01, value=defaults["hs_gpa"])
        college_gpa = st.number_input("Enter your current college GPA:", min_value=0.0, max_value=4.0, step=0.01, value=defaults["college_gpa"])
        
    with col4:
        hon_opts = ["No", "Yes"]
        hon_idx = hon_opts.index(defaults["honors"]) if defaults["honors"] in hon_opts else 0
        honors = st.selectbox("Are you involved in honors at Penn State?", hon_opts, index=hon_idx)
        
        credits_completed = st.number_input("Penn State credit hours completed:", min_value=0, step=1, value=defaults["credits"])

    st.markdown("---")

    # Section 3: Involvement & Address
    st.subheader("Involvement & Living")
    
    hs_involvement = st.text_area("Enter your high school involvement:", value=defaults["hs_inv"])
    college_involvement = st.text_area("Enter your college involvement:", value=defaults["college_inv"])
    
    col5, col6 = st.columns(2)
    with col5:
        res_hall = st.text_input('Name of residence hall or "Off campus":', value=defaults["res_hall"])
        
        loc_opts = ["East", "North", "South", "West", "Pollock", "Off Campus"]
        loc_idx = loc_opts.index(defaults["res_loc"]) if defaults["res_loc"] in loc_opts else 0
        res_location = st.selectbox('Choose residence hall location:', loc_opts, index=loc_idx)
        
        campus_address = st.text_input("Enter your campus or off campus address:", value=defaults["campus_addr"])

    with col6:
        home_address = st.text_input("Enter your mailing or home address:", value=defaults["home_addr"])
        
        rush_opts = ["No", "Yes"]
        rushed_before = st.selectbox("Have you rushed before?", rush_opts, index=rush_opts.index(defaults["rushed"]) if defaults["rushed"] in rush_opts else 0)
        
        all_opts = ["No", "Yes"]
        allergies = st.selectbox("Do you have any allergies?", all_opts, index=all_opts.index(defaults["allergies"]) if defaults["allergies"] in all_opts else 0)

    st.markdown("---")

    # Section 4: Additional Info
    st.subheader("Getting to Know You")
    
    col7, col8 = st.columns(2)
    with col7:
        hear_about = st.text_input("How did you hear about us?", value=defaults["hear_about"])
        video_link = st.text_input("Enter your video link:", value=defaults["video"])
        
    with col8:
        hobbies = st.text_area("Enter your hobbies and interests:", value=defaults["hobbies"])

    # Submit Button
    submit_button = st.form_submit_button(label='Submit Information')

# --- Submission Logic ---
if submit_button:
    if not name or not search_email:
        st.error("Name and Email are required.")
    elif "@psu.edu" not in search_email:
        st.warning("Please ensure you use your @psu.edu email.")
    else:
        if sheet:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Determine ID (Keep existing or Generate new)
                if existing_pnm_id:
                    final_id = existing_pnm_id
                else:
                    # Logic for new ID: Count rows + 1 (Header is row 1)
                    # A safer way if IDs are strictly sequential:
                    existing_data = sheet.get_all_values()
                    final_id = len(existing_data) if existing_data else 1
                
                # Prepare row data (MUST MATCH ORDER IN DEFAULTS FETCH)
                row_data = [
                    timestamp,
                    name,
                    major,
                    minor,
                    hometown,
                    year,
                    search_email, # Using the search variable as the email
                    hs_name,
                    hs_gpa,
                    college_gpa,
                    honors,
                    hs_involvement,
                    college_involvement,
                    res_hall,
                    res_location,
                    credits_completed,
                    rushed_before,
                    allergies,
                    home_address,
                    campus_address,
                    hear_about,
                    video_link,
                    hobbies,
                    final_id,
                    "" # Placeholder for Rank
                ]
                
                if existing_row_index:
                    # --- UPDATE ---
                    # Update range A{row}:Y{row}
                    sheet.update(f"A{existing_row_index}:Y{existing_row_index}", [row_data])
                    st.success(f"✅ Information UPDATED for {name}! (PNM ID: {final_id})")
                else:
                    # --- APPEND ---
                    sheet.append_row(row_data)
                    st.success(f"✅ Welcome, {name}! Your information has been registered. (PNM ID: {final_id})")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
