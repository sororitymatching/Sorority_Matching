import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="PNM Recruitment Form", layout="wide")

# --- Google Sheets Connection Function ---
def get_google_sheet_connection():
    # Define the scope
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Load credentials from Streamlit secrets
    # This expects a secrets.toml file with a [gcp_service_account] section
    try:
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scope
        )
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

# --- Main Form UI ---
st.title("Initial Recruitment Interest Form - PNM")
st.markdown("Please fill out the information below to register your interest.")

with st.form(key='pnm_form'):
    
    # Section 1: Personal Info
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Enter your name:")
        major = st.text_input('Enter your major or "Undecided":')
        minor = st.text_input("Enter your minor or leave blank:")
        hometown = st.text_input("Enter your hometown (City, State):")
        
    with col2:
        year = st.selectbox("Pick your year in school:", 
                            ["Freshman", "Sophomore", "Junior", "Senior"])
        email = st.text_input("Enter your Penn State email (psu.edu):")
        # Basic validation visual
        if email and "@psu.edu" not in email:
            st.warning("Please ensure you use your @psu.edu email.")

    st.markdown("---")

    # Section 2: Academic History
    st.subheader("Academic History")
    col3, col4 = st.columns(2)
    
    with col3:
        hs_name = st.text_input("Enter the name of your high school:")
        hs_gpa = st.number_input("Enter your high school GPA:", min_value=0.0, max_value=5.0, step=0.01)
        college_gpa = st.number_input("Enter your current college GPA:", min_value=0.0, max_value=4.0, step=0.01)
        
    with col4:
        honors = st.selectbox("Are you involved in honors at Penn State (e.g. Schreyer)?", ["No", "Yes"])
        credits_completed = st.number_input("Penn State credit hours completed:", min_value=0, step=1)

    st.markdown("---")

    # Section 3: Involvement & Address
    st.subheader("Involvement & Living")
    
    hs_involvement = st.text_area("Enter your high school involvement (sports, clubs etc.), separate by comma:")
    college_involvement = st.text_area("Enter your college involvement (sports, clubs etc.), separate by comma:")
    
    col5, col6 = st.columns(2)
    with col5:
        res_hall = st.text_input('Enter the name of your residence hall or "Off campus":')
        res_location = st.selectbox('Choose the residence hall location or "Off Campus":', 
                                    ["East", "North", "South", "West", "Pollock", "Off Campus"])
        campus_address = st.text_input("Enter your campus or off campus address:")

    with col6:
        home_address = st.text_input("Enter your mailing or home address:")
        rushed_before = st.selectbox("Have you rushed before?", ["No", "Yes"])
        allergies = st.selectbox("Do you have any allergies?", ["No", "Yes"])

    st.markdown("---")

    # Section 4: Additional Info
    st.subheader("Getting to Know You")
    
    col7, col8 = st.columns(2)
    with col7:
        hear_about = st.text_input("How did you hear about us?")
        video_link = st.text_input("Enter your video link (optional):")
        
    with col8:
        hobbies = st.text_area("Enter your hobbies and interests, separate by comma:")

    # Submit Button
    submit_button = st.form_submit_button(label='Submit Information')

# --- Submission Logic ---
if submit_button:
    if not name or not email:
        st.error("Name and Email are required fields.")
    else:
        client = get_google_sheet_connection()
        
        if client:
            try:
                # Open the Google Sheet
                # Note: Ensure the file name matches exactly what is in your Drive
                sheet = client.open("OverallMatchingInformation").worksheet("PNM Information")
                
                # Prepare row data matching the CSV order
                # Note: We omit PNM ID and Average Recruit Rank as they are likely internal/calculated fields
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                row_data = [
                    timestamp,
                    name,
                    major,
                    minor,
                    hometown,
                    year,
                    email,
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
                    "", # Placeholder for PNM ID (Calculated internally)
                    ""  # Placeholder for Average Recruit Rank (Calculated internally)
                ]
                
                # Append to sheet
                sheet.append_row(row_data)
                
                st.success(f"Thank you, {name}! Your information has been recorded.")
                st.balloons()
                
            except gspread.exceptions.WorksheetNotFound:
                st.error("Could not find the worksheet 'PNM Information'. Please check the sheet name.")
            except gspread.exceptions.SpreadsheetNotFound:
                st.error("Could not find the file 'OverallMatchingInformation'. Please check the file name.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
