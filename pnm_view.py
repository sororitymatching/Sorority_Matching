import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Recruitment Registration",
    page_icon="🌸",
    layout="centered" # "Centered" looks more like a form/letter than "Wide"
)

# --- CUSTOM CSS & THEME ---
# This block injects custom HTML/CSS to override Streamlit's default look
def local_css():
    st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

        /* Background & Main Text */
        .stApp {
            background-color: #FAFAFA; /* clean white/grey background */
            font-family: 'Montserrat', sans-serif;
            color: #333333;
        }

        /* Headers */
        h1, h2, h3 {
            font-family: 'Playfair Display', serif !important;
            color: #B46B84; /* Dusty Rose / Sorority Pink */
            text-align: center;
        }
        
        /* Subheaders styling */
        .css-10trblm {
            font-family: 'Montserrat', sans-serif;
            color: #B46B84;
            font-weight: 600;
        }

        /* Input Fields */
        .stTextInput > div > div > input, .stSelectbox > div > div > div, .stNumberInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #E0E0E0;
            background-color: #FFFFFF;
        }

        /* Buttons */
        .stButton > button {
            background-color: #B46B84;
            color: white;
            border-radius: 25px;
            padding: 10px 25px;
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #965A6E; /* Darker shade on hover */
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Info Box Styling */
        .stAlert {
            background-color: #FFF0F5;
            border: 1px solid #B46B84;
            color: #B46B84;
        }
        
        /* Custom divider */
        hr {
            margin: 2em 0;
            border: 0;
            border-top: 1px solid #E0E0E0;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Google Sheets Connection Function ---
# (Logic Unchanged)
def get_google_sheet_connection():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
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
# (Logic Unchanged)
def find_pnm_by_email(sheet, email):
    try:
        all_values = sheet.get_all_values()
        EMAIL_COL_INDEX = 6 
        for i, row in enumerate(all_values):
            if len(row) > EMAIL_COL_INDEX and row[EMAIL_COL_INDEX].strip().lower() == email.strip().lower():
                return i + 1, row 
        return None, None
    except Exception as e:
        st.error(f"Search error: {e}")
        return None, None

# --- Main Form UI ---

# Optional: Add a banner image (Use a URL or local file)
# st.image("https://your-sorority-banner-url.com/banner.jpg", use_column_width=True)

st.title("Recruitment Interest Form")
st.markdown("<p style='text-align: center; color: #666;'>We are so excited to meet you! Please enter your Penn State Email to get started.</p>", unsafe_allow_html=True)
st.markdown("---")

# 1. THE LOOKUP TRIGGER
col_search, col_space = st.columns([2,1]) # Use columns to center or organize
with col_search:
    search_email = st.text_input("💌 Enter your PSU Email (e.g., xyz123@psu.edu):").strip().lower()

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
        st.info(f"✨ Welcome back! We found your record. You are editing existing data.")
        
        try:
            # Mapping logic preserved exactly as requested
            if len(row_vals) > 1: defaults["name"] = row_vals[1]
            if len(row_vals) > 2: defaults["major"] = row_vals[2]
            if len(row_vals) > 3: defaults["minor"] = row_vals[3]
            if len(row_vals) > 4: defaults["hometown"] = row_vals[4]
            if len(row_vals) > 5: defaults["year"] = row_vals[5]
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
            pass 

# 4. THE FORM
if search_email: # Only show form if email is entered
    with st.form(key='pnm_form'):
        
        # Section 1: Personal Info
        st.markdown("### 🎀 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value=defaults["name"], placeholder="Jane Doe")
            major = st.text_input('Major (or "Undecided")', value=defaults["major"])
            minor = st.text_input("Minor", value=defaults["minor"])
            hometown = st.text_input("Hometown (City, State)", value=defaults["hometown"])
            
        with col2:
            yr_opts = ["Freshman", "Sophomore", "Junior", "Senior"]
            yr_idx = yr_opts.index(defaults["year"]) if defaults["year"] in yr_opts else 0
            year = st.selectbox("Year in School", yr_opts, index=yr_idx)
            
            st.text_input("Email (Locked)", value=search_email, disabled=True, help="To change email, refresh the page.")

        st.markdown("---")

        # Section 2: Academic History
        st.markdown("### 📚 Academic History")
        col3, col4 = st.columns(2)
        
        with col3:
            hs_name = st.text_input("High School Name", value=defaults["hs_name"])
            hs_gpa = st.number_input("High School GPA", min_value=0.0, max_value=5.0, step=0.01, value=defaults["hs_gpa"])
            college_gpa = st.number_input("College GPA", min_value=0.0, max_value=4.0, step=0.01, value=defaults["college_gpa"])
            
        with col4:
            hon_opts = ["No", "Yes"]
            hon_idx = hon_opts.index(defaults["honors"]) if defaults["honors"] in hon_opts else 0
            honors = st.selectbox("Schreyer / Honors College?", hon_opts, index=hon_idx)
            
            credits_completed = st.number_input("PSU Credits Completed", min_value=0, step=1, value=defaults["credits"])

        st.markdown("---")

        # Section 3: Involvement & Address
        st.markdown("### 🏠 Involvement & Living")
        
        hs_involvement = st.text_area("High School Involvement", value=defaults["hs_inv"], height=100)
        college_involvement = st.text_area("College Involvement", value=defaults["college_inv"], height=100)
        
        col5, col6 = st.columns(2)
        with col5:
            res_hall = st.text_input('Residence Hall (or "Off Campus")', value=defaults["res_hall"])
            
            loc_opts = ["East", "North", "South", "West", "Pollock", "Off Campus"]
            loc_idx = loc_opts.index(defaults["res_loc"]) if defaults["res_loc"] in loc_opts else 0
            res_location = st.selectbox('Res Hall Location', loc_opts, index=loc_idx)
            
            campus_address = st.text_input("Campus Address", value=defaults["campus_addr"])

        with col6:
            home_address = st.text_input("Home/Mailing Address", value=defaults["home_addr"])
            
            rush_opts = ["No", "Yes"]
            rushed_before = st.selectbox("Have you rushed before?", rush_opts, index=rush_opts.index(defaults["rushed"]) if defaults["rushed"] in rush_opts else 0)
            
            all_opts = ["No", "Yes"]
            allergies = st.selectbox("Any food allergies?", all_opts, index=all_opts.index(defaults["allergies"]) if defaults["allergies"] in all_opts else 0)

        st.markdown("---")

        # Section 4: Additional Info
        st.markdown("### 💖 Getting to Know You")
        
        col7, col8 = st.columns(2)
        with col7:
            hear_about = st.text_input("How did you hear about us?", value=defaults["hear_about"])
            video_link = st.text_input("Intro Video Link (Optional)", value=defaults["video"])
            
        with col8:
            hobbies = st.text_area("Hobbies & Interests", value=defaults["hobbies"], height=100)

        st.markdown("<br>", unsafe_allow_html=True)
        # Submit Button
        submit_button = st.form_submit_button(label='💖 Submit Information')

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
                    
                    if existing_pnm_id:
                        final_id = existing_pnm_id
                    else:
                        existing_data = sheet.get_all_values()
                        final_id = len(existing_data) if existing_data else 1
                    
                    row_data = [
                        timestamp, name, major, minor, hometown, year, search_email,
                        hs_name, hs_gpa, college_gpa, honors, hs_involvement,
                        college_involvement, res_hall, res_location, credits_completed,
                        rushed_before, allergies, home_address, campus_address,
                        hear_about, video_link, hobbies, final_id, "" 
                    ]
                    
                    if existing_row_index:
                        sheet.update(f"A{existing_row_index}:Y{existing_row_index}", [row_data])
                        st.balloons()
                        st.success(f"✅ Information UPDATED for {name}!")
                    else:
                        sheet.append_row(row_data)
                        st.balloons()
                        st.success(f"✅ Welcome, {name}! Your information has been registered.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
