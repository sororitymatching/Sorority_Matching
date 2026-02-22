import streamlit as st
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- CONFIGURATION ---
SHEET_NAME = "OverallMatchingInformation"

# --- AUTHENTICATION & CONNECTION ---
@st.cache_resource
def get_gspread_client():
    """
    Cached resource to prevent re-authenticating on every run.
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
        st.error(f"❌ Worksheet '{worksheet_name}' not found. Please create it in your Google Spreadsheet.")
        return None

# --- DATA FETCHING (Dynamic) ---

@st.cache_data(ttl=60) 
def get_party_options():
    try:
        sheet = get_sheet("Settings")
        if not sheet: return ["Party 1", "Party 2", "Party 3", "Party 4"]
        
        # Safely get B1 for number of parties
        val = sheet.acell('B1').value
        if val and val.isdigit():
            num_parties = int(val)
        else:
            num_parties = 4 
            
        options = [f"Party {i+1}" for i in range(num_parties)]
        return options
    except:
        return ["Party 1", "Party 2", "Party 3", "Party 4"]

@st.cache_data(ttl=300)
def get_roster():
    """
    Fetches the roster dynamically from the 'Member Information' sheet.
    """
    try:
        sheet = get_sheet("Member Information")
        if not sheet: return []
        
        # Get headers to find the correct column
        headers = sheet.row_values(1)
        name_col_index = 3 # Default to 3rd column
        
        # Try to find specific "Name" or "Full Name" header
        for i, header in enumerate(headers):
            h_lower = header.lower().strip()
            if h_lower == "full name" or h_lower == "name" or h_lower == "member name":
                name_col_index = i + 1 
                break
                
        names = sheet.col_values(name_col_index) 
        
        # Remove header if it exists
        if names:
            first_val = names[0].lower().strip()
            if "name" in first_val or "roster" in first_val:
                names = names[1:]
            
        return sorted([n for n in names if n.strip()])
    except Exception as e:
        print(f"Error fetching roster: {e}")
        return []

@st.cache_data(ttl=300)
def get_pnm_list():
    """
    Fetches the list of PNM names from the 'PNM Information' sheet, Column B.
    """
    try:
        sheet = get_sheet("PNM Information")
        if not sheet: return []
        
        pnm_names = sheet.col_values(2) 
        if pnm_names and ("Name" in pnm_names[0] or "Enter" in pnm_names[0]):
            pnm_names = pnm_names[1:]
            
        return sorted([n for n in pnm_names if n.strip()])
    except Exception as e:
        st.error(f"Error fetching PNMs: {e}")
        return []

@st.cache_data(ttl=300)
def get_pnm_dataframe():
    """
    Fetches all data from the 'PNM Information' sheet as a DataFrame.
    """
    try:
        sheet = get_sheet("PNM Information")
        if not sheet: return pd.DataFrame()
        
        data = sheet.get_all_values()
        if not data: return pd.DataFrame()
        
        # Assume first row is header
        return pd.DataFrame(data[1:], columns=data[0])
    except Exception as e:
        st.error(f"Error fetching PNM dataframe: {e}")
        return pd.DataFrame()

# --- MAIN APP LAYOUT ---
st.title("Sorority Recruitment Portal")

# Load data dynamically
roster = get_roster()
pnm_list = get_pnm_list()
party_options = get_party_options()

if not roster:
    st.warning("⚠️ Roster is empty. Please ensure the 'Member Information' sheet is populated with names.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "My Information", 
    "Create Bump Team", 
    "Party Excuses", 
    "View PNM Information", 
    "Prior PNM Connections"
])

# ==========================
# TAB 1: MY INFORMATION
# ==========================
with tab1:
    st.header("My Member Information")
    st.markdown("Please enter your details below. Your **Sorority ID** will be generated automatically upon submission.")
    
    with st.form(key='member_info_form'):
        member_name = st.selectbox("Your Name:", [""] + roster, key="info_name")
        
        col_a, col_b = st.columns(2)
        with col_a:
            major = st.text_input("Major")
            minor = st.text_input("Minor")
        with col_b:
            year = st.selectbox("Year", ["Sophomore", "Junior", "Senior"])
            hometown = st.text_input("Hometown (City, State)")
            
        hobbies = st.text_area("Hobbies")
        college_inv = st.text_area("College Involvement")
        hs_inv = st.text_area("High School Involvement")
        
        submit_info = st.form_submit_button(label='Save My Information')
        
    if submit_info:
        if not member_name:
             st.warning("⚠️ Please select your name.")
        else:
            sheet = get_sheet("Member Information")
            if sheet:
                try:
                    all_rows = sheet.get_all_values()
                    if not all_rows:
                        next_id = 1 
                    else:
                        next_id = len(all_rows) 

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    row_data = [
                        next_id, 
                        timestamp, 
                        member_name, 
                        major, 
                        minor, 
                        year, 
                        hometown, 
                        hobbies, 
                        college_inv, 
                        hs_inv
                    ]
                    sheet.append_row(row_data)
                    
                    st.success(f"✅ Information saved for {member_name}! Your Sorority ID is: **{next_id}**")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving data: {e}")
            else:
                st.error("Could not find 'Member Information' sheet. Please create it.")

# ==========================
# TAB 2: BUMP TEAM CREATION
# ==========================
with tab2:
    st.header("Bump Team Creation")
    st.markdown("Select your partners and RGL to form a bump team.")

    with st.form(key='bump_form'):
        creator_name = st.selectbox("Choose your name:", [""] + roster, key="bump_creator")
        partner_options = [n for n in roster if n != creator_name] if creator_name else roster
        partners = st.multiselect("Choose your bump partner(s):", partner_options)
        rgl = st.selectbox("Choose your Bump Group Leader (RGL):", ["None"] + roster)

        submit_bump = st.form_submit_button(label='Submit Bump Team')

    if submit_bump:
        if not creator_name or not partners:
            st.error("⚠️ Please fill in your name and at least one partner.")
        else:
            sheet = get_sheet("Bump Teams")
            if sheet:
                try:
                    existing_data = sheet.get_all_values()
                    next_id = len(existing_data) if existing_data else 1
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    partners_str = ", ".join(partners)
                    rgl_val = "" if rgl == "None" else rgl
                    
                    row_data = [timestamp, creator_name, partners_str, rgl_val, next_id, ""] 
                    sheet.append_row(row_data)
                    st.success(f"✅ Bump Team #{next_id} submitted successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving data: {e}")

# ==========================
# TAB 3: PARTY EXCUSES
# ==========================
with tab3:
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

# ==========================
# TAB 4: VIEW PNM INFORMATION (Modified)
# ==========================
with tab4:
    st.header("PNM Roster & Information")
    st.markdown("Use the dropdown to view a specific PNM or search the entire list.")
    
    # Refresh button
    if st.button("🔄 Refresh PNM List"):
        st.cache_data.clear()
        st.rerun()

    df_pnm = get_pnm_dataframe()
    
    if not df_pnm.empty:
        # --- FILTER COLUMNS ---
        # Exclude timestamp and ranking columns (case-insensitive check)
        cols_to_drop = [
            c for c in df_pnm.columns 
            if 'timestamp' in c.lower() or 'recruit rank' in c.lower() or 'average ranking' in c.lower()
        ]
        if cols_to_drop:
            df_pnm = df_pnm.drop(columns=cols_to_drop)

        # --- 1. Download Button ---
        csv = df_pnm.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All PNM Information (CSV)",
            data=csv,
            file_name="pnm_information.csv",
            mime="text/csv",
        )
        st.divider()

        # --- 2. Dropdown Setup ---
        name_col_idx = 1 if len(df_pnm.columns) > 1 else 0
        name_col_name = df_pnm.columns[name_col_idx]
        
        id_col_name = next((c for c in df_pnm.columns if 'id' in c.lower()), None)
        
        pnm_options = ["View All PNMs"]
        pnm_map = {} 
        
        for idx, row in df_pnm.iterrows():
            p_id = row[id_col_name] if id_col_name else (idx + 1)
            p_name = row[name_col_name]
            
            label = f"{p_id} - {p_name}"
            pnm_options.append(label)
            pnm_map[label] = idx

        # Dropdown Widget
        selected_view = st.selectbox("Select a PNM to view individual details:", pnm_options)

        # --- 3. Display Logic ---
        if selected_view == "View All PNMs":
            search_query = st.text_input("🔍 Search All PNMs (Name, Hometown, etc.):", key="pnm_search_input")
            
            if search_query:
                mask = df_pnm.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)
                display_df = df_pnm[mask]
            else:
                display_df = df_pnm
            
            st.dataframe(display_df, use_container_width=True)
            
        else:
            # --- INDIVIDUAL VIEW & RANKING ---
            row_idx = pnm_map[selected_view]
            pnm_data = df_pnm.iloc[row_idx]
            
            # Show formatted profile
            st.markdown(f"### 👤 Profile: {selected_view}")
            st.divider()
            
            col1, col2 = st.columns(2)
            fields = list(pnm_data.items())
            mid_point = (len(fields) + 1) // 2
            left_fields = fields[:mid_point]
            right_fields = fields[mid_point:]
            
            with col1:
                for label, value in left_fields:
                    st.markdown(f"**{label}**")
                    val_str = str(value).strip()
                    st.info(val_str if val_str else "N/A")
                    
            with col2:
                for label, value in right_fields:
                    st.markdown(f"**{label}**")
                    val_str = str(value).strip()
                    st.info(val_str if val_str else "N/A")

            # --- RANKING SECTION ---
            st.divider()
            st.subheader("⭐ Rate this PNM")
            st.markdown("Enter your ranking for this PNM below. This will be saved to the **Rankings** sheet.")

            with st.form(key=f"rank_form"):
                col_rank_1, col_rank_2 = st.columns(2)
                
                with col_rank_1:
                    ranker_name = st.selectbox("Your Name:", [""] + roster, key="ranker_name")
                
                with col_rank_2:
                    # Adjust min_value, max_value, and step as needed for your scoring system
                    score = st.number_input("Ranking Score (e.g., 0-5):", min_value=0.0, max_value=5.0, step=0.1)

                submit_rank = st.form_submit_button("Submit Ranking")

            if submit_rank:
                if not ranker_name:
                    st.warning("⚠️ Please select your name before submitting.")
                else:
                    sheet_rank = get_sheet("PNM Rankings")
                    if sheet_rank:
                        try:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Get Current PNM ID and Name for the record
                            curr_pnm_id = pnm_data[id_col_name] if id_col_name else (row_idx + 1)
                            curr_pnm_name = pnm_data[name_col_name]
                            
                            # Append: [Timestamp, Member Name, PNM ID, PNM Name, Score]
                            sheet_rank.append_row([timestamp, ranker_name, str(curr_pnm_id), curr_pnm_name, score])
                            
                            st.success(f"✅ Ranking saved for {curr_pnm_name}!")
                        except Exception as e:
                            st.error(f"Error saving ranking: {e}")
                    else:
                        st.error("❌ 'Rankings' sheet not found. Please create a sheet named 'Rankings' in your Google Sheet.")

    else:
        st.info("No PNM information found. Please ensure the 'PNM Information' sheet is populated.")

# ==========================
# TAB 5: PRIOR PNM CONNECTIONS
# ==========================
with tab5:
    st.header("Prior PNM Connections")
    st.markdown("Do you know a Potential New Member (PNM)? Let us know!")

    with st.form(key='connection_form'):
        col1, col2 = st.columns(2)
        with col1:
            member_name = st.selectbox("Your Name (Member):", [""] + roster, key="conn_member")
        with col2:
            target_pnm = st.selectbox("PNM Name:", [""] + pnm_list, key="conn_pnm")
        
        submit_connection = st.form_submit_button(label='Submit Connection')

    if submit_connection:
        if not member_name or not target_pnm:
            st.warning("⚠️ Please fill in all fields.")
        else:
            sheet = get_sheet("Prior Connections")
            if sheet:
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sheet.append_row([timestamp, member_name, target_pnm])
                    st.success(f"✅ Connection recorded for {member_name}!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving connection: {e}")
