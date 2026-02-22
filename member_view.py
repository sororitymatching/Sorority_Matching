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
        st.error(f"❌ Worksheet '{worksheet_name}' not found. Please create it.")
        return None

# --- HELPER FUNCTIONS FOR SEARCHING DATA ---

def find_row_by_col(sheet, col_index, value):
    try:
        all_values = sheet.get_all_values()
        idx_py = col_index - 1
        
        for i, row in enumerate(all_values):
            if len(row) > idx_py and row[idx_py].strip() == value.strip():
                return i + 1, row
        return None, None
    except Exception:
        return None, None

def find_row_composite(sheet, col_idx_1, val_1, col_idx_2, val_2):
    try:
        all_values = sheet.get_all_values()
        idx1_py = col_idx_1 - 1
        idx2_py = col_idx_2 - 1
        
        for i, row in enumerate(all_values):
            if (len(row) > max(idx1_py, idx2_py) and 
                row[idx1_py].strip() == val_1.strip() and 
                row[idx2_py].strip() == val_2.strip()):
                return i + 1, row
        return None, None
    except Exception:
        return None, None

# --- DATA FETCHING ---

@st.cache_data(ttl=60) 
def get_party_options():
    try:
        sheet = get_sheet("Settings")
        if not sheet: return ["Party 1", "Party 2", "Party 3", "Party 4"]
        val = sheet.acell('B1').value
        num_parties = int(val) if val and val.isdigit() else 4
        return [f"Party {i+1}" for i in range(num_parties)]
    except:
        return ["Party 1", "Party 2", "Party 3", "Party 4"]

@st.cache_data(ttl=300)
def get_roster():
    try:
        sheet = get_sheet("Member Information")
        if not sheet: return []
        headers = sheet.row_values(1)
        name_col_index = 3 
        for i, header in enumerate(headers):
            if header.lower().strip() in ["full name", "name", "member name"]:
                name_col_index = i + 1 
                break
        names = sheet.col_values(name_col_index) 
        if names:
            # Simple check to remove header if caught
            if "name" in names[0].lower() or "roster" in names[0].lower():
                names = names[1:]
        return sorted([n for n in names if n.strip()])
    except:
        return []

@st.cache_data(ttl=300)
def get_pnm_dataframe():
    try:
        sheet = get_sheet("PNM Information")
        if not sheet: return pd.DataFrame()
        data = sheet.get_all_values()
        if not data: return pd.DataFrame()
        return pd.DataFrame(data[1:], columns=data[0])
    except:
        return pd.DataFrame()

# --- MAIN APP LAYOUT ---
st.title("Sorority Recruitment Portal")

roster = get_roster()
party_options = get_party_options()

# We load the dataframe once here to use in both Tab 4 and Tab 5
df_pnm_global = get_pnm_dataframe()

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
    st.markdown("Select your name to edit your information.")
    
    member_name = st.selectbox("Your Name:", [""] + roster, key="info_name")
    
    defaults = {
        "major": "", "minor": "", "year": "Sophomore", 
        "hometown": "", "hobbies": "", "college_inv": "", "hs_inv": ""
    }
    existing_row_idx = None
    existing_id = None

    if member_name:
        sheet = get_sheet("Member Information")
        if sheet:
            row_idx, row_vals = find_row_by_col(sheet, 3, member_name)
            if row_idx:
                existing_row_idx = row_idx
                if len(row_vals) > 0: existing_id = row_vals[0]
                if len(row_vals) > 3: defaults["major"] = row_vals[3]
                if len(row_vals) > 4: defaults["minor"] = row_vals[4]
                if len(row_vals) > 5: defaults["year"] = row_vals[5]
                if len(row_vals) > 6: defaults["hometown"] = row_vals[6]
                if len(row_vals) > 7: defaults["hobbies"] = row_vals[7]
                if len(row_vals) > 8: defaults["college_inv"] = row_vals[8]
                if len(row_vals) > 9: defaults["hs_inv"] = row_vals[9]

    with st.form(key='member_info_form'):
        col_a, col_b = st.columns(2)
        with col_a:
            major = st.text_input("Major", value=defaults["major"])
            minor = st.text_input("Minor", value=defaults["minor"])
        with col_b:
            yr_opts = ["Sophomore", "Junior", "Senior"]
            yr_idx = yr_opts.index(defaults["year"]) if defaults["year"] in yr_opts else 0
            year = st.selectbox("Year", yr_opts, index=yr_idx)
            hometown = st.text_input("Hometown (City, State)", value=defaults["hometown"])
            
        hobbies = st.text_area("Hobbies", value=defaults["hobbies"])
        college_inv = st.text_area("College Involvement", value=defaults["college_inv"])
        hs_inv = st.text_area("High School Involvement", value=defaults["hs_inv"])
        
        submit_info = st.form_submit_button(label='Save My Information')
        
    if submit_info:
        if not member_name:
             st.warning("⚠️ Please select your name.")
        else:
            sheet = get_sheet("Member Information")
            if sheet:
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if existing_row_idx:
                        row_data = [existing_id, timestamp, member_name, major, minor, year, hometown, hobbies, college_inv, hs_inv]
                        sheet.update(f"A{existing_row_idx}:J{existing_row_idx}", [row_data])
                        st.success(f"✅ Information UPDATED for {member_name}!")
                    else:
                        all_rows = sheet.get_all_values()
                        next_id = len(all_rows) if all_rows else 1
                        row_data = [next_id, timestamp, member_name, major, minor, year, hometown, hobbies, college_inv, hs_inv]
                        sheet.append_row(row_data)
                        st.success(f"✅ Information SAVED for {member_name}!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving data: {e}")

# ==========================
# TAB 2: BUMP TEAM CREATION
# ==========================
with tab2:
    st.header("Bump Team Creation")
    st.markdown("Select your name to create or edit your bump team.")

    creator_name = st.selectbox("Choose your name (Creator):", [""] + roster, key="bump_creator")
    
    bump_defaults = {"partners": [], "rgl": "None"}
    bump_row_idx = None
    bump_id = None
    
    if creator_name:
        sheet_bump = get_sheet("Bump Teams")
        if sheet_bump:
            row_idx, row_vals = find_row_by_col(sheet_bump, 2, creator_name)
            if row_idx:
                bump_row_idx = row_idx
                if len(row_vals) > 2:
                    p_list = [p.strip() for p in row_vals[2].split(",")]
                    bump_defaults["partners"] = [p for p in p_list if p in roster]
                if len(row_vals) > 3:
                    rgl_val = row_vals[3]
                    bump_defaults["rgl"] = rgl_val if rgl_val in roster else "None"
                if len(row_vals) > 4:
                    bump_id = row_vals[4]

    with st.form(key='bump_form'):
        partner_options = [n for n in roster if n != creator_name] if creator_name else roster
        partners = st.multiselect("Choose your bump partner(s):", partner_options, default=bump_defaults["partners"])
        
        rgl_opts = ["None"] + roster
        rgl_idx = rgl_opts.index(bump_defaults["rgl"]) if bump_defaults["rgl"] in rgl_opts else 0
        rgl = st.selectbox("Choose your Bump Group Leader (RGL):", rgl_opts, index=rgl_idx)

        submit_bump = st.form_submit_button(label='Submit Bump Team')

    if submit_bump:
        if not creator_name or not partners:
            st.error("⚠️ Please fill in your name and at least one partner.")
        else:
            sheet = get_sheet("Bump Teams")
            if sheet:
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    partners_str = ", ".join(partners)
                    rgl_val = "" if rgl == "None" else rgl
                    
                    if bump_row_idx:
                        row_data = [timestamp, creator_name, partners_str, rgl_val, bump_id]
                        sheet.update(f"A{bump_row_idx}:E{bump_row_idx}", [row_data])
                        st.success(f"✅ Bump Team #{bump_id} UPDATED!")
                    else:
                        existing_data = sheet.get_all_values()
                        next_id = len(existing_data) if existing_data else 1
                        row_data = [timestamp, creator_name, partners_str, rgl_val, next_id]
                        sheet.append_row(row_data)
                        st.success(f"✅ Bump Team #{next_id} CREATED!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================
# TAB 3: PARTY EXCUSES
# ==========================
with tab3:
    st.header("Recruitment Party Excuse Form")
    st.markdown("Select your name to view or update your excuses.")

    excuse_name = st.selectbox("Choose your name:", [""] + roster, key="excuse_name")
    
    excuse_defaults = []
    excuse_row_idx = None
    
    if excuse_name:
        sheet_excuses = get_sheet("Party Excuses")
        if sheet_excuses:
            row_idx, row_vals = find_row_by_col(sheet_excuses, 2, excuse_name)
            if row_idx:
                excuse_row_idx = row_idx
                if len(row_vals) > 3:
                    p_list = [p.strip() for p in row_vals[3].split(",")]
                    excuse_defaults = [p for p in p_list if p in party_options]

    with st.form(key='excuse_form'):
        parties = st.multiselect("Parties you cannot attend:", party_options, default=excuse_defaults)
        submit_excuse = st.form_submit_button(label='Submit Excuse')

    if submit_excuse:
        if not excuse_name or not parties:
            st.warning("⚠️ Please fill in details.")
        else:
            sheet_excuses = get_sheet("Party Excuses")
            sheet_mem = get_sheet("Member Information")
            
            if sheet_excuses and sheet_mem:
                try:
                    row_i, row_v = find_row_by_col(sheet_mem, 3, excuse_name)
                    member_id = row_v[0] if row_v else "Unknown"
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    parties_str = ", ".join(parties)
                    
                    if excuse_row_idx:
                        sheet_excuses.update(f"A{excuse_row_idx}:D{excuse_row_idx}", [[timestamp, excuse_name, member_id, parties_str]])
                        st.success("✅ Excuse UPDATED!")
                    else:
                        sheet_excuses.append_row([timestamp, excuse_name, member_id, parties_str])
                        st.success("✅ Excuse SUBMITTED!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================
# TAB 4: VIEW PNM INFORMATION & RANKING
# ==========================
with tab4:
    st.header("PNM Roster & Information")
    if st.button("🔄 Refresh PNM List"):
        st.cache_data.clear()
        st.rerun()

    df_pnm = df_pnm_global.copy()
    
    # Pre-calculate mapping for dropdowns (Used in Tab 4 and Tab 5)
    pnm_options = []
    pnm_map = {}
    pnm_id_name_list = [] # For Tab 5
    
    if not df_pnm.empty:
        # Columns cleanup
        cols_to_drop = [c for c in df_pnm.columns if 'timestamp' in c.lower() or 'rank' in c.lower()]
        if cols_to_drop: df_pnm = df_pnm.drop(columns=cols_to_drop)

        # Download
        csv = df_pnm.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "pnm_info.csv", "text/csv")
        st.divider()

        # Identify Key Columns (More Robust Logic)
        # Try to find 'id' in headers, else default to column 0 (usually ID)
        id_col = next((c for c in df_pnm.columns if 'id' in c.lower() and 'user' not in c.lower()), df_pnm.columns[0])
        # Try to find 'name' in headers, else default to column 1
        name_col = next((c for c in df_pnm.columns if 'name' in c.lower() or 'pnm' in c.lower()), df_pnm.columns[1] if len(df_pnm.columns) > 1 else df_pnm.columns[0])
        
        pnm_options = ["View All PNMs"]
        for idx, row in df_pnm.iterrows():
            # Create Label: "ID - Name"
            pnm_id = str(row[id_col]).strip()
            pnm_name = str(row[name_col]).strip()
            label = f"{pnm_id} - {pnm_name}"
            
            pnm_options.append(label)
            pnm_map[label] = idx
            pnm_id_name_list.append(label)

        selected_view = st.selectbox("Select PNM:", pnm_options)

        if selected_view == "View All PNMs":
            search = st.text_input("🔍 Search:", key="pnm_search")
            if search:
                mask = df_pnm.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                st.dataframe(df_pnm[mask], use_container_width=True)
            else:
                st.dataframe(df_pnm, use_container_width=True)
        else:
            # --- SINGLE PNM VIEW ---
            row_idx = pnm_map[selected_view]
            pnm_data = df_pnm.iloc[row_idx]
            
            curr_pnm_id = str(pnm_data[id_col])
            curr_pnm_name = pnm_data[name_col]
            
            # Display Profile
            st.markdown(f"### 👤 {curr_pnm_name} (ID: {curr_pnm_id})")
            st.divider()
            col1, col2 = st.columns(2)
            fields = list(pnm_data.items())
            mid = (len(fields) + 1) // 2
            with col1:
                for k,v in fields[:mid]: st.info(f"**{k}:** {v}")
            with col2:
                for k,v in fields[mid:]: st.info(f"**{k}:** {v}")

            # --- RANKING FORM ---
            st.divider()
            st.subheader("⭐ Rate this PNM")
            
            ranker_name = st.selectbox("Your Name (Ranker):", [""] + roster, key="ranker_name")
            
            rank_default = 0.0
            existing_rank_row = None
            
            if ranker_name:
                sheet_rank = get_sheet("PNM Rankings")
                if sheet_rank:
                    r_idx, r_vals = find_row_composite(sheet_rank, 3, ranker_name, 4, curr_pnm_id)
                    if r_idx:
                        existing_rank_row = r_idx
                        if len(r_vals) > 5:
                            try:
                                rank_default = float(r_vals[5])
                            except:
                                rank_default = 0.0

            with st.form(key=f"rank_form"):
                score = st.number_input("Score (0-5):", min_value=0.0, max_value=5.0, step=0.1, value=rank_default)
                submit_rank = st.form_submit_button("Submit Ranking")

            if submit_rank:
                if not ranker_name:
                    st.warning("⚠️ Select your name.")
                else:
                    sheet_rank = get_sheet("PNM Rankings")
                    sheet_mem = get_sheet("Member Information")
                    
                    if sheet_rank and sheet_mem:
                        try:
                            m_idx, m_vals = find_row_by_col(sheet_mem, 3, ranker_name)
                            ranker_id = m_vals[0] if m_vals else "Unknown"
                            
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            row_data = [timestamp, ranker_id, ranker_name, curr_pnm_id, curr_pnm_name, score]
                            
                            if existing_rank_row:
                                sheet_rank.update(f"A{existing_rank_row}:F{existing_rank_row}", [row_data])
                                st.success("✅ Ranking UPDATED!")
                            else:
                                sheet_rank.append_row(row_data)
                                st.success("✅ Ranking SUBMITTED!")
                        except Exception as e:
                            st.error(f"Error: {e}")

# ==========================
# TAB 5: PRIOR PNM CONNECTIONS (UPDATED)
# ==========================
with tab5:
    st.header("Prior PNM Connections")
    st.markdown("Select a PNM from the list (Sorted by ID - Name)")
    
    col1, col2 = st.columns(2)
    with col1:
        conn_member = st.selectbox("Your Name:", [""] + roster, key="conn_mem")
    with col2:
        # Now uses the combined "ID - Name" list from the DataFrame
        conn_pnm_selection = st.selectbox("PNM (ID - Name):", [""] + pnm_id_name_list, key="conn_pnm")

    existing_conn_row = None
    
    # Parse the selection to get simple Name for searching
    clean_pnm_name = ""
    clean_pnm_id = ""
    
    if conn_pnm_selection:
        # Format is "ID - Name", we split by " - " (first occurrence only)
        parts = conn_pnm_selection.split(" - ", 1)
        if len(parts) == 2:
            clean_pnm_id = parts[0]
            clean_pnm_name = parts[1]
        else:
            clean_pnm_name = conn_pnm_selection # Fallback

    if conn_member and clean_pnm_name:
        sheet_conn = get_sheet("Prior Connections")
        if sheet_conn:
            # We search by Name (Col 4) to check existence
            c_idx, c_vals = find_row_composite(sheet_conn, 2, conn_member, 4, clean_pnm_name)
            if c_idx:
                existing_conn_row = c_idx
                st.info(f"ℹ️ You already have a connection logged with {clean_pnm_name}. Submitting again will update it.")

    with st.form(key='connection_form'):
        submit_connection = st.form_submit_button(label='Submit/Update Connection')

    if submit_connection:
        if not conn_member or not conn_pnm_selection:
            st.warning("⚠️ Please select both names.")
        else:
            sheet_conn = get_sheet("Prior Connections")
            sheet_mem = get_sheet("Member Information")
            # We don't need sheet_pnm lookup anymore because we have ID from the dropdown!

            if sheet_conn and sheet_mem:
                try:
                    # Member ID
                    m_idx, m_vals = find_row_by_col(sheet_mem, 3, conn_member)
                    m_id = m_vals[0] if m_vals else "?"
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row_data = [timestamp, conn_member, m_id, clean_pnm_name, clean_pnm_id]
                    
                    if existing_conn_row:
                        sheet_conn.update(f"A{existing_conn_row}:E{existing_conn_row}", [row_data])
                        st.success("✅ Connection Timestamp UPDATED!")
                    else:
                        sheet_conn.append_row(row_data)
                        st.success("✅ Connection SAVED!")
                except Exception as e:
                    st.error(f"Error: {e}")
