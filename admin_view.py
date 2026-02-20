import streamlit as st
import gspread
import pandas as pd

# --- CONFIGURATION ---
SHEET_NAME = "Party Excuses (Test)"
ADMIN_PASSWORD = "password"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# --- HELPERS ---
def get_gc():
    creds_dict = dict(st.secrets["gcp_service_account"])
    return gspread.service_account_from_dict(creds_dict, scopes=SCOPES)

def get_data(worksheet_name):
    """Gets all data from a specific worksheet."""
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet(worksheet_name)
        data = sheet.get_all_values()
        if not data: return pd.DataFrame()
        return pd.DataFrame(data[1:], columns=data[0])
    except Exception as e:
        if worksheet_name == "Bump Teams":
            st.warning("Could not find 'Bump Teams' tab. Please create it in Google Sheets.")
        return pd.DataFrame()

def update_config(cell, value):
    try:
        gc = get_gc()
        gc.open(SHEET_NAME).worksheet("Config").update_acell(cell, value)
        return True
    except: return False

def update_roster(names_list):
    try:
        gc = get_gc()
        ws = gc.open(SHEET_NAME).worksheet("Config")
        ws.batch_clear(["D2:D1000"])
        names_list.sort()
        formatted = [[n] for n in names_list if n.strip()]
        if formatted: ws.update(range_name='D2', values=formatted)
        return True
    except: return False

# --- MAIN PAGE ---
st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("📊 Sorority Admin Dashboard")

# LOGIN
if "authenticated" not in st.session_state: st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("Enter Admin Password:", type="password")
    if pwd == ADMIN_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
else:
    st.success("Logged in as Admin")

    # TABS
    tab1, tab2, tab3 = st.tabs(["Settings & Roster", "View Bump Teams", "View Excuses"])

    with tab1:
        st.header("Event Configuration")
        
        # Party Count
        with st.form("party_config"):
            count = st.number_input("Number of Parties", 1, 50, 4)
            if st.form_submit_button("Update Party Count"):
                if update_config('B1', count): st.toast("Updated!")
        
        st.divider()
        st.header("Roster Management")
        
        # Roster Upload
        file = st.file_uploader("Upload Member List (CSV)", type="csv")
        if file:
            try:
                new_names = pd.read_csv(file, header=None)[0].astype(str).tolist()
                st.write(f"Preview: {new_names[:3]}...")
                if st.button("Replace Roster"):
                    if update_roster(new_names): st.toast("Roster Updated Successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with tab2:
        if st.button("🔄 Refresh Teams"): st.rerun()
        df_teams = get_data("Bump Teams")
        
        if not df_teams.empty:
            # Search filter
            search = st.text_input("🔍 Search Teams by Creator or Partner:")
            if search:
                # Search across multiple columns
                mask = df_teams.apply(lambda x: x.astype(str).str.contains(search, case=False).any(), axis=1)
                df_teams = df_teams[mask]

            st.metric("Total Teams Created", len(df_teams))
            st.dataframe(df_teams, use_container_width=True)
            csv_teams = df_teams.to_csv(index=False).encode('utf-8')
            st.download_button("Download Bump Teams CSV", csv_teams, "bump_teams.csv", "text/csv")
        else:
            st.info("No bump teams found yet.")

    with tab3:
        if st.button("🔄 Refresh Excuses"): st.rerun()
        # Assuming excuses are on the first sheet (Sheet1)
        # Note: You can rename 'Sheet1' in your Google Sheet, but you must update code if you do.
        # We'll assume the first sheet is always excuses.
        try:
            gc = get_gc()
            df_excuses = get_data("Party Excuses")
        except:
            df_excuses = pd.DataFrame()

        if not df_excuses.empty:
            st.dataframe(df_excuses, use_container_width=True)
            csv = df_excuses.to_csv(index=False).encode('utf-8')
            st.download_button("Download Excuses CSV", csv, "excuses.csv", "text/csv")
        else:
            st.info("No excuses found.")

    
    



