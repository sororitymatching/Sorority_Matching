import streamlit as st
import gspread
import pandas as pd

# --- CONFIGURATION ---
SHEET_NAME = "OverallMatchingInformation"
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

def update_team_ranking(team_id, new_ranking):
    """Finds the Team ID in the Bump Teams sheet and updates its ranking."""
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("Bump Teams")
        # Column E (index 5) is Team ID. Column F (index 6) is Ranking.
        # Find the row where Column E matches the team_id
        cell = sheet.find(str(team_id), in_column=5)
        if cell:
            # Update the cell to the immediate right (Column F)
            sheet.update_cell(cell.row, 6, new_ranking)
            return True
        return False
    except Exception as e:
        st.error(f"Error updating ranking: {e}")
        return False

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

    # --- TAB 2: VIEW BUMP TEAMS (ADMIN VIEW) ---
    with tab2:
        st.header("Bump Team Management")
        
        # LOAD DATA
        df_teams = get_data("Bump Teams")
        
        if not df_teams.empty:
            # --- NEW: RANKING EDITOR ---
            with st.expander("⭐ Assign/Update Team Rankings", expanded=True):
                # 1. Create a descriptive label for each team
                # format: "ID: 1 | Members: Hailey Abbott, Sophie Adams..."
                df_teams['display_label'] = df_teams.apply(
                    lambda x: f"Team {x['Team ID']} | {x['Creator Name']}, {x['Bump Partners']}", 
                    axis=1
                )
                
                col_a, col_b, col_c = st.columns([3, 1, 1])
                
                with col_a:
                    # Dropdown shows the full description, but we map it back to the ID
                    selected_label = st.selectbox("Select Team to Rank:", df_teams['display_label'].tolist())
                    
                    # Extract the actual Team ID from the selection to use for the update function
                    selected_team_id = df_teams[df_teams['display_label'] == selected_label]['Team ID'].values[0]
                
                with col_b:
                    # Default the input to the current rank if it exists, otherwise 1
                    current_rank = df_teams[df_teams['Team ID'] == selected_team_id]['Ranking'].values[0]
                    initial_val = int(current_rank) if str(current_rank).isdigit() else 1
                    
                    new_rank = st.number_input(f"Assign Rank:", min_value=1, value=initial_val, key="rank_input")
                
                with col_c:
                    st.markdown("<br>", unsafe_allow_html=True) # Align button
                    if st.button("Save Rank"):
                        if update_team_ranking(selected_team_id, new_rank):
                            st.success(f"Rank {new_rank} assigned!")
                            st.rerun() 

            st.divider()

            # SEARCH & TABLE
            search = st.text_input("🔍 Search Teams:")
            
            # Remove the temporary display column before showing the table or downloading
            display_df = df_teams.drop(columns=['display_label'])
            
            if search:
                mask = display_df.apply(lambda x: x.astype(str).str.contains(search, case=False).any(), axis=1)
                display_df = display_df[mask]

            st.metric("Total Teams", len(display_df))
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            csv_teams = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Teams CSV", csv_teams, "bump_teams.csv", "text/csv")
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

    
    



