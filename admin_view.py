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
        if worksheet_name == "PNM Information":
             st.warning("Could not find 'PNM Information' tab.")
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
        cell = sheet.find(str(team_id), in_column=5)
        if cell:
            sheet.update_cell(cell.row, 6, new_ranking)
            return True
        return False
    except Exception as e:
        st.error(f"Error updating team ranking: {e}")
        return False

def update_pnm_ranking(pnm_id, new_ranking):
    """Finds the PNM ID in the PNM Information sheet and updates their ranking."""
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("PNM Information")
        
        # PNM ID is the 24th item (Column X)
        # Rank is the 25th item (Column Y)
        cell = sheet.find(str(pnm_id), in_column=24)
        
        if cell:
            sheet.update_cell(cell.row, 25, new_ranking)
            return True
        else:
            st.error(f"Could not find PNM ID {pnm_id}")
            return False
            
    except Exception as e:
        st.error(f"Error updating PNM ranking: {e}")
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

    # TABS - Added "View Prior Connections" as the last tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Settings & Roster", 
        "PNM Rankings", 
        "View Bump Teams", 
        "View Excuses", 
        "View Prior Connections"
    ])

    # --- TAB 1: SETTINGS ---
    with tab1:
        st.header("Event Configuration")
        
        with st.form("party_config"):
            count = st.number_input("Number of Parties", 1, 50, 4)
            if st.form_submit_button("Update Party Count"):
                if update_config('B1', count): st.toast("Updated!")
        
        st.divider()
        st.header("Roster Management")
        
        file = st.file_uploader("Upload Member List (CSV)", type="csv")
        if file:
            try:
                new_names = pd.read_csv(file, header=None)[0].astype(str).tolist()
                st.write(f"Preview: {new_names[:3]}...")
                if st.button("Replace Roster"):
                    if update_roster(new_names): st.toast("Roster Updated Successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # --- TAB 2: PNM RANKINGS ---
    with tab2:
        st.header("PNM Ranking Management")
        
        # Load Data
        df_pnms = get_data("PNM Information")
        
        if not df_pnms.empty:
            # Create Ranking Interface
            with st.expander("⭐ Assign/Update PNM Rankings", expanded=True):
                # Helper column for dropdown
                name_col = "Enter your name:" if "Enter your name:" in df_pnms.columns else df_pnms.columns[1]
                
                df_pnms['display_label'] = df_pnms.apply(
                    lambda x: f"ID: {x['PNM ID']} | {x[name_col]}", 
                    axis=1
                )
                
                col_p1, col_p2, col_p3 = st.columns([3, 1, 1])
                
                with col_p1:
                    selected_pnm_label = st.selectbox("Select PNM to Rank:", df_pnms['display_label'].tolist())
                    selected_pnm_id = df_pnms[df_pnms['display_label'] == selected_pnm_label]['PNM ID'].values[0]

                with col_p2:
                    current_pnm_rank = df_pnms[df_pnms['PNM ID'] == selected_pnm_id]['Average Recruit Rank'].values[0]
                    try:
                        initial_pnm_val = float(current_pnm_rank)
                    except:
                        initial_pnm_val = 0.0
                    
                    new_pnm_rank = st.number_input("Assign Rank:", min_value=0.0, max_value=5.0, value=initial_pnm_val, step=0.01, key="pnm_rank_input")

                with col_p3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Save PNM Rank"):
                        if update_pnm_ranking(selected_pnm_id, new_pnm_rank):
                            st.success(f"Rank {new_pnm_rank} assigned to PNM {selected_pnm_id}!")
                            st.rerun()

            st.divider()

            # Search & Table
            pnm_search = st.text_input("🔍 Search PNMs:")
            display_pnm_df = df_pnms.drop(columns=['display_label'])

            if pnm_search:
                mask = display_pnm_df.apply(lambda x: x.astype(str).str.contains(pnm_search, case=False).any(), axis=1)
                display_pnm_df = display_pnm_df[mask]

            st.metric("Total PNMs", len(display_pnm_df))
            st.dataframe(display_pnm_df, use_container_width=True, hide_index=True)
            
            csv_pnms = display_pnm_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download PNM Data CSV", csv_pnms, "pnm_data.csv", "text/csv")
        
        else:
            st.info("No PNM data found yet.")

    # --- TAB 3: VIEW BUMP TEAMS ---
    with tab3:
        st.header("Bump Team Management")
        
        df_teams = get_data("Bump Teams")
        
        if not df_teams.empty:
            with st.expander("⭐ Assign/Update Team Rankings", expanded=True):
                df_teams['display_label'] = df_teams.apply(
                    lambda x: f"Team {x['Team ID']} | {x['Creator Name']}, {x['Bump Partners']}", 
                    axis=1
                )
                
                col_a, col_b, col_c = st.columns([3, 1, 1])
                
                with col_a:
                    selected_label = st.selectbox("Select Team to Rank:", df_teams['display_label'].tolist())
                    selected_team_id = df_teams[df_teams['display_label'] == selected_label]['Team ID'].values[0]
                
                with col_b:
                    current_rank = df_teams[df_teams['Team ID'] == selected_team_id]['Ranking'].values[0]
                    initial_val = int(current_rank) if str(current_rank).isdigit() else 1
                    new_rank = st.number_input(f"Assign Rank:", min_value=1, value=initial_val, key="team_rank_input")
                
                with col_c:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Save Team Rank"):
                        if update_team_ranking(selected_team_id, new_rank):
                            st.success(f"Rank {new_rank} assigned!")
                            st.rerun() 

            st.divider()
            
            search = st.text_input("🔍 Search Teams:")
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

    # --- TAB 4: VIEW EXCUSES ---
    with tab4:
        if st.button("🔄 Refresh Excuses"): st.rerun()
        try:
            df_excuses = get_data("Party Excuses")
        except:
            df_excuses = pd.DataFrame()

        if not df_excuses.empty:
            st.dataframe(df_excuses, use_container_width=True)
            csv = df_excuses.to_csv(index=False).encode('utf-8')
            st.download_button("Download Excuses CSV", csv, "excuses.csv", "text/csv")
        else:
            st.info("No excuses found.")

    # --- TAB 5: VIEW PRIOR CONNECTIONS ---
    with tab5:
        st.header("Prior PNM Connections Log")
        
        if st.button("🔄 Refresh Connections"): st.rerun()
        
        try:
            # Assumes the user form writes to "Prior Connections" tab
            df_connections = get_data("Prior Connections")
        except:
            df_connections = pd.DataFrame()

        if not df_connections.empty:
            # Optional Search bar
            search_conn = st.text_input("🔍 Search Connections (by Member or PNM):")
            
            if search_conn:
                mask = df_connections.apply(lambda x: x.astype(str).str.contains(search_conn, case=False).any(), axis=1)
                df_connections = df_connections[mask]

            st.dataframe(df_connections, use_container_width=True)
            
            csv_conn = df_connections.to_csv(index=False).encode('utf-8')
            st.download_button("Download Connections CSV", csv_conn, "prior_connections.csv", "text/csv")
        else:
            st.info("No prior connections found.")
