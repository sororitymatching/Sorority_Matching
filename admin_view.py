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

def update_settings(cell, value):
    try:
        gc = get_gc()
        gc.open(SHEET_NAME).worksheet("Settings").update_acell(cell, value)
        return True
    except: return False

def update_roster(names_list):
    try:
        gc = get_gc()
        ws = gc.open(SHEET_NAME).worksheet("Settings")
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
        # PNM ID is column 24 (index 23), Rank is column 25 (index 24)
        cell = sheet.find(str(pnm_id), in_column=24)
        if cell:
            sheet.update_cell(cell.row, 25, new_ranking)
            return True
        else:
            # Silent fail to avoid spamming errors if ID mismatch
            print(f"Could not find PNM ID {pnm_id}")
            return False
    except Exception as e:
        print(f"Error updating PNM ranking: {e}")
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Settings & Roster", 
        "Member Information",
        "PNM Information and Rankings", 
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
                if update_settings('B1', count): st.toast("Updated!")
        st.divider()
        st.header("Roster Management")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.subheader("Option A: Sync from Sheet")
            st.info("Pull names directly from the 'Member Information' tab.")
            if st.button("🔄 Sync Roster from 'Member Information'"):
                df_source = get_data("Member Information")
                if not df_source.empty:
                    possible_cols = ["Full Name", "Name", "Member Name", "Member"]
                    found_col = None
                    for col in possible_cols:
                        match = next((c for c in df_source.columns if c.lower() == col.lower()), None)
                        if match:
                            found_col = match
                            break
                    if found_col:
                        names = df_source[found_col].astype(str).unique().tolist()
                        names = [n for n in names if n.strip()] 
                        if update_roster(names):
                            st.success(f"✅ Successfully synced {len(names)} members!")
                        else:
                            st.error("Failed to update Settings.")
                    else:
                        st.error(f"Could not find name column. Searched: {possible_cols}")
                else:
                    st.error("'Member Information' sheet is empty.")
        with col_r2:
            st.subheader("Option B: Upload CSV")
            file = st.file_uploader("Upload Member List (CSV)", type="csv")
            if file:
                try:
                    new_names = pd.read_csv(file, header=None)[0].astype(str).tolist()
                    if st.button("Replace Roster with CSV"):
                        if update_roster(new_names): st.toast("Roster Updated!")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

    # --- TAB 2: MEMBER INFORMATION ---
    with tab2:
        st.header("Member Information Database")
        if st.button("🔄 Refresh Member Data"): st.rerun()
        try:
            df_members = get_data("Member Information")
        except:
            df_members = pd.DataFrame()
        if not df_members.empty:
            search_mem = st.text_input("🔍 Search Members:")
            if search_mem:
                mask = df_members.apply(lambda x: x.astype(str).str.contains(search_mem, case=False).any(), axis=1)
                display_df = df_members[mask]
            else:
                display_df = df_members
            st.metric("Total Members", len(display_df))
            st.dataframe(display_df, use_container_width=True)
            csv_mem = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_mem, "member_info.csv", "text/csv")
        else:
            st.info("No member information found.")

    # --- TAB 3: PNM RANKINGS (MODIFIED - AUTO SYNC) ---
    with tab3:
        st.header("PNM Ranking Management")
        
        # 1. Fetch Rankings from 'PNM Rankings' sheet
        df_votes = get_data("PNM Rankings")
        
        if not df_votes.empty:
            try:
                df_votes['Score'] = pd.to_numeric(df_votes['Score'], errors='coerce')
                
                if 'PNM ID' in df_votes.columns and 'Score' in df_votes.columns:
                    
                    # 2. Calculate Averages
                    group_cols = ['PNM ID']
                    if 'PNM Name' in df_votes.columns:
                        group_cols.append('PNM Name')
                        
                    avg_df = df_votes.groupby(group_cols)['Score'].mean().reset_index()
                    avg_df.rename(columns={'Score': 'Calculated Average'}, inplace=True)
                    avg_df = avg_df.sort_values(by='Calculated Average', ascending=False)
                    
                    st.info(f"Processing {len(df_votes)} total votes across {len(avg_df)} unique PNMs...")
                    
                    # 3. Automatic Sync (Write back to sheet)
                    # We use a spinner to indicate activity, though it happens on load/refresh
                    with st.spinner("Syncing rankings to 'PNM Information' sheet..."):
                        updates_count = 0
                        # Iterate and update
                        # Optimization: In a real app, batch update is better. 
                        # Here we iterate as per existing helper pattern.
                        for idx, row in avg_df.iterrows():
                            p_id = row['PNM ID']
                            score = round(row['Calculated Average'], 2)
                            # Update the sheet
                            # Note: This is slow if list is long. 
                            if update_pnm_ranking(p_id, score):
                                updates_count += 1
                        
                    if updates_count > 0:
                        st.toast(f"✅ Auto-synced {updates_count} PNM rankings!", icon="🔄")
                    st.subheader("📄 Raw Ranking Data (PNM Rankings Sheet)")
                    st.dataframe(df_votes, use_container_width=True)
                    
                else:
                    st.error("Missing columns 'PNM ID' or 'Score' in 'PNM Rankings' sheet.")
                    
            except Exception as e:
                st.error(f"Error processing rankings: {e}")
        else:
            st.info("No votes found in 'PNM Rankings' sheet yet.")

        st.divider()
        st.subheader("📋 Current PNM Database")
        
        # Load Existing PNM Data
        df_pnms = get_data("PNM Information")
        
        if not df_pnms.empty:
            pnm_search = st.text_input("🔍 Search PNM Database:")
            if pnm_search:
                mask = df_pnms.astype(str).apply(lambda x: x.str.contains(pnm_search, case=False).any(), axis=1)
                display_pnm_df = df_pnms[mask]
            else:
                display_pnm_df = df_pnms

            st.dataframe(display_pnm_df, use_container_width=True)
            csv_pnms = display_pnm_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download PNM Data CSV", csv_pnms, "pnm_data.csv", "text/csv")
        else:
            st.info("No PNM data found.")

    # --- TAB 4: VIEW BUMP TEAMS ---
    with tab4:
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
            st.dataframe(display_df, use_container_width=True)
            csv_teams = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_teams, "bump_teams.csv", "text/csv")
        else:
            st.info("No bump teams found yet.")

    # --- TAB 5: VIEW EXCUSES ---
    with tab5:
        if st.button("🔄 Refresh Excuses"): st.rerun()
        try:
            df_excuses = get_data("Party Excuses")
        except:
            df_excuses = pd.DataFrame()
        if not df_excuses.empty:
            st.dataframe(df_excuses, use_container_width=True)
            csv = df_excuses.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "excuses.csv", "text/csv")
        else:
            st.info("No excuses found.")

    # --- TAB 6: VIEW PRIOR CONNECTIONS ---
    with tab6:
        st.header("Prior PNM Connections Log")
        if st.button("🔄 Refresh Connections"): st.rerun()
        try:
            df_connections = get_data("Prior Connections")
        except:
            df_connections = pd.DataFrame()
        if not df_connections.empty:
            search_conn = st.text_input("🔍 Search Connections:")
            if search_conn:
                mask = df_connections.apply(lambda x: x.astype(str).str.contains(search_conn, case=False).any(), axis=1)
                df_connections = df_connections[mask]
            st.dataframe(df_connections, use_container_width=True)
            csv_conn = df_connections.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_conn, "prior_connections.csv", "text/csv")
        else:
            st.info("No prior connections found.")
