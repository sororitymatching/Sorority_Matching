import streamlit as st
import gspread
import pandas as pd
import numpy as np
import networkx as nx
import re
import difflib
import io
from math import radians
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances

# --- CONFIGURATION ---
SHEET_NAME = "OverallMatchingInformation"
ADMIN_PASSWORD = "password"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# --- CACHED RESOURCES ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- HELPERS ---
def get_gc():
    creds_dict = dict(st.secrets["gcp_service_account"])
    return gspread.service_account_from_dict(creds_dict, scopes=SCOPES)

def get_data(worksheet_name):
    """Gets all data and standardizes headers slightly."""
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet(worksheet_name)
        data = sheet.get_all_values()
        if not data: return pd.DataFrame()
        
        # Create DF
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Basic cleanup: Strip whitespace from column headers
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        if worksheet_name == "PNM Information":
             st.warning("Could not find 'PNM Information' tab.")
        return pd.DataFrame()

def get_setting_value(cell):
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("Settings")
        return sheet.acell(cell).value
    except:
        return None

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

def get_active_roster_names():
    """
    Returns the list of active members.
    Priority:
    1. 'Settings' sheet Column D (Populated via CSV Upload or Sync).
    2. Fallback to 'Member Information' sheet if Settings is empty.
    """
    # 1. Try getting from Settings (The "Uploaded/Synced" Roster)
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("Settings")
        # Assuming D2:D contains the list
        roster_data = sheet.get_values("D2:D")
        names = [r[0].strip() for r in roster_data if r and r[0].strip()]
        if names:
            return names
    except Exception:
        pass

    # 2. Fallback to Member Information Sheet
    df_mem = get_data("Member Information")
    if not df_mem.empty:
        # Attempt to find the Name column dynamically
        possible_cols = ["Full Name", "Name", "Member Name", "Member"]
        found_col = None
        for col in df_mem.columns:
            if any(c.lower() in col.lower() for c in possible_cols):
                found_col = col
                break
        
        if found_col:
            return df_mem[found_col].dropna().astype(str).str.strip().tolist()
    
    return []

def update_team_ranking(team_id, new_ranking):
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

# --- NEW BATCH UPDATE FUNCTIONS ---
def batch_update_pnm_rankings(rankings_map):
    """
    Updates PNM rankings in batch to avoid API limits and slowness.
    rankings_map: { 'pnm_id': new_score }
    """
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("PNM Information")
        
        # Get all data to edit in memory
        all_values = sheet.get_all_values()
        if not all_values: return 0
        
        headers = [h.lower().strip() for h in all_values[0]]
        
        # Find column indices (0-based)
        try:
            id_idx = next(i for i, h in enumerate(headers) if 'pnm id' in h or 'id' == h)
        except StopIteration:
            # Fallback to column 24 (index 23) if header not found
            id_idx = 23
            
        try:
            rank_idx = next(i for i, h in enumerate(headers) if 'recruit rank' in h or 'average' in h)
        except StopIteration:
            # Fallback to column 25 (index 24)
            rank_idx = 24

        updates_count = 0
        
        # Iterate through data (skipping header)
        for i in range(1, len(all_values)):
            row = all_values[i]
            # Ensure row has enough columns
            if len(row) <= id_idx: continue
            
            p_id = str(row[id_idx]).strip()
            
            if p_id in rankings_map:
                # Ensure row is long enough for the rank column
                while len(row) <= rank_idx:
                    row.append("")
                
                # Update the value
                row[rank_idx] = str(rankings_map[p_id])
                updates_count += 1
        
        # Write everything back at once
        sheet.update(values=all_values, range_name="A1")
        return updates_count
        
    except Exception as e:
        st.error(f"Batch update failed: {e}")
        return 0

def batch_update_team_rankings(rankings_map):
    """
    Updates Bump Team rankings in batch.
    rankings_map: { 'team_id': new_rank }
    """
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("Bump Teams")
        
        all_values = sheet.get_all_values()
        if not all_values: return 0
        
        headers = [h.lower().strip() for h in all_values[0]]
        
        # Find ID column
        try:
            id_idx = next(i for i, h in enumerate(headers) if 'team id' in h or 'id' == h)
        except StopIteration:
            # Fallback based on typical structure (Timestamp, Creator, Partners, RGL, ID, Ranking)
            id_idx = 4
            
        # Find Ranking column
        try:
            rank_idx = next(i for i, h in enumerate(headers) if 'ranking' in h or 'rank' in h)
        except StopIteration:
            rank_idx = 5

        updates_count = 0
        
        for i in range(1, len(all_values)):
            row = all_values[i]
            if len(row) <= id_idx: continue
            
            t_id = str(row[id_idx]).strip()
            
            if t_id in rankings_map:
                # Ensure row is long enough
                while len(row) <= rank_idx:
                    row.append("")
                row[rank_idx] = str(rankings_map[t_id])
                updates_count += 1
        
        sheet.update(values=all_values, range_name="A1")
        return updates_count
    except Exception as e:
        st.error(f"Batch update failed: {e}")
        return 0

def auto_adjust_columns(writer, sheet_name, df):
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns):
        max_len = max(
            df[col].astype(str).map(len).max(),
            len(str(col))
        ) + 2
        worksheet.set_column(idx, idx, max_len)

# --- SMART COLUMN MAPPING HELPER ---
def standardize_columns(df, entity_type='pnm'):
    """
    Renames columns from various inputs to the standard names required by the algorithm.
    Includes fix for duplicate columns preventing ValueErrors.
    """
    df.columns = df.columns.str.strip()
    
    mappings = {
        'Full Name': ['name', 'member name', 'pnm name', 'student name'],
        'Major': ['major', 'program of study'],
        'Minor': ['minor'],
        'Hometown': ['hometown', 'city', 'state'],
        'Year': ['year', 'grade', 'class', 'academic year'],
        'Hobbies': ['hobbies', 'interests', 'fun facts'],
        'College Involvement': ['college involvement', 'campus activities', 'organizations'],
        'High School Involvement': ['high school', 'hs involvement'],
        'ID': ['id', 'pnm id', 'member id', 'sorority id']
    }
    
    new_cols = {}
    used_cols = set()

    # 1. Try to find matches
    for canonical, possibilities in mappings.items():
        found = False
        for col in df.columns:
            if col in used_cols: continue
            if col.lower() == canonical.lower():
                new_cols[col] = canonical
                used_cols.add(col)
                found = True
                break
        
        if not found:
            for col in df.columns:
                if col in used_cols: continue
                if any(p in col.lower() for p in possibilities):
                    new_cols[col] = canonical
                    used_cols.add(col)
                    break
    
    # 2. Rename columns
    df = df.rename(columns=new_cols)
    
    # --- CRITICAL FIX: Remove Duplicate Columns ---
    # This prevents df['ID'] from returning a DataFrame if multiple columns mapped to 'ID'
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 3. Handle IDs
    id_col = 'PNM ID' if entity_type == 'pnm' else 'Sorority ID'
    
    # If we found a generic 'ID', rename it to the specific ID type
    if 'ID' in df.columns:
        df.rename(columns={'ID': id_col}, inplace=True)
        # Deduplicate again in case the specific ID column already existed
        df = df.loc[:, ~df.columns.duplicated()]
    
    # If ID column doesn't exist, create it
    if id_col not in df.columns:
        df[id_col] = range(1, len(df) + 1)
    else:
        # Clean existing IDs
        df[id_col] = df[id_col].replace('', np.nan)
        
        # Ensure we are working with a Series (extra safety check)
        if isinstance(df[id_col], pd.DataFrame):
            # If duplicates somehow persist, take the first one
            df[id_col] = df[id_col].iloc[:, 0]

        # Fill missing IDs
        if df[id_col].isnull().any():
            try:
                # Get max existing ID to continue sequence
                max_id = pd.to_numeric(df[id_col], errors='coerce').max()
                if pd.isna(max_id): max_id = 0
            except:
                max_id = 0
            
            missing_count = df[id_col].isnull().sum()
            fill_values = range(int(max_id) + 1, int(max_id) + 1 + missing_count)
            df.loc[df[id_col].isnull(), id_col] = fill_values

    return df

# --- MAIN PAGE ---
st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("Sorority Admin Dashboard")

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Settings & Roster", 
        "Member Information",
        "PNM Information and Rankings", 
        "View Bump Teams", 
        "View Excuses", 
        "View Prior Connections",
        "Run Matching"
    ])

    # --- TAB 1: SETTINGS ---
    with tab1:
        st.header("Event Configuration")
        current_parties = get_setting_value('B1')
        default_val = int(current_parties) if current_parties and str(current_parties).isdigit() else 4
        
        with st.form("party_config"):
            count = st.number_input("Number of Parties", 1, 50, default_val)
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
                    # Smart finding of name column
                    possible_cols = ["Full Name", "Name", "Member Name", "Member"]
                    found_col = None
                    for col in df_source.columns:
                        if any(c.lower() in col.lower() for c in possible_cols):
                            found_col = col
                            break
                    
                    if found_col:
                        names = df_source[found_col].astype(str).unique().tolist()
                        names = [n for n in names if n.strip()] 
                        if update_roster(names):
                            st.success(f"✅ Successfully synced {len(names)} members!")
                        else:
                            st.error("Failed to update Settings.")
                    else:
                        st.error(f"Could not find name column. Found columns: {df_source.columns.tolist()}")
                else:
                    st.error("'Member Information' sheet is empty.")
        
        # --- MODIFIED SECTION: CSV UPLOAD OVERRIDE ---
        with col_r2:
            st.subheader("Option B: Upload CSV")
            st.info("Upload a CSV file to strictly override the roster names.")
            file = st.file_uploader("Upload Member List (CSV)", type="csv")
            if file:
                try:
                    df_upload = pd.read_csv(file)
                    new_names = []
                    
                    # Try to find a specific name column
                    name_col = next((c for c in df_upload.columns if "name" in c.lower()), None)
                    
                    if name_col:
                        new_names = df_upload[name_col].astype(str).tolist()
                    else:
                        # Fallback: Use the first column if no 'name' header found
                        if not df_upload.empty:
                            new_names = df_upload.iloc[:, 0].astype(str).tolist()
                    
                    # Filter out 'nan' and empty strings
                    new_names = [n for n in new_names if n.lower() != 'nan' and n.strip()]
                    
                    if new_names:
                        st.success(f"Found {len(new_names)} names in CSV.")
                        if st.button("Override Roster with CSV"):
                            if update_roster(new_names): 
                                st.toast("Roster Overwritten!")
                                st.success("✅ Roster successfully overwritten with CSV data.")
                            else:
                                st.error("Failed to update settings sheet.")
                    else:
                        st.warning("No valid names found in the uploaded file.")
                        
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

    # --- TAB 3: PNM RANKINGS ---
    with tab3:
        st.header("PNM Ranking Management")
        df_votes = get_data("PNM Rankings")
        if not df_votes.empty:
            try:
                df_votes['Score'] = pd.to_numeric(df_votes['Score'], errors='coerce')
                
                # Dynamic column finding for ID
                id_col = next((c for c in df_votes.columns if 'pnm id' in c.lower()), None)
                if not id_col: id_col = next((c for c in df_votes.columns if 'id' in c.lower()), None)

                if id_col and 'Score' in df_votes.columns:
                    group_cols = [id_col]
                    name_col = next((c for c in df_votes.columns if 'pnm name' in c.lower()), None)
                    if name_col: group_cols.append(name_col)
                    
                    avg_df = df_votes.groupby(group_cols)['Score'].mean().reset_index()
                    avg_df.rename(columns={'Score': 'Calculated Average'}, inplace=True)
                    avg_df = avg_df.sort_values(by='Calculated Average', ascending=False)
                    
                    st.info(f"Processing {len(df_votes)} total votes across {len(avg_df)} unique PNMs...")
                    
                    if st.button("Sync Rankings to PNM Sheet"):
                        with st.spinner("Syncing..."):
                            # 1. Create a map of updates {ID: Score}
                            rankings_map = {}
                            for idx, row in avg_df.iterrows():
                                p_id = str(row[id_col]).strip()
                                score = round(row['Calculated Average'], 2)
                                rankings_map[p_id] = score
                            
                            # 2. Batch Update
                            count = batch_update_pnm_rankings(rankings_map)
                            
                        st.success(f"✅ Auto-synced {count} PNM rankings!")
                    
                    st.subheader("📄 Raw Ranking Data (PNM Rankings Sheet)")
                    st.dataframe(df_votes, use_container_width=True)
                else:
                    st.error("Missing 'PNM ID' or 'Score' columns.")
            except Exception as e:
                st.error(f"Error processing rankings: {e}")
        else:
            st.info("No votes found in 'PNM Rankings' sheet yet.")

        st.divider()
        st.subheader("📋 Current PNM Database")
        df_pnms = get_data("PNM Information")
        if not df_pnms.empty:
            pnm_search = st.text_input("🔍 Search PNM Database:")
            if pnm_search:
                mask = df_pnms.astype(str).apply(lambda x: x.str.contains(pnm_search, case=False).any(), axis=1)
                display_pnm_df = df_pnms[mask]
            else:
                display_pnm_df = df_pnms
            st.dataframe(display_pnm_df, use_container_width=True)
        else:
            st.info("No PNM data found.")

    # --- TAB 4: VIEW BUMP TEAMS ---
    with tab4:
        st.header("Bump Team Management")
        df_teams = get_data("Bump Teams")
        
        if not df_teams.empty:
            # Smart col detection
            id_col = next((c for c in df_teams.columns if 'team id' in c.lower() or 'id' in c.lower()), df_teams.columns[4] if len(df_teams.columns)>4 else None)
            creator_col = next((c for c in df_teams.columns if 'creator' in c.lower()), df_teams.columns[1] if len(df_teams.columns)>1 else None)
            partners_col = next((c for c in df_teams.columns if 'partner' in c.lower()), df_teams.columns[2] if len(df_teams.columns)>2 else None)
            rank_col = next((c for c in df_teams.columns if 'rank' in c.lower()), None)

            if id_col and creator_col:
                df_teams['display_label'] = df_teams.apply(
                    lambda x: f"Team {x[id_col]} | {x[creator_col]}, {x.get(partners_col, '')}", 
                    axis=1
                )
                
                # --- LAYOUT FOR UPDATES ---
                tab_single, tab_bulk = st.tabs(["Single Team Update", "Bulk Upload CSV"])
                
                # 1. SINGLE UPDATE
                with tab_single:
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        selected_label = st.selectbox("Select Team to Rank:", df_teams['display_label'].tolist())
                        selected_team_id = df_teams[df_teams['display_label'] == selected_label][id_col].values[0]
                    with col_b:
                        if rank_col:
                            current_rank = df_teams[df_teams[id_col] == selected_team_id][rank_col].values[0]
                        else:
                            current_rank = 1
                        initial_val = int(current_rank) if str(current_rank).isdigit() else 1
                        new_rank = st.number_input(f"Assign Rank:", min_value=1, value=initial_val, key="team_rank_input")
                    with col_c:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Save Team Rank"):
                            if update_team_ranking(selected_team_id, new_rank):
                                st.success(f"Rank {new_rank} assigned!")
                                st.rerun() 
                
                # 2. BULK UPLOAD
                with tab_bulk:
                    st.info("Upload a CSV with columns: `Team ID` (or `Creator Name`) and `Ranking`.")
                    team_csv = st.file_uploader("Upload Rankings CSV", type=["csv"], key="team_rank_upload")
                    
                    if team_csv:
                        if st.button("Process Bulk Update"):
                            try:
                                df_bulk = pd.read_csv(team_csv)
                                # Normalize headers
                                df_bulk.columns = df_bulk.columns.str.strip().str.lower()
                                
                                # Find columns
                                b_id_col = next((c for c in df_bulk.columns if 'id' in c), None)
                                b_name_col = next((c for c in df_bulk.columns if 'name' in c or 'creator' in c), None)
                                b_rank_col = next((c for c in df_bulk.columns if 'rank' in c), None)
                                
                                if not b_rank_col:
                                    st.error("CSV must have a 'Ranking' column.")
                                elif not (b_id_col or b_name_col):
                                    st.error("CSV must have a 'Team ID' or 'Creator Name' column.")
                                else:
                                    # Prepare map {team_id: rank}
                                    bulk_map = {}
                                    
                                    # Create lookup for names if ID is missing in CSV
                                    name_to_id = {}
                                    if not b_id_col and b_name_col:
                                        # Map Creator Name -> Team ID from existing data
                                        name_to_id = dict(zip(df_teams[creator_col].astype(str).str.strip().str.lower(), df_teams[id_col].astype(str)))
                                    
                                    for _, row in df_bulk.iterrows():
                                        rank_val = row[b_rank_col]
                                        t_id = None
                                        
                                        if b_id_col and pd.notna(row[b_id_col]):
                                            t_id = str(int(row[b_id_col])) if str(row[b_id_col]).replace('.','').isdigit() else str(row[b_id_col])
                                        elif b_name_col and pd.notna(row[b_name_col]):
                                            name_key = str(row[b_name_col]).strip().lower()
                                            t_id = name_to_id.get(name_key)
                                            
                                        if t_id:
                                            bulk_map[t_id] = rank_val
                                    
                                    if bulk_map:
                                        cnt = batch_update_team_rankings(bulk_map)
                                        st.success(f"✅ Updated {cnt} teams!")
                                        st.rerun()
                                    else:
                                        st.warning("No valid teams found to update.")
                                        
                            except Exception as e:
                                st.error(f"Error processing CSV: {e}")

            st.divider()
            st.dataframe(df_teams.drop(columns=['display_label'], errors='ignore'), use_container_width=True)
        else:
            st.info("No bump teams found yet.")

    # --- TAB 5: VIEW EXCUSES ---
    with tab5:
        if st.button("🔄 Refresh Excuses"): st.rerun()
        df_excuses = get_data("Party Excuses")
        if not df_excuses.empty:
            st.dataframe(df_excuses, use_container_width=True)
        else:
            st.info("No excuses found.")

    # --- TAB 6: VIEW PRIOR CONNECTIONS ---
    with tab6:
        st.header("Prior PNM Connections Log")
        if st.button("🔄 Refresh Connections"): st.rerun()
        df_connections = get_data("Prior Connections")
        if not df_connections.empty:
            st.dataframe(df_connections, use_container_width=True)
        else:
            st.info("No prior connections found.")

    # --- TAB 7: RUN MATCHING ---
    with tab7:
        st.subheader("Matching Configuration")
        c1, c2 = st.columns(2)
        with c1:
            num_pnm_matches_needed = st.number_input("PNM Matches per Bump Team (Capacity)", min_value=1, value=5, step=1, key="match_cap")
        with c2:
            num_rounds_party = st.number_input("Rounds per Party", min_value=1, value=4, step=1, key="match_rounds")
        bump_order_set = st.radio("Is the bump order set?", ["yes", "no"], index=1, key="match_order")

        if st.button("Run Matching Algorithm"):
            with st.spinner("Fetching data and initializing models..."):
                # LOAD SETTINGS (Parties)
                parties_val = get_setting_value('B1')
                num_of_parties = int(parties_val) if parties_val and str(parties_val).isdigit() else 4
                st.info(f"Using {num_of_parties} parties (configured in Settings).")
                
                # LOAD ACTIVE ROSTER
                active_roster = get_active_roster_names()
                if active_roster:
                    st.info(f"Loaded {len(active_roster)} active members from Roster (Settings/CSV).")
                else:
                    st.warning("No active roster found. Attempting to match with all members in database.")

                # LOAD DATA
                bump_teams = get_data("Bump Teams")
                party_excuses = get_data("Party Excuses")
                member_interest = get_data("Member Information")
                member_pnm_no_match = get_data("Prior Connections")
                pnm_intial_interest = get_data("PNM Information")

                if pnm_intial_interest.empty or bump_teams.empty:
                    st.error("Missing critical data (PNM Information or Bump Teams).")
                    st.stop()

                # --- 0. SMART COLUMN NORMALIZATION (THE FIX) ---
                pnm_working = standardize_columns(pnm_intial_interest.copy(), entity_type='pnm')
                member_interest = standardize_columns(member_interest, entity_type='member')
                
                # --- FILTER MEMBERS BASED ON ROSTER ---
                if active_roster and not member_interest.empty:
                    # Normalize for case-insensitive comparison
                    active_set = set(n.strip().lower() for n in active_roster)
                    # Filter: Keep members whose normalized name is in the active set
                    member_interest = member_interest[
                        member_interest['Full Name'].astype(str).str.strip().str.lower().isin(active_set)
                    ]
                    st.success(f"Filtered member database to {len(member_interest)} matching entries from active roster.")

                if member_interest.empty:
                    st.error("No member information matched the active roster. Cannot proceed.")
                    st.stop()

                # Also normalize excuses and connections lightly to find names
                if not party_excuses.empty:
                    party_excuses.columns = party_excuses.columns.str.strip()
                if not member_pnm_no_match.empty:
                    member_pnm_no_match.columns = member_pnm_no_match.columns.str.strip()

                # --- 1. PRE-PROCESSING ---
                
                # Assignments
                pnms_per_party = int(np.ceil(len(pnm_working) / num_of_parties))
                party_assignments = np.tile(np.arange(1, num_of_parties + 1), pnms_per_party)
                party_assignments = party_assignments[:len(pnm_working)]
                np.random.seed(42)
                np.random.shuffle(party_assignments)
                pnm_working['Party'] = party_assignments
                
                # --- GEOCODING ---
                url_geo = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"
                try:
                    ref_df = pd.read_csv(url_geo)
                    ref_df['MATCH_KEY'] = (ref_df['CITY'] + ", " + ref_df['STATE_CODE']).str.upper()
                    ref_df = ref_df.drop_duplicates(subset=['MATCH_KEY'], keep='first')
                    city_coords_map = {key: [lat, lon] for key, lat, lon in zip(ref_df['MATCH_KEY'], ref_df['LATITUDE'], ref_df['LONGITUDE'])}
                    ALL_CITY_KEYS = list(city_coords_map.keys())
                except:
                    city_coords_map = {}
                    ALL_CITY_KEYS = []

                def get_coords_offline(hometown_str):
                    if not isinstance(hometown_str, str): return None, None
                    key = hometown_str.strip().upper()
                    if key in city_coords_map: return city_coords_map[key]
                    matches = difflib.get_close_matches(key, ALL_CITY_KEYS, n=1, cutoff=0.8)
                    if matches: return city_coords_map[matches[0]]
                    return None, None

                # --- CLUSTERING ---
                all_coords = []
                geo_tracker = []
                
                # Members
                for idx, row in member_interest.iterrows():
                    if 'Hometown' in row:
                        lat, lon = get_coords_offline(row['Hometown'])
                        if lat:
                            all_coords.append([radians(lat), radians(lon)])
                            geo_tracker.append({'type': 'mem', 'id': row['Sorority ID'], 'hometown': row['Hometown']})
                
                # PNMs
                for idx, row in pnm_working.iterrows():
                    if 'Hometown' in row:
                        lat, lon = get_coords_offline(row['Hometown'])
                        if lat:
                            all_coords.append([radians(lat), radians(lon)])
                            geo_tracker.append({'type': 'pnm', 'id': row['PNM ID'], 'hometown': row['Hometown']})

                mem_geo_tags = {}
                pnm_geo_tags = {}

                if all_coords:
                    dist_matrix = haversine_distances(all_coords, all_coords) * 3958.8
                    geo_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=30, metric='precomputed', linkage='single')
                    geo_labels = geo_clustering.fit_predict(dist_matrix)
                    geo_groups = {}
                    for i, label in enumerate(geo_labels):
                        if label not in geo_groups: geo_groups[label] = []
                        geo_groups[label].append(geo_tracker[i]['hometown'])
                    for i, label in enumerate(geo_labels):
                        group_name = geo_groups[label][0]
                        tracker = geo_tracker[i]
                        if tracker['type'] == 'mem': mem_geo_tags[tracker['id']] = group_name
                        else: pnm_geo_tags[tracker['id']] = group_name

                # Semantic
                model = load_model()
                all_terms_list = []
                def extract_terms(row, cols):
                    text_parts = [str(row.get(c, '')).lower() for c in cols if c in row]
                    combined = ", ".join([p for p in text_parts if p != 'nan' and p.strip() != ''])
                    return [t.strip() for t in combined.split(',') if t.strip()]

                cols_to_extract = ['Major', 'Minor', 'Hobbies', 'College Involvement', 'High School Involvement']

                mem_interest_map = []
                for idx, row in member_interest.iterrows():
                    terms = extract_terms(row, cols_to_extract)
                    for term in terms:
                        all_terms_list.append(term)
                        mem_interest_map.append({'id': row['Sorority ID'], 'term': term})
                        
                pnm_interest_map = []
                for idx, row in pnm_working.iterrows():
                    terms = extract_terms(row, cols_to_extract)
                    for term in terms:
                        all_terms_list.append(term)
                        pnm_interest_map.append({'id': row['PNM ID'], 'term': term})

                term_to_group = {}
                if all_terms_list:
                    unique_terms = list(set(all_terms_list))
                    embeddings = model.encode(unique_terms)
                    sem_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='cosine', linkage='average')
                    sem_labels = sem_clustering.fit_predict(embeddings)
                    temp_map = {}
                    for term, label in zip(unique_terms, sem_labels):
                        if label not in temp_map: temp_map[label] = []
                        temp_map[label].append(term)
                    for label, terms in temp_map.items():
                        attr_name = min(terms, key=len)
                        for term in terms: term_to_group[term] = attr_name

                # Finalize Attributes
                VALID_YEARS = ["Freshman", "Sophomore", "Junior", "Senior"]
                def get_year_tag(year_val):
                    if pd.isna(year_val): return None
                    raw = str(year_val).strip()
                    matches = difflib.get_close_matches(raw, VALID_YEARS, n=1, cutoff=0.6)
                    return matches[0] if matches else raw.title()

                mem_final_attrs = {row['Sorority ID']: set() for _, row in member_interest.iterrows()}
                for idx, row in member_interest.iterrows():
                    pid = row['Sorority ID']
                    yt = get_year_tag(row.get('Year'))
                    if yt: mem_final_attrs[pid].add(yt)
                    if pid in mem_geo_tags: mem_final_attrs[pid].add(mem_geo_tags[pid])
                for entry in mem_interest_map:
                    if entry['term'] in term_to_group: mem_final_attrs[entry['id']].add(term_to_group[entry['term']])
                member_interest['attributes_for_matching'] = member_interest['Sorority ID'].map(lambda x: ", ".join(mem_final_attrs.get(x, set())))

                pnm_final_attrs = {row['PNM ID']: set() for _, row in pnm_working.iterrows()}
                for idx, row in pnm_working.iterrows():
                    pid = row['PNM ID']
                    yt = get_year_tag(row.get('Year'))
                    if yt: pnm_final_attrs[pid].add(yt)
                    if pid in pnm_geo_tags: pnm_final_attrs[pid].add(pnm_geo_tags[pid])
                for entry in pnm_interest_map:
                    if entry['term'] in term_to_group: pnm_final_attrs[entry['id']].add(term_to_group[entry['term']])
                pnm_working['attributes_for_matching'] = pnm_working['PNM ID'].map(lambda x: ", ".join(pnm_final_attrs.get(x, set())))

                # 2. MATCHING LOGIC SETUP
                # Handling Party Excuses
                # Find column for excuses
                excuse_col = next((c for c in party_excuses.columns if "party" in c.lower() and "attend" in c.lower()), None)
                if excuse_col:
                    party_excuses[excuse_col] = party_excuses[excuse_col].apply(
                        lambda x: [int(i) for i in re.findall(r'\d+', str(x))] if pd.notnull(x) else []
                    )
                    party_excuses_exploded = party_excuses.explode(excuse_col)
                else:
                    party_excuses_exploded = pd.DataFrame(columns=['Member Name', 'Party'])

                # Handling No Matches
                no_match_pairs = set()
                if not member_pnm_no_match.empty:
                    # Try to find Member Name and PNM Name columns dynamically
                    m_col = next((c for c in member_pnm_no_match.columns if 'member' in c.lower() and 'name' in c.lower()), None)
                    p_col = next((c for c in member_pnm_no_match.columns if 'pnm' in c.lower() and 'name' in c.lower()), None)
                    if m_col and p_col:
                        no_match_pairs = {
                            (row[m_col], row[p_col])
                            for row in member_pnm_no_match.to_dict('records')
                            if pd.notna(row.get(m_col)) and pd.notna(row.get(p_col))
                        }
                
                member_attr_cache = {
                    row['Sorority ID']: set(str(row.get('attributes_for_matching', '')).split(', '))
                    if row.get('attributes_for_matching') else set()
                    for row in member_interest.to_dict('records')
                }
                
                # Robust name map
                name_to_id_map = {}
                for idx, row in member_interest.iterrows():
                      name = row.get('Full Name')
                      if name: name_to_id_map[name] = row.get('Sorority ID')

                all_member_traits = member_interest['attributes_for_matching'].str.split(', ').explode()
                trait_freq = all_member_traits.value_counts()
                trait_weights = (len(member_interest) / trait_freq).to_dict()
                strength_bonus_map = {1: 1.5, 2: 1.0, 3: 0.5, 4: 0.0}
                
            # --- EXECUTION LOOP ---
            st.write("---")
            st.write("### Results")
            progress_bar = st.progress(0)
            
            results_buffers = []

            # Internal Function Definitions
            def run_internal_rotation(assignment_map, team_list, method='flow'):
                rotation_output = []
                actual_rounds = 1 if bump_order_set == 'yes' else num_rounds_party

                for t_idx, assigned_pnms in assignment_map.items():
                    if not assigned_pnms: continue
                    team_data = next((t for t in team_list if t['t_idx'] == t_idx), None)
                    if not team_data: continue

                    raw_rgl = team_data['row_data'].get('RGL', '')
                    team_rgl_name = "" if pd.isna(raw_rgl) or str(raw_rgl).lower() == 'nan' else str(raw_rgl).strip()

                    valid_members = []
                    for m_id, m_name in zip(team_data['member_ids'], team_data['members']):
                        if m_id: valid_members.append({'id': m_id, 'name': m_name})

                    history = set()

                    for round_num in range(1, actual_rounds + 1):
                        if round_num == 1 and team_rgl_name:
                            active_members = [m for m in valid_members if m['name'].strip() != team_rgl_name]
                        else:
                            active_members = valid_members
                        
                        must_allow_repeats = round_num > len(active_members)

                        if method == 'flow':
                            sub_G = nx.DiGraph()
                            sub_s, sub_t = 's', 't'
                            req = len(assigned_pnms)
                            sub_G.add_node(sub_s, demand=-req)
                            sub_G.add_node(sub_t, demand=req)

                            for p in assigned_pnms:
                                sub_G.add_edge(sub_s, f"p_{p['p_id']}", capacity=1, weight=0)
                            for m in active_members:
                                sub_G.add_edge(f"m_{m['id']}", sub_t, capacity=1, weight=0)

                            for p in assigned_pnms:
                                for m in active_members:
                                    is_repeat = (p['p_id'], m['id']) in history
                                    if is_repeat and not must_allow_repeats: continue
                                    m_attrs = member_attr_cache.get(m['id'], set())
                                    shared = p['p_attrs'].intersection(m_attrs)
                                    score = sum(trait_weights.get(t, 1.0) for t in shared)
                                    base_cost = int((1/(1+score))*10000)
                                    final_cost = base_cost + 50000 if is_repeat else base_cost
                                    reason = ", ".join(shared) if shared else "Rotation"
                                    if is_repeat: reason += " (Repeat)"
                                    sub_G.add_edge(f"p_{p['p_id']}", f"m_{m['id']}", capacity=1, weight=final_cost, reason=reason)

                            try:
                                sub_flow = nx.min_cost_flow(sub_G)
                                for p in assigned_pnms:
                                    p_node = f"p_{p['p_id']}"
                                    if p_node in sub_flow:
                                        for tgt, flow in sub_flow[p_node].items():
                                            if flow > 0 and tgt != sub_t:
                                                m_id_ex = int(tgt.replace("m_", ""))
                                                m_name = next((m['name'] for m in valid_members if m['id'] == m_id_ex), "Unknown")
                                                edge_d = sub_G.get_edge_data(p_node, tgt)
                                                raw_weight = edge_d.get('weight', 10000)
                                                if raw_weight > 40000: raw_weight -= 50000
                                                calc_cost = raw_weight / 10000.0
                                                rotation_output.append({
                                                    'Round': round_num, 'Team ID': t_idx, 'Team Members': team_data['joined_names'],
                                                    'PNM ID': p['p_id'], 'PNM Name': p['p_name'], 'Matched Member': m_name,
                                                    'Match Cost': round(calc_cost, 4), 'Reason': f"Common: {edge_d.get('reason')}"
                                                })
                                                history.add((p['p_id'], m_id_ex))
                            except nx.NetworkXUnfeasible:
                                 fail_reason = "Unfeasible (Capacity)"
                                 rotation_output.append({'Round': round_num, 'Team ID': t_idx, 'PNM Name': "FLOW FAIL", 'Reason': fail_reason})

                        elif method == 'greedy':
                            candidates = []
                            for p in assigned_pnms:
                                for m in active_members:
                                    is_repeat = (p['p_id'], m['id']) in history
                                    if is_repeat and not must_allow_repeats: continue
                                    m_attrs = member_attr_cache.get(m['id'], set())
                                    shared = p['p_attrs'].intersection(m_attrs)
                                    score = sum(trait_weights.get(t, 1.0) for t in shared)
                                    final_score = score - 1000 if is_repeat else score
                                    reason = ", ".join(shared) if shared else "Rotation"
                                    if is_repeat: reason += " (Repeat)"
                                    candidates.append((final_score, p, m, reason, is_repeat))

                            candidates.sort(key=lambda x: x[0], reverse=True)
                            round_pnm_done, round_mem_done = set(), set()

                            for sc, p, m, rs, is_rep in candidates:
                                if p['p_id'] not in round_pnm_done and m['id'] not in round_mem_done:
                                    real_score = sc + 1000 if is_rep else sc
                                    calc_cost = 1.0 / (1.0 + real_score)
                                    rotation_output.append({
                                        'Round': round_num, 'Team ID': t_idx, 'Team Members': team_data['joined_names'],
                                        'PNM ID': p['p_id'], 'PNM Name': p['p_name'], 'Matched Member': m['name'],
                                        'Match Cost': round(calc_cost, 4), 'Reason': f"Common: {rs}" if real_score > 0 else "Greedy Fill"
                                    })
                                    round_pnm_done.add(p['p_id']); round_mem_done.add(m['id']); history.add((p['p_id'], m['id']))

                            for p in assigned_pnms:
                                if p['p_id'] not in round_pnm_done:
                                    for m in active_members:
                                        if m['id'] not in round_mem_done:
                                            m_attrs = member_attr_cache.get(m['id'], set())
                                            shared = p['p_attrs'].intersection(m_attrs)
                                            score = sum(trait_weights.get(t, 1.0) for t in shared)
                                            rotation_output.append({
                                                'Round': round_num, 'Team ID': t_idx, 'Team Members': team_data['joined_names'],
                                                'PNM ID': p['p_id'], 'PNM Name': p['p_name'], 'Matched Member': m['name'],
                                                'Match Cost': round(1.0 / (1.0 + score), 4), 'Reason': "Forced Greedy Fill"
                                            })
                                            round_pnm_done.add(p['p_id']); round_mem_done.add(m['id']); history.add((p['p_id'], m['id']))
                                            break
                return rotation_output

            def generate_bump_instructions(rotation_data):
                if not rotation_data: return []
                df = pd.DataFrame(rotation_data)
                if df.empty or 'Matched Member' not in df.columns: return []
                df = df.sort_values(by=['Team ID', 'PNM ID', 'Round'])
                df['Person_To_Bump'] = df.groupby(['Team ID', 'PNM ID'])['Matched Member'].shift(1)
                instructions = df[df['Person_To_Bump'].notna()].copy()
                instructions['At End Of Round'] = instructions['Round'] - 1
                output = instructions[['Matched Member', 'At End Of Round', 'Person_To_Bump', 'PNM Name']].rename(columns={
                    'Matched Member': 'Member (You)', 'Person_To_Bump': 'Go Bump This Person', 'PNM Name': 'Who is with PNM'
                })
                return output.sort_values(by=['Member (You)', 'At End Of Round']).to_dict('records')

            # --- PARTY LOOP ---
            for party in range(1, num_of_parties + 1):
                pnms_df = pnm_working[pnm_working['Party'] == party].copy()
                if pnms_df.empty:
                    progress_bar.progress(party / num_of_parties)
                    continue
                    
                pnm_list = []
                for i, row in enumerate(pnms_df.to_dict('records')):
                    p_attrs = set(str(row.get('attributes_for_matching', '')).split(', '))
                    try:
                        p_rank = float(row.get("Average Recruit Rank", 1.0))
                    except: p_rank = 1.0
                    
                    pnm_list.append({
                        'idx': i, 'id': row['PNM ID'], 'name': row['Full Name'],
                        'attrs': p_attrs, 'rank': p_rank, 'bonus': 0.75 * (p_rank - 1),
                        'node_id': f"p_{i}"
                    })

                # Fix Excuses Logic with standardized names
                party_excused_names = set()
                if not party_excuses_exploded.empty:
                    # Find names in excuses (first column typically)
                    name_col_ex = party_excuses_exploded.columns[1] if len(party_excuses_exploded.columns) > 1 else party_excuses_exploded.columns[0]
                    # Find party column in excuses
                    if excuse_col:
                        party_excused_names = set(party_excuses_exploded[party_excuses_exploded[excuse_col] == party][name_col_ex])
                
                team_list = []
                for raw_idx, row in enumerate(bump_teams.to_dict('records')):
                    submitter = row["Creator Name"]
                    partners_str = str(row.get("Bump Partners", ""))
                    partners = [p.strip() for p in re.split(r'[,;]\s*', partners_str) if p.strip()] if partners_str.lower() != 'nan' else []
                    current_members = [submitter] + partners
                    
                    if any(m in party_excused_names for m in current_members):
                        continue 
                    
                    try:
                        t_rank = int(float(row.get("Ranking", 4)))
                    except:
                        t_rank = 4

                    team_list.append({
                        't_idx': len(team_list), 'members': current_members, 'team_size': len(current_members),
                        'member_ids': [name_to_id_map.get(m) for m in current_members],
                        'joined_names': ", ".join(current_members),
                        'bonus': strength_bonus_map.get(t_rank, 0.0),
                        'node_id': f"t_{len(team_list)}", 'row_data': row
                    })

                # Calculate Potential Pairs
                potential_pairs = []
                for p_data in pnm_list:
                    for t_data in team_list:
                        if any((m, p_data['name']) in no_match_pairs for m in t_data['members']): continue
                        
                        score = 0
                        reasons_list = []
                        for m_id, m_name in zip(t_data['member_ids'], t_data['members']):
                            if m_id:
                                shared = p_data['attrs'].intersection(member_attr_cache.get(m_id, set()))
                                if shared:
                                    for trait in shared: score += trait_weights.get(trait, 1.0)
                                    reasons_list.append(f"{p_data['name']} + {m_name} ({', '.join(shared)})")
                        
                        final_cost = 1 / (1 + score + t_data['bonus'] + p_data['bonus'])
                        potential_pairs.append({
                            'p_id': p_data['id'], 'p_name': p_data['name'], 'p_attrs': p_data['attrs'],
                            't_idx': t_data['t_idx'], 'team_size': t_data['team_size'],
                            'p_node': p_data['node_id'], 't_node': t_data['node_id'],
                            'cost': final_cost, 'pnm_rank': p_data['rank'],
                            'team_members': t_data['joined_names'], 'reasons': "; ".join(reasons_list)
                        })
                
                potential_pairs.sort(key=lambda x: (x['cost'], -x['pnm_rank']))

                # --- PHASE A: GLOBAL FLOW ---
                G = nx.DiGraph()
                source, sink, no_match_node = 'source', 'sink', 'dummy_nomatch'
                total_flow = len(pnm_list)
                
                G.add_node(source, demand=-total_flow)
                G.add_node(sink, demand=total_flow)
                G.add_edge(no_match_node, sink, capacity=total_flow, weight=0)
                
                for p in pnm_list:
                    G.add_edge(source, p['node_id'], capacity=1, weight=0)
                    G.add_edge(p['node_id'], no_match_node, capacity=1, weight=1000000)
                for t in team_list:
                    G.add_edge(t['node_id'], sink, capacity=num_pnm_matches_needed, weight=0)
                for pair in potential_pairs:
                    G.add_edge(pair['p_node'], pair['t_node'], capacity=1, weight=int(pair['cost'] * 10000))
                
                global_flow_results = []
                assignments_map_flow = {t['t_idx']: [] for t in team_list}
                
                try:
                    flow_dict = nx.min_cost_flow(G)
                    pair_lookup = {(p['p_node'], p['t_node']): p for p in potential_pairs}
                    pnm_ids_with_potential = {p['p_id'] for p in potential_pairs}
                    
                    for p_data in pnm_list:
                        p_node = p_data['node_id']
                        if p_node in flow_dict:
                            for t_node, flow in flow_dict[p_node].items():
                                if flow > 0:
                                    if t_node == no_match_node:
                                        reason = "Conflict List" if p_data['id'] not in pnm_ids_with_potential else "Capacity Full"
                                        global_flow_results.append({
                                            'PNM ID': p_data['id'], 'PNM Name': p_data['name'],
                                            'Bump Team Members': "NO MATCH", 'Match Cost': None, 'Reason': reason
                                        })
                                    else:
                                        match_info = pair_lookup.get((p_node, t_node))
                                        if match_info:
                                            global_flow_results.append({
                                                'PNM ID': p_data['id'], 'PNM Name': p_data['name'],
                                                'Bump Team Members': match_info['team_members'],
                                                'Match Cost': round(match_info['cost'], 4), 'Reason': match_info['reasons']
                                            })
                                            assignments_map_flow[match_info['t_idx']].append(match_info)
                except nx.NetworkXUnfeasible:
                    st.error(f"Optimization failed for Party {party}")

                # --- PHASE A2: GLOBAL GREEDY ---
                global_greedy_results = []
                assignments_map_greedy = {t['t_idx']: [] for t in team_list}
                matched_pnm_ids = set()
                team_counts = {t['t_idx']: 0 for t in team_list}

                for pair in potential_pairs:
                    if pair['p_id'] not in matched_pnm_ids:
                        if team_counts[pair['t_idx']] < num_pnm_matches_needed:
                            matched_pnm_ids.add(pair['p_id'])
                            team_counts[pair['t_idx']] += 1
                            global_greedy_results.append({
                                'PNM ID': pair['p_id'], 'PNM Name': pair['p_name'],
                                'Bump Team Members': pair['team_members'], 'Match Cost': round(pair['cost'], 4),
                                'Reason': pair['reasons']
                            })
                            assignments_map_greedy[pair['t_idx']].append(pair)
                
                for p_data in pnm_list:
                    if p_data['id'] not in matched_pnm_ids:
                        global_greedy_results.append({
                            'PNM ID': p_data['id'], 'PNM Name': p_data['name'],
                            'Bump Team Members': "NO MATCH", 'Match Cost': None, 'Reason': "No match (Greedy/Capacity)"
                        })

                # --- PHASE B: INTERNAL ROTATIONS ---
                internal_flow_results = run_internal_rotation(assignments_map_flow, team_list, method='flow')
                internal_greedy_results = run_internal_rotation(assignments_map_greedy, team_list, method='greedy')

                # --- PHASE C: BUMP INSTRUCTIONS ---
                bump_instruct_flow = generate_bump_instructions(internal_flow_results)
                bump_instruct_greedy = generate_bump_instructions(internal_greedy_results)

                # --- EXPORT TO EXCEL ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # 1. Summary
                    df_glob_flow = pd.DataFrame(global_flow_results)
                    df_glob_greedy = pd.DataFrame(global_greedy_results)
                    
                    flow_costs = df_glob_flow['Match Cost'].dropna()
                    greedy_costs = df_glob_greedy['Match Cost'].dropna()
                    summary_df = pd.DataFrame({
                        'Metric': ['Total Matching Cost', 'Avg Cost', 'Min Cost', 'Max Cost', 'Standard Deviation'],
                        'Global Flow': [
                            flow_costs.sum(), flow_costs.mean(), flow_costs.min(), flow_costs.max(),
                            flow_costs.std() if len(flow_costs) > 1 else 0.0
                        ],
                        'Global Greedy': [
                            greedy_costs.sum(), greedy_costs.mean(), greedy_costs.min(), greedy_costs.max(),
                            greedy_costs.std() if len(greedy_costs) > 1 else 0.0
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    auto_adjust_columns(writer, "Summary", summary_df)

                    # 2. Global Results
                    df_glob_flow.to_excel(writer, sheet_name="Global_Matches_Flow", index=False)
                    auto_adjust_columns(writer, "Global_Matches_Flow", df_glob_flow)
                    
                    df_glob_greedy.to_excel(writer, sheet_name="Global_Matches_Greedy", index=False)
                    auto_adjust_columns(writer, "Global_Matches_Greedy", df_glob_greedy)

                    # 3. Rotations & Bump
                    df_rot_flow = pd.DataFrame(internal_flow_results)
                    df_rot_greedy = pd.DataFrame(internal_greedy_results)
                    
                    if 'Team ID' in df_rot_flow.columns: df_rot_flow = df_rot_flow.drop(columns=['Team ID'])
                    if 'Team ID' in df_rot_greedy.columns: df_rot_greedy = df_rot_greedy.drop(columns=['Team ID'])

                    df_bump_flow = pd.DataFrame(bump_instruct_flow)
                    df_bump_greedy = pd.DataFrame(bump_instruct_greedy)

                    if not df_rot_flow.empty:
                        sheet_name = "Round_1_Flow" if bump_order_set == 'yes' else "Rotation_Flow"
                        df_to_write = df_rot_flow.copy()
                        if bump_order_set == 'yes' and 'Round' in df_to_write.columns:
                            df_to_write = df_to_write.drop(columns=['Round'])

                        df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                        auto_adjust_columns(writer, sheet_name, df_to_write)
                        
                        if not df_bump_flow.empty and bump_order_set == 'no':
                             df_bump_flow.to_excel(writer, sheet_name="Bump_Logistics_Flow", index=False)
                             auto_adjust_columns(writer, "Bump_Logistics_Flow", df_bump_flow)

                    if not df_rot_greedy.empty:
                        sheet_name = "Round_1_Greedy" if bump_order_set == 'yes' else "Rotation_Greedy"
                        df_to_write = df_rot_greedy.copy()
                        if bump_order_set == 'yes' and 'Round' in df_to_write.columns:
                            df_to_write = df_to_write.drop(columns=['Round'])

                        df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                        auto_adjust_columns(writer, sheet_name, df_to_write)

                        if not df_bump_greedy.empty and bump_order_set == 'no':
                             df_bump_greedy.to_excel(writer, sheet_name="Bump_Logistics_Greedy", index=False)
                             auto_adjust_columns(writer, "Bump_Logistics_Greedy", df_bump_greedy)

                results_buffers.append({
                    "party": party,
                    "buffer": output.getvalue()
                })
                
                progress_bar.progress(party / num_of_parties)

            # --- DOWNLOADS ---
            st.success("Matching Complete!")
            if not results_buffers:
                st.warning("No matches generated. Check inputs.")
                
            for res in results_buffers:
                st.download_button(
                    label=f"Download Results for Party {res['party']}",
                    data=res['buffer'],
                    file_name=f"Party_{res['party']}_Matches.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
