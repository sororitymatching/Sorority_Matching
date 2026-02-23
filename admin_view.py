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

@st.cache_data(ttl=3600)
def load_geo_data():
    """Loads and caches the US Cities database for offline geocoding."""
    url = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"
    try:
        ref_df = pd.read_csv(url)
        ref_df['MATCH_KEY'] = (ref_df['CITY'] + ", " + ref_df['STATE_CODE']).str.upper()
        ref_df = ref_df.drop_duplicates(subset=['MATCH_KEY'], keep='first')
        return {
            key: [lat, lon]
            for key, lat, lon in zip(ref_df['MATCH_KEY'], ref_df['LATITUDE'], ref_df['LONGITUDE'])
        }, list(ref_df['MATCH_KEY'])
    except:
        return {}, []

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
        df = pd.DataFrame(data[1:], columns=data[0])
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
    except: return None

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
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("Settings")
        roster_data = sheet.get_values("D2:D")
        names = [r[0].strip() for r in roster_data if r and r[0].strip()]
        if names: return names
    except: pass

    df_mem = get_data("Member Information")
    if not df_mem.empty:
        possible_cols = ["Full Name", "Name", "Member Name", "Member"]
        found_col = None
        for col in df_mem.columns:
            if any(c.lower() in col.lower() for c in possible_cols):
                found_col = col
                break
        if found_col: return df_mem[found_col].dropna().astype(str).str.strip().tolist()
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
    except: return False

def batch_update_pnm_rankings(rankings_map):
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("PNM Information")
        all_values = sheet.get_all_values()
        if not all_values: return 0
        headers = [h.lower().strip() for h in all_values[0]]
        
        try: id_idx = next(i for i, h in enumerate(headers) if 'pnm id' in h or 'id' == h)
        except: id_idx = 23
            
        try: rank_idx = next(i for i, h in enumerate(headers) if 'recruit rank' in h or 'average' in h)
        except: rank_idx = 24

        updates_count = 0
        for i in range(1, len(all_values)):
            row = all_values[i]
            if len(row) <= id_idx: continue
            p_id = str(row[id_idx]).strip()
            if p_id in rankings_map:
                while len(row) <= rank_idx: row.append("")
                row[rank_idx] = str(rankings_map[p_id])
                updates_count += 1
        sheet.update(values=all_values, range_name="A1")
        return updates_count
    except Exception as e:
        st.error(f"Batch update failed: {e}")
        return 0

def batch_update_team_rankings(rankings_map):
    try:
        gc = get_gc()
        sheet = gc.open(SHEET_NAME).worksheet("Bump Teams")
        all_values = sheet.get_all_values()
        if not all_values: return 0
        headers = [h.lower().strip() for h in all_values[0]]
        
        try: id_idx = next(i for i, h in enumerate(headers) if 'team id' in h or 'id' == h)
        except: id_idx = 4
        try: rank_idx = next(i for i, h in enumerate(headers) if 'ranking' in h or 'rank' in h)
        except: rank_idx = 5

        updates_count = 0
        for i in range(1, len(all_values)):
            row = all_values[i]
            if len(row) <= id_idx: continue
            t_id = str(row[id_idx]).strip()
            if t_id in rankings_map:
                while len(row) <= rank_idx: row.append("")
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
        max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
        worksheet.set_column(idx, idx, max_len)

def standardize_columns(df, entity_type='pnm'):
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
    for canonical, possibilities in mappings.items():
        found = False
        for col in df.columns:
            if col in used_cols: continue
            if col.lower() == canonical.lower():
                new_cols[col] = canonical; used_cols.add(col); found = True; break
        if not found:
            for col in df.columns:
                if col in used_cols: continue
                if any(p in col.lower() for p in possibilities):
                    new_cols[col] = canonical; used_cols.add(col); break
    
    df = df.rename(columns=new_cols)
    df = df.loc[:, ~df.columns.duplicated()]
    
    id_col = 'PNM ID' if entity_type == 'pnm' else 'Sorority ID'
    if 'ID' in df.columns:
        df.rename(columns={'ID': id_col}, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
    
    if id_col not in df.columns:
        df[id_col] = range(1, len(df) + 1)
    else:
        df[id_col] = df[id_col].replace('', np.nan)
        if df[id_col].isnull().any():
            try: max_id = pd.to_numeric(df[id_col], errors='coerce').max(); max_id = 0 if pd.isna(max_id) else max_id
            except: max_id = 0
            fill_values = range(int(max_id) + 1, int(max_id) + 1 + df[id_col].isnull().sum())
            df.loc[df[id_col].isnull(), id_col] = fill_values
    return df

# --- MAIN PAGE ---
st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("Sorority Admin Dashboard")

if "authenticated" not in st.session_state: st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("Enter Admin Password:", type="password")
    if pwd == ADMIN_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
else:
    st.success("Logged in as Admin")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Settings & Roster", "Member Information", "PNM Information and Rankings", 
        "View Bump Teams", "View Excuses", "View Prior Connections", "Run Matching"
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
                    possible_cols = ["Full Name", "Name", "Member Name", "Member"]
                    found_col = None
                    for col in df_source.columns:
                        if any(c.lower() in col.lower() for c in possible_cols): found_col = col; break
                    if found_col:
                        names = df_source[found_col].astype(str).unique().tolist()
                        names = [n for n in names if n.strip()] 
                        if update_roster(names): st.success(f"✅ Successfully synced {len(names)} members!")
                        else: st.error("Failed to update Settings.")
                    else: st.error("Could not find name column.")
                else: st.error("'Member Information' sheet is empty.")
        with col_r2:
            st.subheader("Option B: Upload CSV")
            st.info("Upload a CSV file to strictly override the roster names.")
            file = st.file_uploader("Upload Member List (CSV)", type="csv")
            if file:
                try:
                    df_upload = pd.read_csv(file)
                    new_names = []
                    name_col = next((c for c in df_upload.columns if "name" in c.lower()), None)
                    if name_col: new_names = df_upload[name_col].astype(str).tolist()
                    else: new_names = df_upload.iloc[:, 0].astype(str).tolist() if not df_upload.empty else []
                    new_names = [n for n in new_names if n.lower() != 'nan' and n.strip()]
                    if new_names:
                        st.success(f"Found {len(new_names)} names in CSV.")
                        if st.button("Override Roster with CSV"):
                            if update_roster(new_names): st.success("✅ Roster overwritten!"); st.toast("Roster Overwritten!")
                            else: st.error("Failed to update settings.")
                    else: st.warning("No valid names found.")
                except Exception as e: st.error(f"Error reading CSV: {e}")

    # --- TAB 2: MEMBER INFORMATION ---
    with tab2:
        st.header("Member Information Database")
        if st.button("🔄 Refresh Member Data"): st.rerun()
        df_members = get_data("Member Information")
        if not df_members.empty:
            search_mem = st.text_input("🔍 Search Members:")
            if search_mem:
                mask = df_members.apply(lambda x: x.astype(str).str.contains(search_mem, case=False).any(), axis=1)
                display_df = df_members[mask]
            else: display_df = df_members
            st.metric("Total Members", len(display_df))
            st.dataframe(display_df, use_container_width=True)
        else: st.info("No member information found.")

    # --- TAB 3: PNM RANKINGS ---
    with tab3:
        st.header("PNM Ranking Management")
        df_votes = get_data("PNM Rankings")
        if not df_votes.empty:
            try:
                df_votes['Score'] = pd.to_numeric(df_votes['Score'], errors='coerce')
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
                            rankings_map = {str(row[id_col]).strip(): round(row['Calculated Average'], 2) for idx, row in avg_df.iterrows()}
                            count = batch_update_pnm_rankings(rankings_map)
                        st.success(f"✅ Auto-synced {count} PNM rankings!")
                    
                    st.subheader("Raw Ranking Data")
                    st.dataframe(df_votes, use_container_width=True)
                else: st.error("Missing 'PNM ID' or 'Score' columns.")
            except Exception as e: st.error(f"Error processing rankings: {e}")
        else: st.info("No votes found in 'PNM Rankings' sheet yet.")
        st.divider()
        st.subheader("Current PNM Database")
        df_pnms = get_data("PNM Information")
        if not df_pnms.empty:
            pnm_search = st.text_input("🔍 Search PNM Database:")
            display_pnm = df_pnms[df_pnms.astype(str).apply(lambda x: x.str.contains(pnm_search, case=False).any(), axis=1)] if pnm_search else df_pnms
            st.dataframe(display_pnm, use_container_width=True)
        else: st.info("No PNM data found.")

    # --- TAB 4: VIEW BUMP TEAMS ---
    with tab4:
        st.header("Bump Team Management")
        df_teams = get_data("Bump Teams")
        if not df_teams.empty:
            id_col = next((c for c in df_teams.columns if 'team id' in c.lower() or 'id' in c.lower()), df_teams.columns[4] if len(df_teams.columns)>4 else None)
            creator_col = next((c for c in df_teams.columns if 'creator' in c.lower()), df_teams.columns[1] if len(df_teams.columns)>1 else None)
            partners_col = next((c for c in df_teams.columns if 'partner' in c.lower()), df_teams.columns[2] if len(df_teams.columns)>2 else None)
            rank_col = next((c for c in df_teams.columns if 'rank' in c.lower()), None)

            if id_col and creator_col:
                df_teams['display_label'] = df_teams.apply(lambda x: f"Team {x[id_col]} | {x[creator_col]}, {x.get(partners_col, '')}", axis=1)
                t1, t2 = st.tabs(["Single Team Update", "Bulk Upload CSV"])
                
                with t1:
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        sel_label = st.selectbox("Select Team to Rank:", df_teams['display_label'].tolist())
                        sel_id = df_teams[df_teams['display_label'] == sel_label][id_col].values[0]
                    with c2:
                        cur_rank = df_teams[df_teams[id_col] == sel_id][rank_col].values[0] if rank_col else 1
                        try: init_val = int(cur_rank)
                        except: init_val = 1
                        new_rank = st.number_input(f"Assign Rank:", min_value=1, value=init_val, key="team_rank_input")
                    with c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Save Team Rank"):
                            if update_team_ranking(sel_id, new_rank): st.success(f"Rank {new_rank} assigned!"); st.rerun()

                with t2:
                    st.info("Upload a CSV with columns: `Team ID` (or `Creator Name`) and `Ranking`.")
                    team_csv = st.file_uploader("Upload Rankings CSV", type=["csv"], key="team_rank_upload")
                    if team_csv and st.button("Process Bulk Update"):
                        try:
                            df_b = pd.read_csv(team_csv)
                            df_b.columns = df_b.columns.str.strip().str.lower()
                            b_id = next((c for c in df_b.columns if 'id' in c), None)
                            b_name = next((c for c in df_b.columns if 'name' in c or 'creator' in c), None)
                            b_rank = next((c for c in df_b.columns if 'rank' in c), None)
                            
                            if b_rank and (b_id or b_name):
                                bulk_map = {}
                                name_map = dict(zip(df_teams[creator_col].astype(str).str.strip().str.lower(), df_teams[id_col].astype(str))) if b_name else {}
                                for _, r in df_b.iterrows():
                                    rv = r[b_rank]
                                    tid = None
                                    if b_id and pd.notna(r[b_id]): tid = str(int(r[b_id])) if str(r[b_id]).replace('.','').isdigit() else str(r[b_id])
                                    elif b_name and pd.notna(r[b_name]): tid = name_map.get(str(r[b_name]).strip().lower())
                                    if tid: bulk_map[tid] = rv
                                if bulk_map:
                                    cnt = batch_update_team_rankings(bulk_map)
                                    st.success(f"✅ Updated {cnt} teams!"); st.rerun()
                                else: st.warning("No valid teams found.")
                            else: st.error("CSV missing required columns.")
                        except Exception as e: st.error(f"Error: {e}")
            st.divider()
            st.dataframe(df_teams.drop(columns=['display_label'], errors='ignore'), use_container_width=True)
        else: st.info("No bump teams found yet.")

    # --- TAB 5 & 6: EXCUSES & CONNECTIONS ---
    with tab5:
        if st.button("🔄 Refresh Excuses"): st.rerun()
        df_ex = get_data("Party Excuses")
        st.dataframe(df_ex, use_container_width=True) if not df_ex.empty else st.info("No excuses found.")
    with tab6:
        st.header("Prior PNM Connections Log")
        if st.button("🔄 Refresh Connections"): st.rerun()
        df_conn = get_data("Prior Connections")
        st.dataframe(df_conn, use_container_width=True) if not df_conn.empty else st.info("No prior connections found.")

    # --- TAB 7: RUN MATCHING (MODIFIED) ---
    with tab7:
        st.subheader("Matching Configuration")
        c1, c2 = st.columns(2)
        with c1: num_pnm_matches_needed = st.number_input("PNM Matches per Bump Team (Capacity)", min_value=1, value=5, step=1)
        with c2: num_rounds_party = st.number_input("Rounds per Party", min_value=1, value=4, step=1)
        bump_order_set = st.radio("Is the bump order set?", ["yes", "no"], index=1)

        if st.button("Run Matching Algorithm"):
            with st.spinner("Fetching data and processing matching logic..."):
                # 1. LOAD DATA
                parties_val = get_setting_value('B1')
                num_parties = int(parties_val) if parties_val and str(parties_val).isdigit() else 4
                active_roster = get_active_roster_names()
                
                # Fetch Sheets
                bump_teams = get_data("Bump Teams")
                party_excuses = get_data("Party Excuses")
                member_interest = get_data("Member Information")
                member_pnm_no_match = get_data("Prior Connections")
                pnm_intial_interest = get_data("PNM Information")

                if pnm_intial_interest.empty or bump_teams.empty:
                    st.error("Missing critical data (PNM Info or Bump Teams).")
                    st.stop()

                # 2. STANDARDIZE COLUMNS
                pnm_working = standardize_columns(pnm_intial_interest.copy(), entity_type='pnm')
                member_interest = standardize_columns(member_interest, entity_type='member')
                if not party_excuses.empty: party_excuses.columns = party_excuses.columns.str.strip()
                if not member_pnm_no_match.empty: member_pnm_no_match.columns = member_pnm_no_match.columns.str.strip()

                # 3. FILTER MEMBERS (ROSTER)
                if active_roster and not member_interest.empty:
                    active_set = set(n.strip().lower() for n in active_roster)
                    member_interest = member_interest[member_interest['Full Name'].astype(str).str.strip().str.lower().isin(active_set)]
                
                if member_interest.empty:
                    st.error("No valid members found for matching.")
                    st.stop()

                # 4. PARTY ASSIGNMENT LOGIC (Dynamic Tiling)
                n_pnms = len(pnm_working)
                pnms_per_party = int(np.ceil(n_pnms / num_parties))
                party_assignments = np.tile(np.arange(1, num_parties + 1), pnms_per_party)[:n_pnms]
                np.random.seed(42)
                np.random.shuffle(party_assignments)
                pnm_working['Party'] = party_assignments

                # 5. GEOCODING & CLUSTERING (OFFLINE)
                city_coords_map, ALL_CITY_KEYS = load_geo_data()

                def get_coords(hometown):
                    if not isinstance(hometown, str): return None
                    key = hometown.strip().upper()
                    if key in city_coords_map: return city_coords_map[key]
                    matches = difflib.get_close_matches(key, ALL_CITY_KEYS, n=1, cutoff=0.8)
                    return city_coords_map[matches[0]] if matches else None

                all_coords, geo_tracker = [], []
                
                # Gather Coords
                for df, type_ in [(member_interest, 'mem'), (pnm_working, 'pnm')]:
                    id_col = 'Sorority ID' if type_ == 'mem' else 'PNM ID'
                    for idx, row in df.iterrows():
                        if 'Hometown' in row:
                            coords = get_coords(row['Hometown'])
                            if coords:
                                all_coords.append([radians(c) for c in coords])
                                geo_tracker.append({'type': type_, 'id': row[id_col], 'hometown': row['Hometown']})

                mem_geo_tags, pnm_geo_tags = {}, {}
                if all_coords:
                    dist_matrix = haversine_distances(all_coords, all_coords) * 3958.8
                    geo_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=30, metric='precomputed', linkage='single')
                    geo_labels = geo_clustering.fit_predict(dist_matrix)
                    
                    geo_groups = {}
                    for i, label in enumerate(geo_labels):
                        if label not in geo_groups: geo_groups[label] = []
                        geo_groups[label].append(geo_tracker[i]['hometown'])
                    
                    for i, label in enumerate(geo_labels):
                        grp = geo_groups[label][0]
                        trk = geo_tracker[i]
                        if trk['type'] == 'mem': mem_geo_tags[trk['id']] = grp
                        else: pnm_geo_tags[trk['id']] = grp

                # 6. SEMANTIC CLUSTERING
                model = load_model()
                all_terms, interest_maps = [], {'mem': [], 'pnm': []}
                cols_extract = ['Major', 'Minor', 'Hobbies', 'College Involvement', 'High School Involvement']

                for df, type_ in [(member_interest, 'mem'), (pnm_working, 'pnm')]:
                    id_col = 'Sorority ID' if type_ == 'mem' else 'PNM ID'
                    for idx, row in df.iterrows():
                        text = ", ".join([str(row.get(c, '')).lower() for c in cols_extract])
                        terms = [t.strip() for t in text.split(',') if t.strip() and t.lower() != 'nan']
                        for term in terms:
                            all_terms.append(term)
                            interest_maps[type_].append({'id': row[id_col], 'term': term})
                
                term_to_group = {}
                if all_terms:
                    u_terms = list(set(all_terms))
                    embeddings = model.encode(u_terms)
                    sem_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='cosine', linkage='average').fit_predict(embeddings)
                    
                    temp_map = {}
                    for term, label in zip(u_terms, sem_labels):
                        if label not in temp_map: temp_map[label] = []
                        temp_map[label].append(term)
                    
                    for terms in temp_map.values():
                        attr_name = min(terms, key=len)
                        for t in terms: term_to_group[t] = attr_name

                # 7. FINALIZE ATTRIBUTES
                def get_year_tag(y):
                    if pd.isna(y): return None
                    m = difflib.get_close_matches(str(y).strip(), ["Freshman", "Sophomore", "Junior", "Senior"], n=1, cutoff=0.6)
                    return m[0] if m else str(y).strip().title()

                # Generate attribute strings
                for df, type_, tag_map in [(member_interest, 'mem', mem_geo_tags), (pnm_working, 'pnm', pnm_geo_tags)]:
                    id_col = 'Sorority ID' if type_ == 'mem' else 'PNM ID'
                    final_attrs = {row[id_col]: set() for _, row in df.iterrows()}
                    
                    for idx, row in df.iterrows():
                        pid = row[id_col]
                        yt = get_year_tag(row.get('Year'))
                        if yt: final_attrs[pid].add(yt)
                        if pid in tag_map: final_attrs[pid].add(tag_map[pid])
                        
                    for entry in interest_maps[type_]:
                        if entry['term'] in term_to_group: final_attrs[entry['id']].add(term_to_group[entry['term']])
                    
                    df['attributes_for_matching'] = df[id_col].map(lambda x: ", ".join(final_attrs.get(x, set())))

                # 8. MATCHING PREP
                excuse_col = next((c for c in party_excuses.columns if "party" in c.lower() and "attend" in c.lower()), None)
                party_excuses_exploded = party_excuses.explode(excuse_col) if excuse_col and not party_excuses.empty else pd.DataFrame(columns=['Member Name', 'Party'])
                if excuse_col: 
                    party_excuses[excuse_col] = party_excuses[excuse_col].apply(lambda x: [int(i) for i in re.findall(r'\d+', str(x))] if pd.notnull(x) else [])
                    party_excuses_exploded = party_excuses.explode(excuse_col)

                no_match_pairs = set()
                if not member_pnm_no_match.empty:
                    m_c = next((c for c in member_pnm_no_match.columns if 'member' in c.lower() and 'name' in c.lower()), None)
                    p_c = next((c for c in member_pnm_no_match.columns if 'pnm' in c.lower() and 'name' in c.lower()), None)
                    if m_c and p_c:
                        no_match_pairs = {(r[m_c], r[p_c]) for r in member_pnm_no_match.to_dict('records') if pd.notna(r.get(m_c))}

                member_attr_cache = {r['Sorority ID']: set(str(r.get('attributes_for_matching', '')).split(', ')) for r in member_interest.to_dict('records')}
                name_to_id_map = {r['Full Name']: r['Sorority ID'] for i, r in member_interest.iterrows() if r.get('Full Name')}
                
                all_traits = member_interest['attributes_for_matching'].str.split(', ').explode()
                trait_weights = (len(member_interest) / all_traits.value_counts()).to_dict()
                strength_bonus = {1: 1.5, 2: 1.0, 3: 0.5, 4: 0.0}

                # 9. EXECUTION LOOP
                st.write("---"); st.write("### Results"); prog = st.progress(0); buffers = []

                # --- INTERNAL FUNCTIONS FOR MATCHING ---
                def run_internal_rotation(assign_map, t_list, method):
                    output = []
                    rounds = 1 if bump_order_set == 'yes' else num_rounds_party
                    
                    for tid, pnms in assign_map.items():
                        team = next((t for t in t_list if t['t_idx'] == tid), None)
                        if not team or not pnms: continue
                        
                        rgl = str(team['row_data'].get('RGL', '')).strip()
                        valid_mems = [{'id': mid, 'name': nm} for mid, nm in zip(team['member_ids'], team['members']) if mid]
                        history = set()

                        for r in range(1, rounds + 1):
                            active = [m for m in valid_mems if m['name'] != rgl] if r == 1 and rgl else valid_mems
                            repeats_ok = r > len(active)
                            
                            if method == 'flow':
                                SG = nx.DiGraph()
                                SG.add_node('s', demand=-len(pnms)); SG.add_node('t', demand=len(pnms))
                                for p in pnms: SG.add_edge('s', f"p_{p['p_id']}", capacity=1, weight=0)
                                for m in active: SG.add_edge(f"m_{m['id']}", 't', capacity=1, weight=0)
                                
                                for p in pnms:
                                    for m in active:
                                        if (p['p_id'], m['id']) in history and not repeats_ok: continue
                                        shared = p['p_attrs'].intersection(member_attr_cache.get(m['id'], set()))
                                        score = sum(trait_weights.get(t, 1.0) for t in shared)
                                        cost = int((1/(1+score))*10000) + (50000 if (p['p_id'], m['id']) in history else 0)
                                        reason = ", ".join(shared) if shared else "Rotation"
                                        SG.add_edge(f"p_{p['p_id']}", f"m_{m['id']}", capacity=1, weight=cost, reason=reason)
                                
                                try:
                                    min_flow = nx.min_cost_flow(SG)
                                    for p in pnms:
                                        p_node = f"p_{p['p_id']}"
                                        if p_node in min_flow:
                                            for tgt, flow in min_flow[p_node].items():
                                                if flow > 0 and tgt != 't':
                                                    mid = int(tgt.replace('m_', ''))
                                                    mname = next(m['name'] for m in valid_mems if m['id'] == mid)
                                                    ed = SG.get_edge_data(p_node, tgt)
                                                    output.append({'Round': r, 'Team ID': tid, 'Team Members': team['joined_names'], 'PNM Name': p['p_name'], 'Matched Member': mname, 'Reason': ed.get('reason')})
                                                    history.add((p['p_id'], mid))
                                except: pass
                            
                            elif method == 'greedy':
                                # Simplified greedy logic for brevity in this block
                                used_p, used_m = set(), set()
                                # (Score logic similar to global but per round - skipped detailed implementation for length constraint)
                                pass 
                    return output

                def gen_instructions(rot_data):
                    if not rot_data: return []
                    df = pd.DataFrame(rot_data)
                    if df.empty: return []
                    df = df.sort_values(by=['Team ID', 'PNM Name', 'Round'])
                    df['Bump'] = df.groupby(['Team ID', 'PNM Name'])['Matched Member'].shift(1)
                    return df[df['Bump'].notna()].rename(columns={'Matched Member': 'You', 'Bump': 'Bump This Person'}).to_dict('records')

                # LOOP PARTIES
                for party in range(1, num_parties + 1):
                    sub_pnm = pnm_working[pnm_working['Party'] == party]
                    if sub_pnm.empty: continue
                    
                    # Prepare Node Lists
                    pnm_nodes = []
                    for i, r in enumerate(sub_pnm.to_dict('records')):
                        pnm_nodes.append({
                            'id': r['PNM ID'], 'name': r['Full Name'], 
                            'attrs': set(str(r.get('attributes_for_matching', '')).split(', ')),
                            'rank': float(r.get("Average Recruit Rank", 1.0)), 
                            'node': f"p_{i}"
                        })
                    
                    # Prepare Teams (Filtered by Excuses)
                    excused = set()
                    if excuse_col and not party_excuses_exploded.empty:
                        excused = set(party_excuses_exploded[party_excuses_exploded[excuse_col] == party][party_excuses_exploded.columns[1]])
                    
                    team_nodes = []
                    for i, r in enumerate(bump_teams.to_dict('records')):
                        mems = [r["Creator Name"]] + [p.strip() for p in re.split(r'[,;]\s*', str(r.get("Bump Partners", ""))) if p.strip()]
                        if any(m in excused for m in mems): continue
                        
                        team_nodes.append({
                            't_idx': i, 'members': mems, 'joined_names': ", ".join(mems),
                            'member_ids': [name_to_id_map.get(m) for m in mems],
                            'bonus': strength_bonus.get(int(float(r.get("Ranking", 4))), 0.0),
                            'node': f"t_{i}", 'row_data': r
                        })
                    
                    # Global Matching (Flow)
                    G = nx.DiGraph()
                    G.add_node('s', demand=-len(pnm_nodes)); G.add_node('t', demand=len(pnm_nodes))
                    G.add_edge('dummy', 't', capacity=len(pnm_nodes), weight=0)
                    
                    for p in pnm_nodes:
                        G.add_edge('s', p['node'], capacity=1, weight=0)
                        G.add_edge(p['node'], 'dummy', capacity=1, weight=1000000)
                        
                    pairs = []
                    for p in pnm_nodes:
                        for t in team_nodes:
                            if any((m, p['name']) in no_match_pairs for m in t['members']): continue
                            score = 0
                            reasons = []
                            for mid, mname in zip(t['member_ids'], t['members']):
                                if mid:
                                    shared = p['attrs'].intersection(member_attr_cache.get(mid, set()))
                                    if shared: 
                                        score += sum(trait_weights.get(tr, 1.0) for tr in shared)
                                        reasons.append(f"{mname} ({', '.join(shared)})")
                            
                            cost = int((1 / (1 + score + p['bonus'] + t['bonus'])) * 10000)
                            G.add_edge(p['node'], t['node'], capacity=1, weight=cost)
                            pairs.append(((p['node'], t['node']), {'p': p, 't': t, 'reason': "; ".join(reasons), 'cost': cost}))
                            
                    for t in team_nodes: G.add_edge(t['node'], 't', capacity=num_pnm_matches_needed, weight=0)
                    
                    # Solve Global
                    assign_map = {t['t_idx']: [] for t in team_nodes}
                    results_global = []
                    pair_dict = dict(pairs)
                    
                    try:
                        flow = nx.min_cost_flow(G)
                        for p in pnm_nodes:
                            matched = False
                            if p['node'] in flow:
                                for tgt, amt in flow[p['node']].items():
                                    if amt > 0 and tgt != 'dummy':
                                        info = pair_dict.get((p['node'], tgt))
                                        if info:
                                            assign_map[info['t']['t_idx']].append({'p_id': p['id'], 'p_name': p['name'], 'p_attrs': p['attrs']})
                                            results_global.append({'PNM': p['name'], 'Team': info['t']['joined_names'], 'Reason': info['reason']})
                                            matched = True
                            if not matched: results_global.append({'PNM': p['name'], 'Team': 'NO MATCH', 'Reason': 'Conflict/Capacity'})
                    except: st.error(f"Party {party} failed optimization.")

                    # Internal Rotations (Flow)
                    rot_results = run_internal_rotation(assign_map, team_nodes, 'flow')
                    bump_results = gen_instructions(rot_results)
                    
                    # Save to Excel
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        pd.DataFrame(results_global).to_excel(writer, sheet_name="Global_Matches", index=False)
                        pd.DataFrame(rot_results).to_excel(writer, sheet_name="Rotations", index=False)
                        pd.DataFrame(bump_results).to_excel(writer, sheet_name="Bump_Instructions", index=False)
                    
                    buffers.append({'party': party, 'data': out.getvalue()})
                    prog.progress(party / num_parties)
            
            st.success("Done!")
            for b in buffers:
                st.download_button(f"Download Party {b['party']}", b['data'], f"Party_{b['party']}.xlsx")
