import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sorority Matching App", layout="wide")
st.title("Sorority Matching Algorithm")

# --- CACHED RESOURCES ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- GOOGLE SHEETS CONNECTION HELPER ---
def get_gspread_client():
    """
    Authenticates with Google Sheets using credentials from st.secrets.
    Requires a [gcp_service_account] section in .streamlit/secrets.toml.
    """
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        if "gcp_service_account" not in st.secrets:
            st.error("⚠️ Missing `[gcp_service_account]` in .streamlit/secrets.toml")
            st.stop()
            
        creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = Credentials.from_service_account_info(
            creds_dict,
            scopes=scopes
        )
        
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        return None

def load_data_from_sheet(sheet_name):
    """
    Loads specific worksheets from a single Google Workbook into DataFrames.
    """
    client = get_gspread_client()
    if not client:
        return None

    try:
        sh = client.open(sheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"❌ Could not find Google Sheet named: '{sheet_name}'. Please check the name and ensure the Service Account email has access.")
        return None

    data_dict = {}
    required_tabs = [
        "Member Information", 
        "PNM Information",
        "PNM Rankings",  
        "Bump Teams", 
        "Party Excuses", 
        "Prior Connections",
        "Settings"       
    ]

    with st.spinner(f"Fetching data from '{sheet_name}'..."):
        for tab in required_tabs:
            try:
                worksheet = sh.worksheet(tab)
                records = worksheet.get_all_records()
                df = pd.DataFrame(records)
                data_dict[tab] = df
            except gspread.exceptions.WorksheetNotFound:
                if tab not in ["PNM Rankings", "Settings", "Prior Connections", "Party Excuses"]: 
                    st.warning(f"⚠️ Worksheet '{tab}' not found. Using empty DataFrame.")
                data_dict[tab] = pd.DataFrame()
            except Exception as e:
                data_dict[tab] = pd.DataFrame()
    
    return data_dict

# --- EXCEL FORMATTING HELPER ---
def auto_adjust_columns(writer, sheet_name, df):
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns):
        max_len = max(
            df[col].astype(str).map(len).max(),
            len(str(col))
        ) + 2
        worksheet.set_column(idx, idx, max_len)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Google Sheets Connection")
st.sidebar.info("Ensure you have shared your Google Sheet with the Service Account email in your `secrets.toml`.")
sheet_name_input = st.sidebar.text_input("Enter Google Sheet Name", value="OverallMatchingInformation")

# Initialize Session State for Data
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
    st.session_state['member_interest'] = pd.DataFrame()
    st.session_state['pnm_intial_interest'] = pd.DataFrame()
    st.session_state['pnm_rankings'] = pd.DataFrame()
    st.session_state['bump_teams'] = pd.DataFrame()
    st.session_state['party_excuses'] = pd.DataFrame()
    st.session_state['member_pnm_no_match'] = pd.DataFrame()
    st.session_state['settings'] = pd.DataFrame()

# Load Data Button
if st.sidebar.button("Load Data from Google Sheets"):
    if not sheet_name_input:
        st.sidebar.error("Please enter a sheet name.")
    else:
        loaded_data = load_data_from_sheet(sheet_name_input)
        if loaded_data:
            st.session_state['member_interest'] = loaded_data.get("Member Information", pd.DataFrame())
            st.session_state['pnm_intial_interest'] = loaded_data.get("PNM Information", pd.DataFrame())
            st.session_state['pnm_rankings'] = loaded_data.get("PNM Rankings", pd.DataFrame())
            st.session_state['bump_teams'] = loaded_data.get("Bump Teams", pd.DataFrame())
            st.session_state['party_excuses'] = loaded_data.get("Party Excuses", pd.DataFrame())
            st.session_state['member_pnm_no_match'] = loaded_data.get("Prior Connections", pd.DataFrame())
            st.session_state['settings'] = loaded_data.get("Settings", pd.DataFrame())
            
            st.session_state['data_loaded'] = True
            
            # Validation
            if not st.session_state['member_interest'].empty: st.sidebar.success("✅ Member Info Loaded")
            if not st.session_state['pnm_intial_interest'].empty: st.sidebar.success("✅ PNM Info Loaded")
            if not st.session_state['bump_teams'].empty: st.sidebar.success("✅ Bump Teams Loaded")

# --- SETTINGS CONFIGURATION ---
# Determine defaults based on loaded settings or fallbacks
def_parties = 1
def_capacity = 5
def_rounds = 4
def_bump_idx = 1 # 'no'

if not st.session_state['settings'].empty:
    try:
        # Expecting columns "Setting" and "Value"
        settings_df = st.session_state['settings']
        settings_df.columns = settings_df.columns.astype(str).str.strip()
        if 'Setting' in settings_df.columns and 'Value' in settings_df.columns:
            s_dict = dict(zip(settings_df['Setting'], settings_df['Value']))
            
            if 'Number of Parties' in s_dict: def_parties = int(s_dict['Number of Parties'])
            if 'PNM Matches per Bump Team' in s_dict: def_capacity = int(s_dict['PNM Matches per Bump Team'])
            if 'Rounds per Party' in s_dict: def_rounds = int(s_dict['Rounds per Party'])
            if 'Bump Order Set' in s_dict: 
                def_bump_idx = 0 if str(s_dict['Bump Order Set']).lower() == 'yes' else 1
    except Exception as e:
        st.sidebar.warning(f"Could not parse settings sheet: {e}")

st.sidebar.markdown("---")
st.sidebar.header("Configuration")
num_of_parties = st.sidebar.number_input("Total Number of Parties", min_value=1, value=def_parties, step=1)
num_pnm_matches_needed = st.sidebar.number_input("PNM Matches per Bump Team (Capacity)", min_value=1, value=def_capacity, step=1)
num_rounds_party = st.sidebar.number_input("Rounds per Party", min_value=1, value=def_rounds, step=1)
bump_order_set = st.sidebar.radio("Is the bump order set?", ["yes", "no"], index=def_bump_idx)

# Retrieve from session state
member_interest = st.session_state['member_interest']
pnm_intial_interest = st.session_state['pnm_intial_interest']
pnm_rankings = st.session_state['pnm_rankings']
bump_teams = st.session_state['bump_teams']
party_excuses = st.session_state['party_excuses']
member_pnm_no_match = st.session_state['member_pnm_no_match']

# --- MAIN LOGIC ---
if st.button("Run Matching Algorithm"):
    if pnm_intial_interest.empty or bump_teams.empty or member_interest.empty:
        st.error("Missing critical data. Please load data from Google Sheets first.")
        st.stop()

    with st.spinner("Initializing models and processing data..."):
        # 0. CLEAN COLUMNS
        for df in [member_interest, pnm_intial_interest, pnm_rankings, bump_teams, party_excuses, member_pnm_no_match]:
            if not df.empty:
                df.columns = df.columns.astype(str).str.strip()
        
        # --- 1. PRE-PROCESSING ---
        
        # MEMBERS: Check for "Full Name"
        if "Full Name" not in member_interest.columns:
             # Fallback if spec was wrong
             if "First Name" in member_interest.columns and "Last Name" in member_interest.columns:
                  member_interest["Full Name"] = member_interest["First Name"] + " " + member_interest["Last Name"]
             else:
                  st.error("Member sheet must have a 'Full Name' column.")
                  st.stop()

        # PNMS: Check for "PNM Name" and rename to "Full Name" for internal logic
        pnm_working = pnm_intial_interest.copy()
        
        if "PNM Name" in pnm_working.columns:
             pnm_working["Full Name"] = pnm_working["PNM Name"]
        elif "Full Name" in pnm_working.columns:
             pass # Already exists
        else:
             st.error("PNM sheet must have a 'PNM Name' column.")
             st.stop()
        
        # Clean PNM IDs for consistency
        if 'PNM ID' in pnm_working.columns:
            pnm_working['PNM ID'] = pnm_working['PNM ID'].astype(str).str.strip()

        # --- RANKING LOGIC UPDATE ---
        # User specified "Average Recruit Rank" is already in PNM Information sheet
        # We ensure it is numeric and fill NaNs with 1.0 (highest priority/default)
        if "Average Recruit Rank" in pnm_working.columns:
            pnm_working["Average Recruit Rank"] = pd.to_numeric(pnm_working["Average Recruit Rank"], errors='coerce').fillna(1.0)
        else:
            # Fallback if column is missing
            st.warning("Column 'Average Recruit Rank' not found in PNM Information. Defaulting to 1.0.")
            pnm_working["Average Recruit Rank"] = 1.0

        # Handle Party Assignments
        if 'Party' in pnm_working.columns and pnm_working['Party'].any():
             pnm_working['Party'] = pd.to_numeric(pnm_working['Party'], errors='coerce').fillna(1).astype(int)
        else:
            # Fallback random assignment if column missing or empty
            pnms_per_party = 45 
            party_assignments = np.tile(np.arange(1, num_of_parties + 1), 200)
            if len(party_assignments) > len(pnm_working): party_assignments = party_assignments[:len(pnm_working)]
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
        
        # Process Members
        for idx, row in member_interest.iterrows():
            if 'Hometown' in row and row['Hometown']:
                lat, lon = get_coords_offline(str(row['Hometown']))
                if lat:
                    all_coords.append([radians(lat), radians(lon)])
                    geo_tracker.append({'type': 'mem', 'id': row.get('Sorority ID', idx), 'hometown': row['Hometown']})
        
        # Process PNMs
        for idx, row in pnm_working.iterrows():
            p_id = row.get('PNM ID', idx)
            if 'Hometown' in row and row['Hometown']:
                lat, lon = get_coords_offline(str(row['Hometown']))
                if lat:
                    all_coords.append([radians(lat), radians(lon)])
                    geo_tracker.append({'type': 'pnm', 'id': p_id, 'hometown': row['Hometown']})

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
        
        # Using specific columns from your attached sheets
        semantic_cols = ['Major', 'Minor', 'Hobbies', 'College Involvement', 'High School Involvement']
        
        def extract_terms(row, cols):
            text_parts = []
            for c in cols:
                val = str(row.get(c, '')).lower()
                if val and val != 'nan':
                    text_parts.append(val)
            combined = ", ".join(text_parts)
            return [t.strip() for t in combined.split(',') if t.strip()]

        mem_interest_map = []
        for idx, row in member_interest.iterrows():
            m_id = row.get('Sorority ID', idx)
            terms = extract_terms(row, semantic_cols)
            for term in terms:
                all_terms_list.append(term)
                mem_interest_map.append({'id': m_id, 'term': term})
                
        pnm_interest_map = []
        for idx, row in pnm_working.iterrows():
            p_id = row.get('PNM ID', idx)
            terms = extract_terms(row, semantic_cols)
            for term in terms:
                all_terms_list.append(term)
                pnm_interest_map.append({'id': p_id, 'term': term})

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

        mem_final_attrs = {row.get('Sorority ID', i): set() for i, row in member_interest.iterrows()}
        for idx, row in member_interest.iterrows():
            pid = row.get('Sorority ID', idx)
            yt = get_year_tag(row.get('Year'))
            if yt: mem_final_attrs[pid].add(yt)
            if pid in mem_geo_tags: mem_final_attrs[pid].add(mem_geo_tags[pid])
        for entry in mem_interest_map:
            if entry['term'] in term_to_group: mem_final_attrs[entry['id']].add(term_to_group[entry['term']])
        member_interest['attributes_for_matching'] = member_interest.apply(lambda r: ", ".join(mem_final_attrs.get(r.get('Sorority ID'), set())), axis=1)

        pnm_final_attrs = {row.get('PNM ID', i): set() for i, row in pnm_working.iterrows()}
        for idx, row in pnm_working.iterrows():
            pid = row.get('PNM ID', idx)
            yt = get_year_tag(row.get('Year'))
            if yt: pnm_final_attrs[pid].add(yt)
            if pid in pnm_geo_tags: pnm_final_attrs[pid].add(pnm_geo_tags[pid])
        for entry in pnm_interest_map:
            if entry['term'] in term_to_group: pnm_final_attrs[entry['id']].add(term_to_group[entry['term']])
        pnm_working['attributes_for_matching'] = pnm_working.apply(lambda r: ", ".join(pnm_final_attrs.get(r.get('PNM ID'), set())), axis=1)

        # 2. MATCHING LOGIC SETUP
        
        # --- PARSE EXCUSES ---
        party_excuses["Choose the party/parties you are unable to attend:"] = party_excuses["Choose the party/parties you are unable to attend:"].apply(
            lambda x: [int(i) for i in re.findall(r'\d+', str(x))] if pd.notnull(x) else []
        )
        party_excuses_exploded = party_excuses.explode("Choose the party/parties you are unable to attend:")

        member_pnm_no_match["Choose the PNM or PNMs you should NOT match with:"] = member_pnm_no_match["Choose the PNM or PNMs you should NOT match with:"].str.split(r',\s*', regex=True)
        no_match_exploded = member_pnm_no_match.explode("Choose the PNM or PNMs you should NOT match with:")
        
        member_attr_cache = {
            row['Sorority ID']: set(str(row.get('attributes_for_matching', '')).split(', '))
            if row.get('attributes_for_matching') else set()
            for row in member_interest.to_dict('records')
        }
        name_to_id_map = member_interest.set_index('Full Name')['Sorority ID'].to_dict()
        all_member_traits = member_interest['attributes_for_matching'].str.split(', ').explode()
        trait_freq = all_member_traits.value_counts()
        trait_weights = (len(member_interest) / trait_freq).to_dict()
        strength_bonus_map = {1: 1.5, 2: 1.0, 3: 0.5, 4: 0.0}
        
        no_match_pairs = {
            (row["Choose your name:"], row["Choose the PNM or PNMs you should NOT match with:"])
            for row in no_match_exploded.to_dict('records')
        }

    # --- EXECUTION LOOP ---
    st.write("---")
    st.write("### Results")
    progress_bar = st.progress(0)
    
    results_buffers = []

    # --- DEFINE INTERNAL ROTATION LOGIC (NESTED TO ACCESS CACHES) ---
    def run_internal_rotation(assignment_map, team_list, method='flow'):
        rotation_output = []
        actual_rounds = 1 if bump_order_set == 'yes' else num_rounds_party

        for t_idx, assigned_pnms in assignment_map.items():
            if not assigned_pnms: continue
            team_data = next((t for t in team_list if t['t_idx'] == t_idx), None)
            if not team_data: continue

            raw_rgl = team_data['row_data'].get('Choose your Bump Group Leader (RGL). Leave blank if none.', '')
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
        with st.spinner(f"Processing Party {party}..."):
            pnms_df = pnm_working[pnm_working['Party'] == party].copy()
            if pnms_df.empty:
                progress_bar.progress(party / num_of_parties)
                continue
                
            pnm_list = []
            for i, row in enumerate(pnms_df.to_dict('records')):
                p_attrs = set(str(row['attributes_for_matching']).split(', '))
                p_rank = row.get("Average Recruit Rank", 1.0)
                pnm_list.append({
                    'idx': i, 'id': row['PNM ID'], 'name': row['Enter your name:'],
                    'attrs': p_attrs, 'rank': p_rank, 'bonus': 0.75 * (p_rank - 1),
                    'node_id': f"p_{i}"
                })

            party_excused_names = set(party_excuses_exploded[party_excuses_exploded["Choose the party/parties you are unable to attend:"] == party]["Choose your name:"])
            
            team_list = []
            for raw_idx, row in enumerate(bump_teams.to_dict('records')):
                submitter = row["Choose your name:"]
                partners_str = str(row.get("Choose the name(s) of your bump partner(s):", ""))
                partners = [p.strip() for p in re.split(r'[,;]\s*', partners_str) if p.strip()] if partners_str.lower() != 'nan' else []
                current_members = [submitter] + partners
                
                if any(m in party_excused_names for m in current_members):
                    continue 
                
                t_rank = row.get("Bump Team Ranking:", 4)
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
                
                # --- APPLY USER MODIFICATIONS HERE ---
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

