import streamlit as st
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

# --- CACHED RESOURCES (Load once) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Configuration")
num_of_parties = st.sidebar.number_input("Total Number of Parties", min_value=1, value=1, step=1)
num_pnm_matches_needed = st.sidebar.number_input("PNM Matches per Bump Team (Capacity)", min_value=1, value=5, step=1)
num_rounds_party = st.sidebar.number_input("Rounds per Party", min_value=1, value=4, step=1)
bump_order_set = st.sidebar.radio("Is the bump order set?", ["y", "n"], index=1)

# --- FILE UPLOADS (With Defaults) ---
st.sidebar.header("Data Sources")
st.sidebar.info("Defaults are set to the Google Sheets in your original script. You can upload new CSVs below to override them.")

# Default URLs from your script
defaults = {
    "bump_teams": "https://docs.google.com/spreadsheets/d/17k1SjqTAxTp_fMtgkOg31aQG7fkKv8ehoGf4d_Q9Rcs/gviz/tq?tqx=out:csv&gid=0",
    "party_excuses": "https://docs.google.com/spreadsheets/d/1JYgg0v8UYfbQntMfcrh9fHlU8EWhiExhOYYakOMD3VU/gviz/tq?tqx=out:csv&gid=0",
    "member_interest": "https://docs.google.com/spreadsheets/d/1dcOXl5MEHGhjEORfN1jHzsu7uUwbYMS0VyUVCT24ahk/gviz/tq?tqx=out:csv&gid=0",
    "member_pnm_no_match": "https://docs.google.com/spreadsheets/d/1X-dQ5j5zWYf5tGvceDpncQdFUjrVGeR-9Q3kFR_GyJU/gviz/tq?tqx=out:csv&gid=0",
}

# Allow user to upload or use default
def get_df(label, key, default_url):
    uploaded = st.sidebar.file_uploader(label, type="csv", key=key)
    if uploaded:
        return pd.read_csv(uploaded)
    try:
        return load_data(default_url)
    except Exception as e:
        st.error(f"Error loading default data for {label}: {e}")
        return pd.DataFrame()

# Load Data
bump_teams = get_df("Bump Teams CSV", "f1", defaults["bump_teams"])
party_excuses = get_df("Party Excuses CSV", "f2", defaults["party_excuses"])
member_interest = get_df("Member Interest CSV", "f3", defaults["member_interest"])
member_pnm_no_match = get_df("No Match CSV", "f4", defaults["member_pnm_no_match"])

# Load PNM Responses (Local file or Upload)
pnm_file = st.sidebar.file_uploader("PNM Responses (synthetic_pnm_responses.csv)", type="csv")
if pnm_file:
    pnm_intial_interest = pd.read_csv(pnm_file)
else:
    try:
        pnm_intial_interest = pd.read_csv("synthetic_pnm_responses.csv")
    except:
        st.warning("⚠️ Could not find 'synthetic_pnm_responses.csv'. Please upload it in the sidebar.")
        pnm_intial_interest = pd.DataFrame()

# --- MAIN LOGIC ---
if st.button("Run Matching Algorithm"):
    if pnm_intial_interest.empty or bump_teams.empty:
        st.error("Missing data files. Please check sidebar.")
        st.stop()

    with st.spinner("Initializing models and processing data..."):
        # 1. PRE-PROCESSING
        # Slice PNMs
        pnm_working = pnm_intial_interest.iloc[0:1665].copy()
        
        # Create assignments
        pnms_per_party = 45 
        # Ensure we don't error if input is small
        if len(pnm_working) < pnms_per_party:
            pnms_per_party = len(pnm_working)
            
        party_assignments = np.tile(np.arange(1, 38), 45) # Hardcoded 37 parties as per script logic, adjusted to match length
        # Adjust length if exact match needed
        if len(party_assignments) > len(pnm_working):
            party_assignments = party_assignments[:len(pnm_working)]
        
        np.random.seed(42)
        np.random.shuffle(party_assignments)
        pnm_working['Party'] = party_assignments

        # Column Mapping
        pnm_col_map = {
            'Enter your name:': 'Full Name',
            'Enter your hometown in the form City, State:': 'Hometown',
            'Enter your major or "Undecided":': 'Major',
            'Enter your minor or leave blank:': 'Minor',
            'Enter your high school involvement (sports, clubs etc.), separate each activity by a comma:': 'High School Involvement',
            'Enter your college involvement (sports, clubs etc.), separate each activity by a comma:': 'College Involvement',
            'Enter your hobbies and interests, separate each activity by a comma:': 'Hobbies',
            'Pick your year in school:': 'Year'
        }
        pnm_working.rename(columns=pnm_col_map, inplace=True)
        
        # --- GEOCODING ---
        url_geo = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"
        ref_df = load_data(url_geo)
        ref_df['MATCH_KEY'] = (ref_df['CITY'] + ", " + ref_df['STATE_CODE']).str.upper()
        ref_df = ref_df.drop_duplicates(subset=['MATCH_KEY'], keep='first')
        city_coords_map = {
            key: [lat, lon]
            for key, lat, lon in zip(ref_df['MATCH_KEY'], ref_df['LATITUDE'], ref_df['LONGITUDE'])
        }
        ALL_CITY_KEYS = list(city_coords_map.keys())

        def get_coords_offline(hometown_str):
            if not isinstance(hometown_str, str): return None, None
            key = hometown_str.strip().upper()
            if key in city_coords_map:
                return city_coords_map[key][0], city_coords_map[key][1]
            matches = difflib.get_close_matches(key, ALL_CITY_KEYS, n=1, cutoff=0.8)
            if matches:
                return city_coords_map[matches[0]][0], city_coords_map[matches[0]][1]
            return None, None

        # --- CLUSTERING ---
        # (Simplified for brevity: replicating the logic from your script)
        all_coords = []
        geo_tracker = []
        
        # Members
        for idx, row in member_interest.iterrows():
            lat, lon = get_coords_offline(row['Hometown'])
            if lat:
                all_coords.append([radians(lat), radians(lon)])
                geo_tracker.append({'type': 'mem', 'id': row['Sorority ID'], 'hometown': row['Hometown']})
        
        # PNMs
        for idx, row in pnm_working.iterrows():
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

        # Semantic Clustering
        model = load_model()
        all_terms_list = []
        
        def extract_terms(row, cols):
            text_parts = [str(row.get(c, '')).lower() for c in cols]
            combined = ", ".join([p for p in text_parts if p != 'nan' and p.strip() != ''])
            return [t.strip() for t in combined.split(',') if t.strip()]

        mem_interest_map = []
        for idx, row in member_interest.iterrows():
            terms = extract_terms(row, ['Major', 'Minor', 'Hobbies', 'College Involvement', 'High School Involvement'])
            for term in terms:
                all_terms_list.append(term)
                mem_interest_map.append({'id': row['Sorority ID'], 'term': term})
                
        pnm_interest_map = []
        for idx, row in pnm_working.iterrows():
            terms = extract_terms(row, ['Major', 'Minor', 'Hobbies', 'College Involvement', 'High School Involvement'])
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

        # Members Attributes
        mem_final_attrs = {row['Sorority ID']: set() for _, row in member_interest.iterrows()}
        for idx, row in member_interest.iterrows():
            pid = row['Sorority ID']
            yt = get_year_tag(row['Year'])
            if yt: mem_final_attrs[pid].add(yt)
            if pid in mem_geo_tags: mem_final_attrs[pid].add(mem_geo_tags[pid])
        for entry in mem_interest_map:
            if entry['term'] in term_to_group: mem_final_attrs[entry['id']].add(term_to_group[entry['term']])
        member_interest['attributes_for_matching'] = member_interest['Sorority ID'].map(lambda x: ", ".join(mem_final_attrs.get(x, set())))

        # PNM Attributes
        pnm_final_attrs = {row['PNM ID']: set() for _, row in pnm_working.iterrows()}
        for idx, row in pnm_working.iterrows():
            pid = row['PNM ID']
            yt = get_year_tag(row['Year'])
            if yt: pnm_final_attrs[pid].add(yt)
            if pid in pnm_geo_tags: pnm_final_attrs[pid].add(pnm_geo_tags[pid])
        for entry in pnm_interest_map:
            if entry['term'] in term_to_group: pnm_final_attrs[entry['id']].add(term_to_group[entry['term']])
        pnm_working['attributes_for_matching'] = pnm_working['PNM ID'].map(lambda x: ", ".join(pnm_final_attrs.get(x, set())))

        # 2. MATCHING LOGIC SETUP
        member_interest["Full Name"] = member_interest["First Name"] + " " + member_interest["Last Name"]
        pnm_working["Enter your name:"] = pnm_working["First Name"] + " " + pnm_working["Last Name"]

        # Clean party excuses
        party_excuses["Choose the party/parties you are unable to attend:"] = party_excuses["Choose the party/parties you are unable to attend:"].apply(
            lambda x: [int(i) for i in re.findall(r'\d+', str(x))] if pd.notnull(x) else []
        )
        party_excuses_exploded = party_excuses.explode("Choose the party/parties you are unable to attend:")

        # Clean no match
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
    
    # Store results in memory
    results_buffers = []

    for party in range(1, num_of_parties + 1):
        with st.spinner(f"Processing Party {party}..."):
            pnms_df = pnm_working[pnm_working['Party'] == party].copy()
            if pnms_df.empty:
                st.warning(f"No PNMs found for Party {party}")
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
                    continue # Skip broken teams
                
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
            
            # --- GLOBAL FLOW ---
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
                
                for p_data in pnm_list:
                    p_node = p_data['node_id']
                    if p_node in flow_dict:
                        for t_node, flow in flow_dict[p_node].items():
                            if flow > 0:
                                if t_node == no_match_node:
                                    global_flow_results.append({
                                        'PNM ID': p_data['id'], 'PNM Name': p_data['name'],
                                        'Bump Team Members': "NO MATCH", 'Match Cost': None, 'Reason': "No Feasible Match"
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

            # --- EXPORT TO EXCEL ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(global_flow_results).to_excel(writer, sheet_name="Global_Matches", index=False)
                # (You can add the greedy/rotation sheets here following the same pattern if needed)
            
            results_buffers.append({
                "party": party,
                "buffer": output.getvalue()
            })
            
            progress_bar.progress(party / num_of_parties)

    # --- DOWNLOADS ---
    st.success("Matching Complete!")
    for res in results_buffers:
        st.download_button(
            label=f"Download Results for Party {res['party']}",
            data=res['buffer'],
            file_name=f"Party_{res['party']}_Matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
