import streamlit as st
from genshin_optimizer import GenshinTeamOptimizer

# Konfigurasi halaman
st.set_page_config(
    page_title="Genshin Team Optimizer",
    page_icon="🎮",
    layout="wide"
)

@st.cache_resource
def load_optimizer():
    return GenshinTeamOptimizer(data_root=".")

optimizer = load_optimizer()

# Inisialisasi session state
if "teams" not in st.session_state:
    st.session_state.teams = []
if "current_team" not in st.session_state:
    st.session_state.current_team = 0
if "selected_chars" not in st.session_state:
    st.session_state.selected_chars = []

st.title("🎮 Genshin Impact Team Optimizer")

# Sidebar untuk input
with st.sidebar:
    st.header("⚙️ Konfigurasi Tim")
    
    # Gunakan tabs untuk memisahkan bagian
    tab1, tab2, tab3 = st.tabs(["Karakter & Senjata", "Artefak", "Algoritma"])
    
    with tab1:
        st.subheader("📦 Pilih Karakter dan Senjata")
        char_filter = st.text_input("Cari Karakter", key="char_filter")
        all_chars = sorted(optimizer.characters.keys())
        filtered_chars = [c for c in all_chars if char_filter.lower() in c.lower()]
        
        # Gabungkan dengan karakter yang sudah dipilih
        combined_chars = list(set(filtered_chars + st.session_state.selected_chars))
        combined_chars.sort()
        
        selected_chars = st.multiselect(
            "Pilih Karakter (min 4)",
            combined_chars,
            default=st.session_state.selected_chars,
            key="char_select"
        )
        
        # Update session state
        st.session_state.selected_chars = selected_chars

        user_weapons = {}
        if selected_chars:
            st.subheader("🔪 Pilih Senjata", divider="gray")
            for char in selected_chars:
                char_data = optimizer.characters[char]
                weapon_type = char_data.get("weapon_type", "").capitalize()
                weapon_options = [
                    w for w, data in optimizer.weapons.items()
                    if data.get("type", "").capitalize() == weapon_type
                ]
                selected_weapon = st.selectbox(
                    f"{char} ({weapon_type})", 
                    weapon_options, 
                    key=f"weapon_{char}"
                )
                user_weapons[char] = selected_weapon
    
    with tab2:
        st.subheader("🎯 Artefak")
        all_artifacts = sorted(optimizer.artifacts.keys())
        selected_artifacts = st.multiselect(
            "Pilih Set Artefak",
            all_artifacts,
            default=st.session_state.get("selected_artifacts", []),
            key="artifacts_select"
        )
        st.session_state.selected_artifacts = selected_artifacts
    
    with tab3:
        st.subheader("🧠 Algoritma Optimasi")
        algo_choice = st.selectbox(
            "Metode Optimasi",
            ["A* Search (Rekomendasi)", "Simulated Annealing", "Hill Climbing"],
            index=0,
            key="algo_select"
        )
        st.caption("Pilih algoritma untuk mencari kombinasi tim terbaik")

    # Tombol optimalkan di bagian bawah sidebar
    if st.button("🚀 Optimalkan Tim", type="primary", use_container_width=True):
        if len(st.session_state.selected_chars) < 4:
            st.error("Minimal 4 karakter diperlukan.")
        else:
            algo_map = {
                "A* Search (Rekomendasi)": "a_star",
                "Simulated Annealing": "simulated_annealing",
                "Hill Climbing": "hill_climbing"
            }
            with st.spinner("Mengoptimalkan tim..."):
                result = optimizer.generate_teams(
                    st.session_state.selected_chars,
                    user_weapons,
                    st.session_state.selected_artifacts,
                    algo_map[algo_choice]
                )
                st.session_state.teams = result
                st.session_state.current_team = 0
            st.success("Tim berhasil dioptimalkan!")

# Tampilkan hasil (sisa kode tetap sama)

# Tampilkan hasil
teams = st.session_state.teams
current_idx = st.session_state.current_team

# Proteksi agar index tidak keluar batas
if teams:
    if current_idx >= len(teams):
        current_idx = len(teams) - 1
        st.session_state.current_team = current_idx

    team_data = teams[current_idx]

    st.header(f"✨ Tim #{current_idx + 1} - Skor: {team_data['score']}")
    elements = [optimizer.characters[m['name']]['vision'].lower() for m in team_data["team"]]
    resonance = optimizer._get_resonance_name(elements)
    st.markdown(f"**Resonansi Elemen:** {resonance}")

    cols = st.columns(4)
    for i, member in enumerate(team_data["team"]):
        char = optimizer.characters[member['name']]
        with cols[i]:
            st.subheader(f"{member['role'].upper()} - {char['name']}")
            st.write(f"Elemen: {char['vision']}")
            st.write(f"Tipe Senjata: {char['weapon_type']}")
            st.write(f"Senjata: {member['weapon']}")
            arts = optimizer._get_recommended_artifacts(char, member['role'], selected_artifacts)
            st.write("Artefak:")
            for art in arts[:3]:
                st.markdown(f"- {art}")

    # Gunakan callback untuk navigasi yang lebih andal
    def go_prev():
        if st.session_state.current_team > 0:
            st.session_state.current_team -= 1

    def go_next():
        if st.session_state.current_team < len(st.session_state.teams) - 1:
            st.session_state.current_team += 1

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("⬅️ Tim Sebelumnya", on_click=go_prev)
    with col2:
        st.button("➡️ Tim Berikutnya", on_click=go_next)