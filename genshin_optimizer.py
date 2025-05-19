import json
import os
import re
import math
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
from heapq import heappush, heappop

class GenshinTeamOptimizer:
    def __init__(self, data_root: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_root = os.path.join(base_dir, data_root) if data_root else base_dir
        
        print(f"\nMemuat database dari: {self.data_root}")
        
        self.characters = self._load_category("characters")
        self.weapons = self._load_category("weapons")
        self.artifacts = self._load_category("artifacts")
        self.elements = self._load_category("elements")
        
        # Bersihkan data yang tidak valid
        self.weapons = {k:v for k,v in self.weapons.items() if "type" in v}
        
        # Data reaksi elemen
        self.elemental_reactions = {
            # Pyro reactions
            ('pyro', 'hydro'): ('vaporize', 2.0),  # Reverse vaporize
            ('hydro', 'pyro'): ('vaporize', 1.5),  # Forward vaporize
            ('pyro', 'cryo'): ('melt', 2.0),      # Reverse melt
            ('cryo', 'pyro'): ('melt', 1.5),      # Forward melt
            ('pyro', 'electro'): ('overloaded', 1.0),
            ('pyro', 'dendro'): ('burning', 1.0),
            
            # Hydro reactions
            ('hydro', 'electro'): ('electro-charged', 1.0),
            ('hydro', 'dendro'): ('bloom', 1.0),
            
            # Electro reactions
            ('electro', 'cryo'): ('superconduct', 0.5),
            ('electro', 'dendro'): ('quicken', 1.0),
            
            # Dendro reactions
            ('dendro', 'pyro'): ('burning', 1.0),
            ('dendro', 'electro'): ('quicken', 1.0),
            
            # Geo reactions (crystallize)
            ('geo', 'pyro'): ('pyro-crystallize', 0.0),
            ('geo', 'hydro'): ('hydro-crystallize', 0.0),
            ('geo', 'electro'): ('electro-crystallize', 0.0),
            ('geo', 'cryo'): ('cryo-crystallize', 0.0),
            
            # Anemo reactions (swirl)
            ('anemo', 'pyro'): ('pyro-swirl', 0.5),
            ('anemo', 'hydro'): ('hydro-swirl', 0.5),
            ('anemo', 'electro'): ('electro-swirl', 0.5),
            ('anemo', 'cryo'): ('cryo-swirl', 0.5),
        }
        
        # Load dynamic artifact recommendations
        self.artifact_recommendations = self._load_artifact_recommendations()
        
        # Untuk fitur rotasi tim
        self.team_rotations = []
        self.current_rotation_index = 0
        
        # Untuk fitur penyimpanan
        self.config_dir = os.path.join(os.path.expanduser("~"), ".genteamopt")
        os.makedirs(self.config_dir, exist_ok=True)
        
        self._validate_data()
        self.load_config()

    def _load_artifact_recommendations(self):
        """Load artifact recommendations from artifact data"""
        recommendations = {
            "main-dps": defaultdict(list),
            "sub-dps": defaultdict(list),
            "support": defaultdict(list),
            "utility": defaultdict(list)
        }
        
        element_keywords = {
            "pyro": ["pyro", "fire", "flame", "crimson", "witch"],
            "hydro": ["hydro", "water", "depth", "ocean"],
            "electro": ["electro", "thunder", "lightning", "fury"],
            "cryo": ["cryo", "ice", "blizzard", "snow"],
            "geo": ["geo", "stone", "petra", "husk"],
            "anemo": ["anemo", "wind", "viridescent"],
            "dendro": ["dendro", "wood", "deepwood", "gilded"]
        }
        
        for artifact_name, artifact_data in self.artifacts.items():
            if not artifact_data:
                continue
                
            art_name_lower = artifact_name.lower()
            art_type = str(artifact_data.get("type", "")).lower()
            art_desc = str(artifact_data.get("description", "")).lower()
            
            # Determine artifact roles based on type and description
            roles = []
            if "dps" in art_type or "damage" in art_desc:
                roles.append("main-dps")
                roles.append("sub-dps")
            if "support" in art_type or "heal" in art_desc or "noblesse" in art_name_lower:
                roles.append("support")
            if "utility" in art_type or "shield" in art_desc or "tenacity" in art_name_lower:
                roles.append("utility")
                
            if not roles:  # Default fallback
                roles = ["main-dps", "sub-dps", "support", "utility"]
            
            # Determine element based on artifact name and description
            element = None
            for ele, keywords in element_keywords.items():
                if any(keyword in art_name_lower for keyword in keywords) or any(keyword in art_desc for keyword in keywords):
                    element = ele
                    break
            
            # Add to recommendations
            for role in roles:
                if element:
                    recommendations[role][element].append(artifact_name)
                recommendations[role]["default"].append(artifact_name)
        
        # Remove duplicates and sort
        for role in recommendations:
            for element in recommendations[role]:
                recommendations[role][element] = sorted(list(set(recommendations[role][element])))
        
        return recommendations

    def _validate_data(self):
        """Validate character and weapon data structure"""
        required_char_fields = ['name', 'vision', 'weapon_type', 'skillTalents']
        required_weapon_fields = ['name', 'type', 'baseAttack']
        
        print("\nMemvalidasi data...")
        
        for char_name, char_data in self.characters.items():
            for field in required_char_fields:
                if field not in char_data:
                    print(f"Warning: Karakter {char_name} tidak memiliki field {field}")
            
            # Normalize vision data
            if 'vision' in char_data:
                char_data['vision'] = str(char_data['vision']).strip().capitalize()

        for weap_name, weap_data in self.weapons.items():
            for field in required_weapon_fields:
                if field not in weap_data:
                    print(f"Warning: Senjata {weap_name} tidak memiliki field {field}")

    def _load_category(self, category: str) -> Dict:
        """Load data from category folder with better validation"""
        category_path = os.path.join(self.data_root, category)
        data = {}
    
        if not os.path.exists(category_path):
            print(f"Warning: Folder '{category}' tidak ditemukan di {category_path}")
            return data
        
        for item_dir in os.listdir(category_path):
            item_path = os.path.join(category_path, item_dir, "en.json")
        
            if os.path.exists(item_path):
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        item_data = json.load(f)

                        # Buat nama lebih rapi
                        pretty_name = item_data.get("name", item_dir).replace("-", " ").title()
                        item_data["name"] = pretty_name
                        data[pretty_name] = item_data
                    
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error: Gagal memuat {item_path} - {str(e)}")
            else:
                print(f"Warning: File en.json tidak ditemukan di {os.path.join(category_path, item_dir)}")
            
        return data


    def _convert_damage_value(self, value_str: str) -> float:
        """Convert damage value string to float with better handling of various formats"""
        try:
            # Handle empty or invalid input
            if not value_str or not isinstance(value_str, str):
                return 0.0
            
            value_str = value_str.strip().lower()
        
            # Handle empty string
            if not value_str:
                return 0.0
        
            # Handle time values (e.g. "6s", "12.0s")
            if 's' in value_str:
                return 0.0  # Skip time values as they don't contribute to damage
        
            # Handle gender-specific values (e.g. "Male: 55.9% + 72.2% Female: 55.9% + 60.7%")
            if 'male:' in value_str or 'female:' in value_str:
                # Take the first value after male/female marker
                parts = [p.strip() for p in re.split(r'male:|female:', value_str) if p.strip()]
                if parts:
                    return self._convert_damage_value(parts[0].split()[0])
                return 0.0
        
            # Handle combined values (e.g. "82.08% ATK + 102.6% DEF")
            if '+' in value_str:
                parts = value_str.split('+')
                # Take the first part only
                first_part = parts[0].strip()
                # Extract numeric value from first part
                match = re.search(r'(\d+\.?\d*)', first_part)
                if match:
                    return float(match.group(1))
                return 0.0
        
            # Handle division format (e.g. "5.14 / 3.96")
            if '/' in value_str:
                parts = value_str.split('/')
                try:
                    return float(parts[0].strip())
                except:
                    return 0.0
        
            # Handle percentage values
            if '%' in value_str:
                num_part = value_str.split('%')[0]
                # Remove any non-numeric characters except . and -
                num_part = re.sub(r'[^\d.-]', '', num_part)
                if num_part:
                    return float(num_part)
                return 0.0
        
            # Handle simple numeric values
            # Remove any non-numeric characters except . and -
            clean_str = re.sub(r'[^\d.-]', '', value_str)
            if clean_str:
                return float(clean_str)
        
            return 0.0
        
        except Exception as e:
            print(f"Warning: Gagal konversi damage value: {value_str} - {str(e)}")
            return 0.0

    def _get_elemental_resonance_bonus(self, elements: List[str]) -> float:
        """Calculate elemental resonance bonus"""
        element_counts = Counter(elements)
        
        if element_counts.get('pyro', 0) >= 2:
            return 1.25
        if element_counts.get('hydro', 0) >= 2:
            return 1.25
        if element_counts.get('cryo', 0) >= 2:
            return 1.15
        if element_counts.get('electro', 0) >= 2:
            return 1.20
        if element_counts.get('geo', 0) >= 2:
            return 1.15
        if element_counts.get('anemo', 0) >= 2:
            return 1.10
        if element_counts.get('dendro', 0) >= 2:
            return 1.15
            
        return 1.0

    def _get_resonance_name(self, elements: List[str]) -> str:
        """Get resonance name based on elements"""
        element_counts = Counter(elements)
        
        if element_counts.get('pyro', 0) >= 2:
            return "Pyro (ATK Bonus)"
        if element_counts.get('hydro', 0) >= 2:
            return "Hydro (HP Bonus)"
        if element_counts.get('cryo', 0) >= 2:
            return "Cryo (CRIT Rate)"
        if element_counts.get('electro', 0) >= 2:
            return "Electro (Energy Bonus)"
        if element_counts.get('geo', 0) >= 2:
            return "Geo (Shield Strength)"
        if element_counts.get('anemo', 0) >= 2:
            return "Anemo (Stamina Reduction)"
        if element_counts.get('dendro', 0) >= 2:
            return "Dendro (Elemental Mastery)"
        return "Tidak ada"

    def _calculate_reaction_potential(self, elements: List[str]) -> float:
        """Calculate potential elemental reaction damage for a team"""
        if len(elements) < 2:
            return 0.0
            
        reaction_score = 0.0
        reaction_count = Counter()
        
        # Hitung semua kemungkinan reaksi elemen
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                elem1, elem2 = elements[i], elements[j]
                reaction = self.elemental_reactions.get((elem1, elem2)) or self.elemental_reactions.get((elem2, elem1))
                
                if reaction:
                    reaction_type, multiplier = reaction
                    reaction_count[reaction_type] += 1
                    reaction_score += multiplier * 100  # Base damage multiplier
        
        # Berikan bonus untuk tim dengan reaksi yang beragam
        diversity_bonus = len(reaction_count) * 50
        reaction_score += diversity_bonus
        
        return reaction_score

    def _get_character_power(self, char_name: str, weapon: Optional[str]) -> float:
        """Calculate character's individual power"""
        if char_name not in self.characters:
            return 0.0
        
        char = self.characters[char_name]
        power = 0.0
    
        # Base stats - berikan nilai default jika tidak ada
        power += char.get('baseATK', 0) * 0.5
        power += char.get('baseHP', 0) * 0.1
        power += char.get('baseDEF', 0) * 0.1
    
        # Weapon bonus - pastikan weapon tidak None
        if weapon and weapon in self.weapons:
            weap_data = self.weapons[weapon]
            power += weap_data.get('baseAttack', 0) * 0.8
            power += weap_data.get('secondaryStat', {}).get('value', 0) * 0.5
    
        # Talent scaling - dengan penanganan error yang lebih baik
        if 'skillTalents' in char:
            for talent in char['skillTalents']:
                if talent.get('type') in ['NORMAL_ATTACK', 'ELEMENTAL_SKILL', 'ELEMENTAL_BURST']:
                    for upgrade in talent.get('upgrades', []):
                        if 'DMG' in upgrade.get('name', ''):
                            try:
                                power += self._convert_damage_value(upgrade.get('value', '0')) * 10
                            except:
                                continue  # Lewati jika ada error konversi
    
        return power

    def _determine_role(self, char_name: str, current_team: List[Dict]) -> str:
        """Determine the best role for a character based on team composition"""
        if char_name not in self.characters:
            return "support"
            
        char = self.characters[char_name]
        weapon_type = char.get('weapon_type', '').lower()
        
        # Count existing roles
        role_counts = Counter(m.get('role') for m in current_team)
        
        # Prioritize roles needed in team
        if role_counts['main-dps'] == 0:
            # Check if this character can be DPS
            if weapon_type in ['sword', 'claymore', 'polearm', 'catalyst', 'bow']:
                return "main-dps"
        
        if role_counts['sub-dps'] < 2:
            # Check if character has good elemental skills
            if 'skillTalents' in char:
                for talent in char['skillTalents']:
                    if talent.get('type') == 'ELEMENTAL_SKILL' and 'DMG' in talent.get('name', ''):
                        return "sub-dps"
        
        # Default to support
        return "support"

    def calculate_team_score(self, team: List[Dict], user_weapons: Dict[str, str]) -> float:
        if not team:
            return 0.0

        # 1. Hitung skor individual karakter dengan NumPy
        individual_scores = np.array([
            self._get_character_power(member["name"], member.get("weapon"))
            for member in team
            if member.get("name") in self.characters
        ])
        avg_individual_score = np.mean(individual_scores)  # Rerata lebih representatif

        # 2. Hitung sinergi elemen (lebih ringkas)
        elements = [
            str(self.characters[member["name"]]["vision"]).lower().strip()
            for member in team
            if member.get("name") in self.characters
        ]
        resonance_bonus = self._get_elemental_resonance_bonus(elements)
        reaction_potential = self._calculate_reaction_potential(elements)

        # 3. Hitung bonus role dengan NumPy where
        roles = np.array([member.get("role", "") for member in team])
        role_bonus = 1.0
        if np.any(roles == "main-dps") and np.any(roles == "support"):
            role_bonus = 1.2
        elif np.any(roles == "sub-dps") and np.any(roles == "utility"):
            role_bonus = 1.1

        # 4. Gabungkan dengan dot product NumPy
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        factors = np.array([
            avg_individual_score,
            resonance_bonus * 3000,
            reaction_potential,
            role_bonus * 1000
        ])
        
        total_score = np.dot(weights, factors)
        return round(total_score, 2)

    def a_star_search(self, user_chars: List[str], user_weapons: Dict[str, str], user_artifacts: List[str]) -> List[Dict]:
        """A* Search algorithm for optimal team composition"""
        class TeamNode:
            def __init__(self, team, remaining_chars, g_score, h_score):
                self.team = team
                self.remaining_chars = remaining_chars
                self.g_score = g_score  # Current score
                self.h_score = h_score  # Heuristic score
                self.f_score = g_score + h_score  # Total score
            
            def __lt__(self, other):
                return self.f_score < other.f_score
            
            def is_goal(self):
                return len(self.team) == 4
        
        def heuristic(node: TeamNode) -> float:
            """Estimate the best possible score from current state"""
            if len(node.team) == 4:
                return 0
            
            # Estimasi skor terbaik dengan menambahkan karakter terkuat yang tersisa
            estimated_score = node.g_score
            best_chars = sorted(node.remaining_chars, 
                              key=lambda x: self._get_character_power(x, user_weapons.get(x)), 
                              reverse=True)
            
            for i in range(4 - len(node.team)):
                if i < len(best_chars):
                    char = best_chars[i]
                    estimated_score += self._get_character_power(char, user_weapons.get(char))
            
            return estimated_score * 1.2  # Optimisitic heuristic
        
        open_set = []
        start_node = TeamNode([], user_chars.copy(), 0, heuristic(TeamNode([], user_chars.copy(), 0, 0)))
        heappush(open_set, (start_node.f_score, random.random(), start_node))  # random.random() untuk tie-breaker
        
        best_teams = []
        
        while open_set and len(best_teams) < 10:
            _, _, current = heappop(open_set)
            
            if current.is_goal():
                team_score = self.calculate_team_score(current.team, user_weapons)
                best_teams.append({
                    "team": current.team,
                    "score": team_score
                })
                continue
                
            for char in current.remaining_chars:
                new_team = current.team + [{
                    "name": char,
                    "role": self._determine_role(char, current.team),
                    "weapon": user_weapons.get(char)
                }]
                
                new_remaining = [c for c in current.remaining_chars if c != char]
                g_score = self.calculate_team_score(new_team, user_weapons)
                h_score = heuristic(TeamNode(new_team, new_remaining, g_score, 0))
                
                new_node = TeamNode(new_team, new_remaining, g_score, h_score)
                heappush(open_set, (new_node.f_score, random.random(), new_node))
        
        return sorted(best_teams, key=lambda x: x["score"], reverse=True)[:5]

    def simulated_annealing(self, user_chars: List[str], user_weapons: Dict[str, str], user_artifacts: List[str]) -> List[Dict]:
        """Simulated Annealing algorithm for team optimization"""
        def random_team():
            team = []
            chars = user_chars.copy()
            random.shuffle(chars)
            
            for i in range(min(4, len(chars))):
                team.append({
                    "name": chars[i],
                    "role": self._determine_role(chars[i], team[:i]),
                    "weapon": user_weapons.get(chars[i])
                })
            return team
        
        current_team = random_team()
        current_score = self.calculate_team_score(current_team, user_weapons)
        
        best_team = current_team.copy()
        best_score = current_score
        
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 0.01
        
        while temperature > min_temperature:
            # Generate neighbor solution
            neighbor = current_team.copy()
            
            # Random modification (swap, add, or remove)
            mod_type = random.choice(["swap", "change_role"])
            
            if mod_type == "swap" and len(user_chars) > 4:
                # Swap a team member with a non-member
                team_indices = [i for i, char in enumerate(user_chars) if any(m['name'] == char for m in neighbor)]
                non_team_indices = [i for i, char in enumerate(user_chars) if not any(m['name'] == char for m in neighbor)]
                
                if team_indices and non_team_indices:
                    swap_out = random.choice(range(len(neighbor)))
                    swap_in = random.choice(non_team_indices)
                    
                    neighbor[swap_out] = {
                        "name": user_chars[swap_in],
                        "role": self._determine_role(user_chars[swap_in], [m for j, m in enumerate(neighbor) if j != swap_out]),
                        "weapon": user_weapons.get(user_chars[swap_in])
                    }
            
            elif mod_type == "change_role":
                # Change role of a random member
                change_idx = random.choice(range(len(neighbor)))
                possible_roles = ["main-dps", "sub-dps", "support", "utility"]
                current_role = neighbor[change_idx].get("role")
                possible_roles.remove(current_role)
                
                neighbor[change_idx]["role"] = random.choice(possible_roles)
            
            # Calculate neighbor score
            neighbor_score = self.calculate_team_score(neighbor, user_weapons)
            
            # Decide whether to accept the neighbor
            if neighbor_score > current_score:
                current_team = neighbor
                current_score = neighbor_score
                
                if neighbor_score > best_score:
                    best_team = neighbor.copy()
                    best_score = neighbor_score
            else:
                # Calculate acceptance probability
                prob = math.exp((neighbor_score - current_score) / temperature)
                if random.random() < prob:
                    current_team = neighbor
                    current_score = neighbor_score
            
            # Cool the temperature
            temperature *= cooling_rate
        
        return [{
            "team": best_team,
            "score": best_score
        }]

    def hill_climbing(self, user_chars: List[str], user_weapons: Dict[str, str], user_artifacts: List[str]) -> List[Dict]:
        """Hill Climbing algorithm for team optimization"""
        def generate_random_team():
            team = []
            chars = random.sample(user_chars, min(4, len(user_chars)))
            
            for i, char in enumerate(chars):
                team.append({
                    "name": char,
                    "role": self._determine_role(char, team[:i]),
                    "weapon": user_weapons.get(char)
                })
            return team
        
        current_team = generate_random_team()
        current_score = self.calculate_team_score(current_team, user_weapons)
        
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            neighbors = []
            
            # Generate all possible neighbors by swapping one character
            for i in range(len(current_team)):
                for char in user_chars:
                    if char not in [m['name'] for m in current_team]:
                        neighbor = current_team.copy()
                        neighbor[i] = {
                            "name": char,
                            "role": self._determine_role(char, [m for j, m in enumerate(neighbor) if j != i]),
                            "weapon": user_weapons.get(char)
                        }
                        neighbors.append(neighbor)
            
            # Also generate neighbors by changing roles
            for i in range(len(current_team)):
                possible_roles = ["main-dps", "sub-dps", "support", "utility"]
                current_role = current_team[i].get("role")
                possible_roles.remove(current_role)
                
                for new_role in possible_roles:
                    neighbor = current_team.copy()
                    neighbor[i]["role"] = new_role
                    neighbors.append(neighbor.copy())
            
            # Evaluate all neighbors
            for neighbor in neighbors:
                neighbor_score = self.calculate_team_score(neighbor, user_weapons)
                
                if neighbor_score > current_score:
                    current_team = neighbor
                    current_score = neighbor_score
                    improved = True
                    break
            
            iteration += 1
        
        return [{
            "team": current_team,
            "score": current_score
        }]

    def _get_recommended_artifacts(self, character: Dict, role: str, available_artifacts: List[str]) -> List[str]:
        """Get recommended artifacts based on actual artifact data"""
        if not character or not role or not available_artifacts:
            return ["Belum dioptimalkan"]
        
        element = str(character.get("vision", "")).lower().strip()
        role = role.lower()
        
        # Get recommendations from our loaded data
        role_recs = self.artifact_recommendations.get(role, {})
        
        recommendations = []
        
        # 1. Element-specific artifacts
        if element in role_recs:
            for art in role_recs[element]:
                if art in available_artifacts and art not in recommendations:
                    recommendations.append(art)
        
        # 2. Default artifacts for role
        for art in role_recs.get("default", []):
            if art in available_artifacts and art not in recommendations:
                recommendations.append(art)
        
        # 3. Other artifacts that match element
        element_keywords = {
            "pyro": ["pyro", "fire", "flame", "crimson", "witch"],
            "hydro": ["hydro", "water", "depth", "ocean"],
            "electro": ["electro", "thunder", "lightning", "fury"],
            "cryo": ["cryo", "ice", "blizzard", "snow"],
            "geo": ["geo", "stone", "petra", "husk"],
            "anemo": ["anemo", "wind", "viridescent"],
            "dendro": ["dendro", "wood", "deepwood", "gilded"]
        }
        
        if element in element_keywords:
            for art in available_artifacts:
                art_lower = art.lower()
                if any(keyword in art_lower for keyword in element_keywords[element]):
                    if art not in recommendations:
                        recommendations.append(art)
        
        # 4. Other available artifacts (exclude low-tier)
        other_arts = [art for art in available_artifacts 
                     if art not in recommendations
                     and art.lower() not in ["adventurer", "berserker", "lucky-dog", "traveling-doctor"]]
        
        return (recommendations + other_arts)[:3] or ["Belum dioptimalkan"]

    def _get_valid_input(self, options: List[str], prompt: str, input_type: str) -> List[str]:
        """Get validated user input"""
        while True:
            print(f"\n{prompt} ({len(options)} {input_type} tersedia):")
            print(", ".join(sorted(options)))
            
            user_input = input(f"Masukkan {input_type} (pisahkan dengan koma): ").strip()
            if not user_input:
                print(f"Warning: Anda harus memasukkan minimal 1 {input_type}!")
                continue
                
            selected = [item.strip() for item in user_input.split(',')]
            invalid_items = [item for item in selected if item not in options]
            
            if invalid_items:
                print(f"Error: {input_type.capitalize()} tidak valid - {', '.join(invalid_items)}")
                continue
                
            return selected

    def get_user_inventory(self) -> Tuple[List[str], Dict[str, str], List[str]]:
        """Get user's inventory with robust error handling"""
        print("\n" + "="*50)
        print("PEMBENTUKAN TIM GENSHIN IMPACT")
        print("="*50)
        
        # Get characters
        print("\nSTEP 1: Pilih karakter yang dimiliki")
        user_chars = self._get_valid_input(
            list(self.characters.keys()),
            "Karakter yang tersedia",
            "karakter"
        )
        
        # Group by weapon type with validation
        weapon_type_groups = defaultdict(list)
        for char in user_chars:
            char_data = self.characters.get(char, {})
            if not char_data:
                print(f"Warning: Data karakter {char} tidak valid, diabaikan")
                continue
                
            weapon_type = char_data.get("weapon_type", "").lower()
            if not weapon_type:
                print(f"Warning: Karakter {char} tidak memiliki tipe senjata, diabaikan")
                continue
                
            weapon_type_groups[weapon_type].append(char)
        
        # Get weapons with type validation
        user_weapons = {}
        print("\nSTEP 2: Pilih senjata untuk setiap tipe senjata")
        
        for weapon_type, chars in weapon_type_groups.items():
            compatible_weapons = {}
            for weap, data in self.weapons.items():
                weap_type = str(data.get("type", "")).lower()
                if weap_type == weapon_type:
                    compatible_weapons[weap] = data
            
            if not compatible_weapons:
                print(f"\nWarning: Tidak ada senjata tipe {weapon_type} yang valid")
                for char in chars:
                    user_weapons[char] = None
                continue
                
            print(f"\nKarakter {', '.join(chars)} menggunakan senjata tipe {weapon_type}")
            selected_weapons = self._get_valid_input(
                list(compatible_weapons.keys()),
                f"Senjata {weapon_type} yang dimiliki",
                "senjata"
            )
            
            # Dalam loop distribusi senjata:
            for i, char in enumerate(chars):
                if i < len(selected_weapons):
                    user_weapons[char] = selected_weapons[i]
                else:
                    # Berikan senjata default jika tidak cukup
                    user_weapons[char] = list(compatible_weapons.keys())[0] if compatible_weapons else None
        
        # Get artifacts
        print("\nSTEP 3: Pilih artefak yang dimiliki")
        user_artifacts = self._get_valid_input(
            list(self.artifacts.keys()),
            "Artefak yang tersedia",
            "artefak"
        )
        
        return user_chars, user_weapons, user_artifacts

    def generate_teams(self, user_chars: List[str], user_weapons: Dict[str, str], user_artifacts: List[str], algorithm: str = 'all') -> Dict[str, List[Dict]]:
        """Generate teams and visualize comparison using Seaborn"""
        if not user_chars or len(user_chars) < 4:
            print("Error: Minimal butuh 4 karakter")
            return {}

        # Validasi karakter
        valid_chars = [char for char in user_chars if char in self.characters]
        if len(valid_chars) < 4:
            print("Error: Tidak cukup karakter valid (minimal 4)")
            return {}

        # Jalankan semua algoritma jika 'all' dipilih
        algorithms = {
            'a_star': self.a_star_search,
            'simulated_annealing': self.simulated_annealing,
            'hill_climbing': self.hill_climbing
        }

        results = {}
        for algo_name, algo_func in algorithms.items():
            if algorithm == 'all' or algorithm == algo_name:
                teams = algo_func(valid_chars, user_weapons, user_artifacts)
                results[algo_name] = teams

        # Visualisasi jika membandingkan semua algoritma
        if len(results) > 1:
            self._visualize_algorithm_comparison(results)

        return results if algorithm == 'all' else results.get(algorithm, [])
    
    def _visualize_algorithm_comparison(self, results: Dict[str, List[Dict]]):
        """Visualisasi perbandingan skor tim antar algoritma menggunakan Seaborn"""
        plt.figure(figsize=(10, 6))
        
        # Siapkan data
        algo_names = []
        best_scores = []
        
        for algo_name, teams in results.items():
            if not teams:
                continue
            best_score = max(team["score"] for team in teams)
            algo_names.append(algo_name.replace('_', ' ').title())
            best_scores.append(best_score)
        
        # Buat plot
        sns.barplot(
            x=algo_names,
            y=best_scores,
            palette="viridis",
            hue=algo_names,
            legend=False
        )
        
        plt.title("Perbandingan Skor Tim Tertinggi per Algoritma", pad=20)
        plt.xlabel("Algoritma")
        plt.ylabel("Skor Tim")
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        # Simpan dan tampilkan
        plt.savefig("algorithm_comparison.png")
        plt.show()

    def create_team_rotation(self, teams: List[Dict]) -> None:
        """Membuat rotasi tim dari beberapa tim yang direkomendasikan"""
        if not teams:
            print("Error: Tidak ada tim yang tersedia untuk rotasi")
            return
            
        self.team_rotations = teams
        self.current_rotation_index = 0
        print("\nRotasi tim berhasil dibuat!")
        
    def next_rotation(self) -> Optional[Dict]:
        """Memutar ke tim berikutnya dalam rotasi"""
        if not self.team_rotations:
            print("Tidak ada rotasi tim yang aktif")
            return None
            
        self.current_rotation_index = (self.current_rotation_index + 1) % len(self.team_rotations)
        return self.team_rotations[self.current_rotation_index]
        
    def prev_rotation(self) -> Optional[Dict]:
        """Memutar ke tim sebelumnya dalam rotasi"""
        if not self.team_rotations:
            print("Tidak ada rotasi tim yang aktif")
            return None
            
        self.current_rotation_index = (self.current_rotation_index - 1) % len(self.team_rotations)
        return self.team_rotations[self.current_rotation_index]
        
    def display_current_rotation(self, user_artifacts: List[str]) -> None:
        """Menampilkan tim saat ini dalam rotasi"""
        if not self.team_rotations:
            print("Tidak ada rotasi tim yang aktif")
            return
            
        current_team = self.team_rotations[self.current_rotation_index]
        print(f"\nTim saat ini dalam rotasi ({self.current_rotation_index + 1}/{len(self.team_rotations)})")
        self.display_teams([current_team], user_artifacts)

    def save_config(self) -> None:
        """Menyimpan konfigurasi pengguna ke file"""
        config = {
            'last_used': datetime.now().isoformat(),
            'user_chars': list(self.characters.keys()),
            'user_weapons': self.weapons,
            'user_artifacts': list(self.artifacts.keys()),
            'team_rotations': self.team_rotations,
            'current_rotation_index': self.current_rotation_index
        }
        
        config_path = os.path.join(self.config_dir, "config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print("\nKonfigurasi berhasil disimpan!")
        except Exception as e:
            print(f"\nGagal menyimpan konfigurasi: {str(e)}")
    
    def load_config(self) -> None:
        """Memuat konfigurasi pengguna dari file"""
        config_path = os.path.join(self.config_dir, "config.json")
        
        if not os.path.exists(config_path):
            return
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # Validasi data yang dimuat
            if 'user_chars' in config:
                self.characters = {k: v for k, v in self.characters.items() if k in config['user_chars']}
                
            if 'user_weapons' in config:
                self.weapons = config['user_weapons']
                
            if 'user_artifacts' in config:
                self.artifacts = {k: v for k, v in self.artifacts.items() if k in config['user_artifacts']}
                
            if 'team_rotations' in config:
                self.team_rotations = config['team_rotations']
                
            if 'current_rotation_index' in config:
                self.current_rotation_index = config['current_rotation_index']
                
            print("\nKonfigurasi berhasil dimuat!")
        except Exception as e:
            print(f"\nGagal memuat konfigurasi: {str(e)}")

    def display_menu(self) -> None:
        """Menampilkan menu utama dengan tampilan yang lebih baik"""
        print("\n" + "="*50)
        print(" GENSHIN TEAM OPTIMIZER ".center(50, "="))
        print("="*50)
        print("\nMenu Utama:")
        print("1. Bentuk Tim Baru")
        print("2. Lihat Rekomendasi Tim")
        print("3. Rotasi Tim")
        print("4. Simpan Konfigurasi")
        print("5. Muat Konfigurasi")
        print("6. Keluar")
        
    def display_teams(self, teams: List[Dict], user_artifacts: List[str]):
        """Display team recommendations with better UI"""
        if not teams:
            print("\nTidak ada tim yang tersedia untuk ditampilkan")
            return
            
        print("\n" + "="*50)
        print(" REKOMENDASI TIM OPTIMAL ".center(50, "="))
        print("="*50)
        
        for i, team_data in enumerate(teams, 1):
            if not team_data or "team" not in team_data:
                continue
                
            elements = []
            for member in team_data["team"]:
                char_name = member.get("name")
                if char_name and char_name in self.characters:
                    element = str(self.characters[char_name].get("vision", "")).lower().strip()
                    if element:
                        elements.append(element)
            
            resonance = self._get_elemental_resonance_bonus(elements)
            
            print(f"\n{' TIM #' + str(i) + ' ':=^50}")
            print(f"| {'Skor Tim:':<20} {team_data.get('score', 0):.2f}{' ':>24} |")
            
            if resonance > 1.0:
                res_name = self._get_resonance_name(elements)
                res_bonus = f"+{(resonance-1)*100:.0f}%"
                padding = 24 - len(res_name) - len(res_bonus)
                print(f"| {'Elemental Resonance:':<20} {res_name} ({res_bonus}){' ' * padding} |")
            
            print(f"{'':=^50}")
            
            for member in team_data["team"]:
                char_name = member.get("name")
                if not char_name or char_name not in self.characters:
                    continue
                    
                char = self.characters[char_name]
                role = member.get("role", "unknown").lower()
                recommended_arts = self._get_recommended_artifacts(char, role, user_artifacts)
                
                print(f"\n  {role.upper():<10} {char.get('name', 'Unknown')} ({char.get('vision', 'Unknown')})")
                print(f"    {'Tipe Senjata:':<15} {char.get('weapon_type', 'Unknown').capitalize()}")
                print(f"    {'Senjata:':<15} {member.get('weapon', 'Tidak ada')}")
                print(f"    {'Artefak Utama:':<15} {recommended_arts[0]}")
                if len(recommended_arts) > 1:
                    print(f"    {'Artefak Alternatif:':<15} {', '.join(recommended_arts[1:3])}")
                    
            print(f"{'':=^50}")

    def display_rotation_menu(self):
        """Menampilkan menu rotasi tim"""
        print("\n" + "="*50)
        print(" MANAJEMEN ROTASI TIM ".center(50, "="))
        print("="*50)
        print("1. Buat Rotasi Tim dari Rekomendasi")
        print("2. Tim Berikutnya")
        print("3. Tim Sebelumnya")
        print("4. Kembali ke Menu Utama")

    def run(self):
        """Main program flow with new features"""
        top_teams = []
        user_artifacts = []
        
        while True:
            self.display_menu()
            choice = input("\nPilih menu (1-6): ").strip()
            
            try:
                if choice == "1":
                    # Bentuk tim baru
                    user_chars, user_weapons, user_artifacts = self.get_user_inventory()
                    
                    if len(user_chars) < 4:
                        print("\nError: Membutuhkan minimal 4 karakter!")
                        continue
                        
                    print("\nMembentuk tim optimal...")
                    
                    # Pilih algoritma
                    print("\nPilih algoritma optimasi:")
                    print("1. A* Search (Rekomendasi)")
                    print("2. Simulated Annealing")
                    print("3. Hill Climbing")
                    algo_choice = input("Masukkan pilihan (1-3): ").strip()
                    
                    if algo_choice == "1":
                        top_teams = self.generate_teams(user_chars, user_weapons, user_artifacts, 'a_star')
                    elif algo_choice == "2":
                        top_teams = self.generate_teams(user_chars, user_weapons, user_artifacts, 'simulated_annealing')
                    elif algo_choice == "3":
                        top_teams = self.generate_teams(user_chars, user_weapons, user_artifacts, 'hill_climbing')
                    else:
                        print("Pilihan tidak valid, menggunakan A* Search default")
                        top_teams = self.generate_teams(user_chars, user_weapons, user_artifacts, 'a_star')
                    
                    if top_teams:
                        self.display_teams(top_teams, user_artifacts)
                    else:
                        print("\nGagal membentuk tim. Pastikan data karakter dan senjata valid.")
                    
                elif choice == "2":
                    # Lihat rekomendasi tim
                    if not top_teams:
                        print("\nBelum ada rekomendasi tim. Silakan buat tim baru terlebih dahulu.")
                    else:
                        self.display_teams(top_teams, user_artifacts)
                        
                elif choice == "3":
                    # Menu rotasi tim
                    if not top_teams:
                        print("\nBelum ada rekomendasi tim. Silakan buat tim baru terlebih dahulu.")
                        continue
                        
                    while True:
                        self.display_rotation_menu()
                        rotation_choice = input("\nPilih menu rotasi (1-4): ").strip()
                        
                        if rotation_choice == "1":
                            self.create_team_rotation(top_teams)
                            self.display_current_rotation(user_artifacts)
                        elif rotation_choice == "2":
                            self.next_rotation()
                            self.display_current_rotation(user_artifacts)
                        elif rotation_choice == "3":
                            self.prev_rotation()
                            self.display_current_rotation(user_artifacts)
                        elif rotation_choice == "4":
                            break
                        else:
                            print("Pilihan tidak valid!")
                            
                elif choice == "4":
                    # Simpan konfigurasi
                    self.save_config()
                    
                elif choice == "5":
                    # Muat konfigurasi
                    self.load_config()
                    
                elif choice == "6":
                    # Keluar
                    print("\nTerima kasih telah menggunakan Genshin Team Optimizer!")
                    break
                    
                else:
                    print("Pilihan tidak valid!")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Silakan coba lagi.")