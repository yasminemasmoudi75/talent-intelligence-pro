import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

# --- Configuration de la page ---
st.set_page_config(
    page_title="Talent Intelligence Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styles CSS Personnalisés (Premium & Moderne) ---
st.markdown("""
<style>
    /* Importation de la police Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global App Style */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Titres */
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Cards & Containers */
    .css-1r6slb0 {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* Custom Metric Card */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        padding-bottom: 10px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #6b7280;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #4f46e5;
        border-bottom: 2px solid #4f46e5;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# --- Chargement des ressources ---
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
        feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
        
        # Charger les métriques
        try:
            with open("metrics.json", "r") as f:
                model_metrics = json.load(f)
        except FileNotFoundError:
            model_metrics = None
            
        return model, scaler, label_encoders, feature_columns, model_metrics
    except FileNotFoundError:
        return None, None, None, None, None

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

model, scaler, label_encoders, feature_columns, model_metrics = load_models()
dataset = load_dataset("employee_performance_data.csv")

# --- Header ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("<div style='font-size: 3rem; text-align: center;'>🔮</div>", unsafe_allow_html=True)
with col_title:
    st.title("Talent Intelligence Pro")
    st.markdown("### Système d'Analyse Prédictive & Dashboard RH")

# --- Sidebar ---
with st.sidebar:
    st.header("Navigation")
    st.info("Bienvenue sur votre espace d'analyse RH.")
    
    if dataset is not None:
        st.markdown("---")
        st.subheader("Aperçu des Données")
        st.markdown(f"**Total Employés:** {len(dataset)}")
        st.markdown(f"**Départements:** {dataset['department'].nunique()}")
        st.markdown(f"**Score Moyen:** {dataset['performance_score'].mean():.2f}")
    
    st.markdown("---")
    st.markdown("Created with ❤️ by AI")

# --- Onglets Principaux ---
tab_dashboard, tab_analysis, tab_prediction = st.tabs(["📊 Dashboard Global", "🔎 Analyse Approfondie", "🎯 Simulateur de Performance"])

# ==========================================
# TAB 1: DASHBOARD GLOBAL
# ==========================================
with tab_dashboard:
    if dataset is None:
        st.error("⚠️ Fichier de données 'employee_performance_data.csv' introuvable.")
    else:
        st.markdown("#### 📈 Vue d'ensemble de la performance")
        
        # KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{dataset['performance_score'].mean():.2f}</div>
                <div class="metric-label">Score Moyen Global</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{dataset['attendance_rate'].mean():.1f}%</div>
                <div class="metric-label">Taux de Présence Moyen</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{dataset['training_hours'].mean():.1f}h</div>
                <div class="metric-label">Heures de Formation Moy.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi4:
            high_perf_count = len(dataset[dataset['performance_score'] >= 4]) # Assumption: 4+ is high
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{high_perf_count}</div>
                <div class="metric-label">Top Performers (4+)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Charts Row 1
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Répartition par Département")
            dept_counts = dataset['department'].value_counts().reset_index()
            dept_counts.columns = ['Département', 'Nombre']
            fig_dept = px.pie(dept_counts, values='Nombre', names='Département', hole=0.4, 
                              color_discrete_sequence=px.colors.qualitative.Prism)
            fig_dept.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_dept, use_container_width=True)
            
        with c2:
            st.subheader("Performance par Département")
            perf_by_dept = dataset.groupby('department')['performance_score'].mean().reset_index().sort_values('performance_score', ascending=False)
            fig_perf = px.bar(perf_by_dept, x='department', y='performance_score', 
                              color='performance_score', color_continuous_scale='Viridis',
                              labels={'performance_score': 'Score Moyen', 'department': 'Département'})
            fig_perf.update_layout(xaxis_title=None, showlegend=False)
            st.plotly_chart(fig_perf, use_container_width=True)

        st.markdown("---")
        
        # Section Détails / Source
        st.subheader("📋 Détails des Données Sources")
        
        mean_score = dataset['performance_score'].mean()
        col_toggle, col_stat = st.columns([1, 3])
        
        with col_toggle:
            show_below_avg = st.checkbox(f"📉 Afficher uniquement sous la moyenne (< {mean_score:.2f})")
            
        if show_below_avg:
            display_df = dataset[dataset['performance_score'] < mean_score]
            with col_stat:
                st.caption(f"Affichage de {len(display_df)} employés (sur {len(dataset)})")
        else:
            display_df = dataset
            with col_stat:
                st.caption(f"Affichage de l'ensemble des {len(dataset)} employés")
                
        st.dataframe(display_df, use_container_width=True)

# ==========================================
# TAB 2: ANALYSE APPROFONDIE & ÉVALUATION
# ==========================================
with tab_analysis:
    if dataset is None:
        st.error("Données non disponibles.")
    else:
        st.markdown("#### 📊 Performance du Modèle (Projet 2)")
        
        if model_metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("F1-Score (Global)", f"{model_metrics.get('f1', 0):.2f}")
            with col2:
                st.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.2f}")
            with col3:
                st.metric("Modèle", model_metrics.get('model_name', 'N/A'))
                
            st.markdown("#### Matrice de Confusion")
            cm = np.array(model_metrics.get('confusion_matrix', [[0,0],[0,0]]))
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                              labels={'x': "Prédiction", 'y': "Réalité", 'color': "Nombre"},
                              x=['Non-Performant', 'Performant'],
                              y=['Non-Performant', 'Performant'])
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Les métriques du modèle ne sont pas disponibles (metrics.json introuvable).")

        st.markdown("---")
        st.markdown("#### 🔍 Exploration des Variables (Projet 2)")
        
        # Filtres
        with st.expander("🛠️ Filtres Avancés", expanded=False):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                selected_depts = st.multiselect("Filtrer par Spécialité (Département)", dataset['department'].unique(), default=dataset['department'].unique())
            with col_f2:
                selected_roles = st.multiselect("Filtrer par Poste", dataset['job_role'].unique(), default=dataset['job_role'].unique())
        
        filtered_df = dataset[dataset['department'].isin(selected_depts) & dataset['job_role'].isin(selected_roles)]
        
        # Correlation Heatmap
        st.subheader("Matrice de Corrélation")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter Plot Multivarié
        st.markdown("---")
        st.subheader("Analyse Multivariée")
        c5, c6, c7 = st.columns(3)
        x_axis = c5.selectbox("Axe X", numeric_cols, index=0)
        y_axis = c6.selectbox("Axe Y", numeric_cols, index=1)
        color_axis = c7.selectbox("Couleur (Catégorie)", ['department', 'job_role', 'education_level', 'salary_band'])
        
        fig_scatter = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_axis, size='performance_score', hover_data=['employee_id'], title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 3: SIMULATEUR DE PERFORMANCE (PROJET 2)
# ==========================================
with tab_prediction:
    st.markdown("#### 🤖 Évaluation Candidat (Projet 2)")
    
    if model is None:
        st.warning("⚠️ Modèle de prédiction non chargé (fichiers .pkl manquants).")
    else:
        with st.form("prediction_form"):
            st.markdown("##### 1. Profil & Expérience")
            c_p1, c_p2, c_p3 = st.columns(3)
            
            with c_p1:
                # Mapping: Spécialité -> Department
                department = st.selectbox("Spécialité (Département)", label_encoders["department"].classes_.tolist())
                # Mapping: Postes occupés -> Job Role
                job_role = st.selectbox("Poste Visé", label_encoders["job_role"].classes_.tolist())
                # Mapping: Niveau d'études -> Education Level
                education_level = st.selectbox("Niveau d'études", label_encoders["education_level"].classes_.tolist())
            
            with c_p2:
                # Mapping: Mobilité -> Work Location (Proxy)
                work_location = st.selectbox("Mobilité / Lieu", label_encoders["work_location"].classes_.tolist())
                # Mapping: Années d'expérience -> Years at company (Proxy)
                years_at_company = st.slider("Années d'expérience", 0, 40, 5)
                salary_band = st.selectbox("Tranche salariale attendue", label_encoders["salary_band"].classes_.tolist())
            
            with c_p3:
                # Mapping: Secteur précédent -> (Pas de colonne, on ignore ou on utilise promotions comme proxy d'évolution)
                promotions = st.number_input("Promotions passées", 0, 10, 0)
                # Mapping: Disponibilité -> Attendance Rate (Proxy)
                attendance = st.slider("Disponibilité / Assiduité (%)", 50.0, 100.0, 95.0)
                
            
            st.markdown("##### 2. Compétences & Tests")
            c_i1, c_i2, c_i3 = st.columns(3)
            with c_i1:
                # Mapping: Score Test Technique -> Tasks Completed / Training Hours
                tasks = st.number_input("Score Test Technique (Simulé via Tâches)", 0, 500, 50)
                training = st.slider("Heures de Formation", 0, 200, 20)
            with c_i2:
                # Mapping: Score Softskills -> Peer Review
                peer_score = st.slider("Score Softskills (Pairs)", 0.0, 10.0, 7.5)
            with c_i3:
                # Mapping: Avis Manager -> Manager Rating
                manager_score = st.slider("Avis Manager Précédent", 0.0, 10.0, 7.5)
                
            submit_btn = st.form_submit_button("🚀 Évaluer le Candidat")
            
        if submit_btn:
            # Préparation des données
            input_data = pd.DataFrame([{
                "department": department,
                "job_role": job_role,
                "education_level": education_level,
                "salary_band": salary_band,
                "work_location": work_location,
                "promotions_last_3years": promotions,
                "years_at_company": years_at_company,
                "monthly_hours_worked": 160.0, # Valeur par défaut
                "attendance_rate": attendance,
                "training_hours": training,
                "tasks_completed": tasks,
                "peer_review_score": peer_score,
                "manager_rating": manager_score,
            }])
            
            # Encodage
            for col, encoder in label_encoders.items():
                if col in input_data.columns:
                    input_data[col] = encoder.transform(input_data[col])
            
            # Scaling
            input_scaled = input_data[feature_columns] # Ensure correct order
            if scaler:
                input_scaled = scaler.transform(input_scaled)
            
            # Prédiction
            try:
                # Prédiction (0 ou 1 pour classification binaire)
                prediction_class = model.predict(input_scaled)[0]
                
                # Probabilités
                score_confiance = 0.5
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)[0]
                    # Probabilité de la classe positive (1 = Performant)
                    if len(proba) == 2:
                        score_confiance = proba[1]
                    else:
                        score_confiance = proba[1] if len(proba) > 1 else proba[0]
                
                st.markdown("---")
                st.markdown("### 🎯 Résultat de l'Évaluation")
                
                # Affichage du score de confiance (Probabilité de succès)
                st.metric("Probabilité de Succès (Candidat Performant)", f"{score_confiance:.1%}")
                
                # Jauge de probabilité
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score_confiance * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilité (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71" if score_confiance >= 0.5 else "#e74c3c"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "white"}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50}}))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Message contextuel
                if prediction_class == 1:
                    st.success("✅ **Candidat Potentiellement Performant** : Ce profil présente des caractéristiques associées aux employés performants.")
                else:
                    st.error("⚠️ **Risque de Performance** : Ce profil présente des caractéristiques moins associées à une haute performance.")
                
                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    st.markdown("##### Facteurs d'influence clés")
                    try:
                        feat_imp = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True).tail(5)
                        
                        fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Top 5 Facteurs d'Influence")
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")
                st.info("Vérifiez que les données d'entrée sont correctes.")
