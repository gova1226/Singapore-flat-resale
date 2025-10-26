import streamlit as st #building interactive apps in Python
from streamlit_option_menu import option_menu #sidebar menu with icons and better styling
import numpy as np #numeric computations like log and arrays
import datetime  #handles date/time functions
import pickle #load the pre-trained Decision Tree model for prediction

# Page config
st.set_page_config(page_title='HDB Flat Resale Price Predictor', page_icon='🏠', layout='wide')

# ---- SIMPLIFIED, BRIGHTER GLASSMORPHISM STYLE ----
st.markdown("""
<style>
:root {
  --glass-bg: rgba(255,255,255,0.65);
  --border: rgba(200,200,200,0.3);
  --accent1: #2bb78f;
  --accent2: #3a7bd5;
  --text-dark: #1a1a1a;
  --muted: #555;
}

[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #e8f7f4 0%, #f3f8fd 100%);
  color: var(--text-dark);
}

#MainMenu, footer {visibility: hidden;}

.header, .glass-card {
  border-radius: 14px;
  backdrop-filter: blur(8px);
  background: var(--glass-bg);
  border: 1px solid var(--border);
  box-shadow: 0 4px 10px rgba(0,0,0,0.05);
  padding: 16px 22px;
}

.title-large {font-size:28px; font-weight:700; color:var(--text-dark);}
.subtitle {color:var(--muted);}
.small-muted {font-size:13px; color:#666;}

button, .stButton > button {
  background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
}
</style>
""", unsafe_allow_html=True)

# Header
with st.container():
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown("""
        <div class='header'>
            <div style='display:flex; align-items:center; gap:14px'>
                <div style='width:60px; height:60px; border-radius:12px; background:linear-gradient(135deg,#2bb78f,#3a7bd5);
                            display:flex; align-items:center; justify-content:center; font-size:28px;'>🏠</div>
                <div>
                    <div class='title-large'>HDB Flat Resale Price Predictor</div>
                    <div class='subtitle'>Predict resale prices of Singapore HDB flats</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        try:
            st.image("singapore.jpg", width=150)
        except:
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/58/Bishan_HDB.JPG", width=110)

# Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Get Prediction", "About"],
        icons=['house', 'bar-chart', 'info-circle'],
        default_index=0,
        styles={
            "container": {"background-color": "rgba(255,255,255,0.4)", "padding": "6px"},
            "icon": {"color": "#2bb78f", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "color": "#1a1a1a", "margin": "4px 0"},
            "nav-link-selected": {"background-color": "rgba(43,183,143,0.15)", "border-left": "4px solid #2bb78f"},
        }
    )

# Helper class
class option:
    option_months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    encoded_month = {m:i+1 for i,m in enumerate(option_months)}
    option_town = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA',
                   'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                   'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG','SERANGOON','TAMPINES',
                   'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG',
                   'PASIR RIS','PUNGGOL']
    encoded_town = {t:i for i,t in enumerate(option_town)}
    option_flat_type = ['1 ROOM', '2 ROOM','3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION']
    encoded_flat_type = {v:i for i,v in enumerate(option_flat_type)}
    option_flat_model = ['2-ROOM','3GEN','ADJOINED FLAT', 'APARTMENT' ,'DBSS','IMPROVED' ,'IMPROVED-MAISONETTE',
                         'MAISONETTE','MODEL A', 'MODEL A-MAISONETTE','MODEL A2' ,'MULTI GENERATION' ,'NEW GENERATION',
                         'PREMIUM APARTMENT','PREMIUM APARTMENT LOFT', 'PREMIUM MAISONETTE','SIMPLIFIED',
                         'STANDARD','TERRACE','TYPE S1','TYPE S2']
    encoded_flat_model = {v:i for i,v in enumerate(option_flat_model)}

# Prediction Page
if selected == 'Get Prediction':
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Provide details to predict resale price")
    with st.form('prediction_form'):
        left, right = st.columns(2)
        with left:
            user_month = st.selectbox('Month', option.option_months)
            user_town = st.selectbox('Town', option.option_town)
            user_flat_type = st.selectbox('Flat Type', option.option_flat_type)
            user_flat_model = st.selectbox('Flat Model', option.option_flat_model)
            floor_area_sqm = st.number_input('Floor area (sqm)', min_value=10.0, value=45.0)
            price_per_sqm = st.number_input('Price per sqm (SGD)', min_value=100.0, value=1200.0)
        with right:
            year = st.text_input('Year of sale (YYYY)', max_chars=4, value=str(datetime.datetime.now().year))
            block = st.text_input('Block', max_chars=8)
            lease_commence_date = st.text_input('Lease commence year (YYYY)', max_chars=4, value='1990')
            remaining_lease = st.number_input('Remaining lease (years)', min_value=0, max_value=999, value=60)
            years_holding = st.number_input('Years holding', min_value=0, max_value=200, value=10)
            storey_start = st.number_input('Storey start', min_value=1, max_value=100, value=1)
            storey_end = st.number_input('Storey end', min_value=1, max_value=100, value=5)
        submit = st.form_submit_button('PREDICT', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        try:
            current_year = datetime.datetime.now().year
            current_remaining_lease = remaining_lease - (current_year - int(year))
            age_of_property = current_year - int(lease_commence_date)
            month = option.encoded_month[user_month]
            town = option.encoded_town[user_town]
            flat_type = option.encoded_flat_type[user_flat_type]
            flat_model = option.encoded_flat_model[user_flat_model]
            floor_area_sqm_log = np.log(floor_area_sqm)
            remaining_lease_log = np.log1p(remaining_lease)
            price_per_sqm_log = np.log(price_per_sqm)
            with open('Decisiontree.pkl', 'rb') as f:
                model = pickle.load(f)
            user_data = np.array([[month, town, flat_type, block, flat_model, lease_commence_date, year,
                                   storey_start, storey_end, years_holding, current_remaining_lease,
                                   age_of_property, floor_area_sqm_log, remaining_lease_log, price_per_sqm_log]],
                                 dtype=object)
            resale_price = np.exp(model.predict(user_data)[0])
            st.markdown(f"""
                <div class='glass-card'>
                    <h3>Prediction Result</h3>
                    <p class='small-muted'>Estimated resale price:</p>
                    <h2 style='color:#2bb78f;'>SGD {resale_price:,.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# Home
if selected == 'Home':
    # --- Page Background & CSS Animations ---
    st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
        background: linear-gradient(120deg, #E0F2FE, #FDE68A);
        overflow-x: hidden;
        font-family: Arial, sans-serif;
    }
    /* Glass card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
        position: relative;
        z-index: 2;
    }
    .glass-card:hover {
        transform: translateY(-5px);
    }
    /* Highlight text animation */
    .highlight {
        background: linear-gradient(90deg, #FACC15, #22D3EE, #F472B6);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        font-weight: bold;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    /* Pulsating emojis */
    .pulse {
        display: inline-block;
        animation: pulseAnim 1.2s infinite;
    }
    @keyframes pulseAnim {
        0% {transform: scale(1);}
        50% {transform: scale(1.3);}
        100% {transform: scale(1);}
    }
    /* Floating shapes */
    .float-shape {
        position: fixed;
        width: 20px;
        height: 20px;
        background: rgba(255,255,255,0.3);
        border-radius: 50%;
        animation: floatAnim linear infinite;
        z-index: 1;
    }
    @keyframes floatAnim {
        0% {transform: translateY(0) translateX(0);}
        50% {transform: translateY(-200px) translateX(50px);}
        100% {transform: translateY(0) translateX(0);}
    }
    </style>
    
    <!-- Floating shapes HTML -->
    <div class="float-shape" style="top:10%; left:5%; width:15px; height:15px; animation-duration:12s;"></div>
    <div class="float-shape" style="top:50%; left:80%; width:20px; height:20px; animation-duration:15s;"></div>
    <div class="float-shape" style="top:70%; left:30%; width:10px; height:10px; animation-duration:10s;"></div>
    <div class="float-shape" style="top:20%; left:60%; width:25px; height:25px; animation-duration:18s;"></div>
    """, unsafe_allow_html=True)

    # --- Glass Card Wrapper ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    # --- Header ---
    st.markdown("<h2 class='highlight'>🏠 About Modern HDB Living</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#333; font-size:16px; line-height:1.5;'>
    Singapore’s Housing & Development Board (HDB) is redefining public housing through smart, sustainable, and community-centric design.<br>
    Modern HDB estates combine digital innovation, green architecture, and inclusive planning — making every neighbourhood a connected, eco-friendly, and vibrant place to live.<br>
    The goal is to create homes that are future-ready and enhance the quality of urban life for every Singaporean.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:10px 0; border:1px solid #ccc;'>", unsafe_allow_html=True)

    # --- Key Features Section (Columns with Animated Emojis & Highlights) ---
    st.markdown("<h3 class='highlight'>Key Features of Modern HDB Living</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <span class='pulse'>🏙️</span> <span class='highlight'>Smart Towns</span><br>
        - <span class='pulse'>💡</span> Technology-enabled apartments<br>
        - <span class='pulse'>🔒</span> Integrated security systems<br>
        - <span class='pulse'>📱</span> Smart community services
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <span class='pulse'>🌿</span> <span class='highlight'>Eco-Friendly Design</span><br>
        - <span class='pulse'>🌳</span> Green roofs & parks<br>
        - <span class='pulse'>☀️</span> Solar panels & energy efficiency<br>
        - <span class='pulse'>🌎</span> Sustainable town planning
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <span class='pulse'>🏘️</span> <span class='highlight'>Community Spaces</span><br>
        - <span class='pulse'>🛝</span> Playgrounds & social hubs<br>
        - <span class='pulse'>🤝</span> Gathering spaces for residents<br>
        - <span class='pulse'>🎉</span> Inclusive neighborhood activities
        """, unsafe_allow_html=True)

    # --- Vision Section (Collapsible with Animated Highlights) ---
    with st.expander("🎯 Vision · Mission · Innovation"):
        st.markdown("""
        <p class='highlight'>Vision: To shape a smart and sustainable city through innovative public housing.</p>
        <p style='color:#047857;'><b>Mission:</b> Build connected, community-driven homes using technology and eco-friendly design.</p>
        <p style='color:#B45309;'><b>Innovation:</b> Solar energy, smart lighting, and green architecture are part of every new HDB town.</p>
        """, unsafe_allow_html=True)

    # --- Explore Section (Collapsible Highlights with Animated Key Points) ---
    with st.expander("🔍 Explore HDB Innovation"):
        st.markdown("""
        <ul style='color:#374151; font-size:15px; line-height:1.5;'>
            <li><span class='pulse'>💡</span> <span class='highlight'>Smart lighting and energy monitoring systems</span> in apartments</li>
            <li><span class='pulse'>🏘️</span> <span class='highlight'>Community-driven design workshops</span> for residents’ input</li>
            <li><span class='pulse'>🌳</span> <span class='highlight'>Eco-friendly town planning</span> with green corridors and parks</li>
            <li><span class='pulse'>📲</span> <span class='highlight'>Digital services</span> for residents, including e-payments and smart home apps</li>
        </ul>
        """, unsafe_allow_html=True)

    # --- Official Link ---
    st.markdown("""
    <div style='margin-top:20px; text-align:center;'>
        <a href='https://www.hdb.gov.sg/cs/infoweb/homepage' target='_blank'>
            <button style='background-color:#1E3A8A; color:white; padding:10px 20px; border-radius:8px; border:none; font-size:16px;'>Learn More at HDB Official Website</button>
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



# About
if selected == 'About':
    # --- CSS for glass-card, highlights, pulsating emojis, floating shapes ---
    st.markdown("""
    <style>
    body {margin:0; padding:0; background: linear-gradient(120deg, #E0F2FE, #FDE68A); overflow-x:hidden; font-family:Arial,sans-serif;}
    .glass-card {background:rgba(255,255,255,0.6); backdrop-filter:blur(8px); padding:20px; border-radius:15px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.1); transition:transform 0.2s; position:relative; z-index:2;}
    .glass-card:hover {transform:translateY(-5px);}
    .highlight {background:linear-gradient(90deg,#FACC15,#22D3EE,#F472B6); background-size:200% 200%; -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:gradientShift 3s ease infinite; font-weight:bold;}
    @keyframes gradientShift {0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;}}
    .pulse {display:inline-block; animation:pulseAnim 1.2s infinite;}
    @keyframes pulseAnim {0%{transform:scale(1);}50%{transform:scale(1.3);}100%{transform:scale(1);}}
    .float-shape {position:fixed; background:rgba(255,255,255,0.3); border-radius:50%; animation:floatAnim linear infinite; z-index:1;}
    @keyframes floatAnim {0%{transform:translateY(0) translateX(0);}50%{transform:translateY(-200px) translateX(50px);}100%{transform:translateY(0) translateX(0);}}
    </style>

    <!-- Floating shapes -->
    <div class="float-shape" style="top:15%; left:10%; width:15px; height:15px; animation-duration:12s;"></div>
    <div class="float-shape" style="top:60%; left:70%; width:20px; height:20px; animation-duration:15s;"></div>
    <div class="float-shape" style="top:75%; left:40%; width:10px; height:10px; animation-duration:10s;"></div>
    <div class="float-shape" style="top:25%; left:55%; width:25px; height:25px; animation-duration:18s;"></div>
    """, unsafe_allow_html=True)

    # --- Glass Card ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.markdown("<h2 class='highlight'>📄 Project & Author</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#333; font-size:16px; line-height:1.5;'>
    <span class='pulse'>📝</span> <b>Project:</b> Singapore Resale Flat Prices Prediction<br>
    <span class='pulse'>🏠</span> <b>Domain:</b> Real Estate<br>
    <span class='pulse'>💻</span> <b>Tech Stack:</b> Python, Streamlit, Decision Tree Regressor, Data Preprocessing<br>
    <span class='pulse'>👤</span> <b>Author:</b> Govardhanan G
    </p>
    """, unsafe_allow_html=True)

    # --- Buttons ---
    st.markdown("""
    <div style='margin-top:10px'>
        <a href='https://www.hdb.gov.sg/cs/infoweb/homepage' target='_blank'>
            <button style='padding:8px 15px; border-radius:8px; border:none; background-color:#1E3A8A; color:white; font-weight:bold;'>Official HDB Website</button></a>
        &nbsp;
        <a href='https://www.linkedin.com/in/govardhanan-g-1b2770102/' target='_blank'>
            <button style='padding:8px 15px; border-radius:8px; border:none; background-color:#1E3A8A; color:white; font-weight:bold;'>Author LinkedIn</button></a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<div style='text-align:center; color:#555; margin-top:20px;'>Built with ❤️ Bright Glassmorphism UI</div>", unsafe_allow_html=True)
