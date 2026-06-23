import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# --- Models Load Karna ---
@st.cache_resource # Isse baar-baar file load nahi hogi aur app fast chalegi
def load_models():
    kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    similarity_df = pickle.load(open('product_similarity.pkl', 'rb'))
    return kmeans, scaler, similarity_df

try:
    kmeans, scaler, similarity_df = load_models()
except FileNotFoundError:
    st.error("Bhai, .pkl files nahi mili! Pehle notebook se models save karke is folder mein rakho.")

# --- Sidebar Navigation (Document ke mutabik) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Clustering", "🎁 Recommendation"])

# ================= 🏠 HOME PAGE =================
if page == "🏠 Home":
    st.title("🛒 Shopper Spectrum: Customer Segmentation & Recommendations")
    st.markdown("""
    ### Welcome Bhai! 
    Yeh aapka E-Commerce and Retail Analytics Dashboard hai.
    * **Clustering Module:** Isme aap kisi bhi customer ka RFM data daal kar uska segment jaan sakte hain.
    * **Recommendation Module:** Isme kisi product ka naam daal kar top 5 milti-julti items dhoondh sakte hain.
    """)

# ================= 📊 CLUSTERING MODULE =================
elif page == "📊 Clustering":
    st.title("🎯 Customer Segmentation Module")
    st.write("Enter the customer's RFM behavior details below:")

    # Document ke hisab se 3 Number Inputs [cite: 61]
    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1) [cite: 62]
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1) [cite: 63]
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, step=10.0) [cite: 64]

    if st.button("Predict Segment"): [cite: 65]
        # 1. Input ko array mein convert karna
        user_data = np.array([[recency, frequency, monetary]])
        
        # 2. StandardScaler se transform karna (Zaruri step!)
        user_scaled = scaler.transform(user_data)
        
        # 3. K-Means se cluster predict karna
        cluster_pred = kmeans.predict(user_scaled)[0]
        
        # 4. Cluster mapping (Aapke document wale labels) [cite: 45]
        # Note: Aapne model mein jis cluster ko jo naam diya hai, us hisab se numbers (0,1,2,3) set kar lena
        cluster_labels = {
            0: "Occasional Shopper (Low F, Low M, older R)",
            1: "High-Value Customer (High R, High F, High M)",
            2: "Regular Customer (Medium F, Medium M)",
            3: "At-Risk Customer (High R, Low F, Low M)"
        }
        
        final_label = cluster_labels.get(cluster_pred, f"Cluster {cluster_pred}")
        
        # Display Output [cite: 66]
        st.subheader("Prediction Result:")
        st.success(f"This customer belongs to: **{final_label}**")

# ================= 🎁 RECOMMENDATION MODULE =================
elif page == "🎁 Recommendation":
    st.title("🎯 Product Recommendation Module") [cite: 53]
    st.write("Enter a product name to get 5 similar recommendations based on Collaborative Filtering:") [cite: 54]

    # Text Input box [cite: 56]
    product_input = st.text_input("Enter Product Name", value="GREEN VINTAGE SPOT BEAKER")

    if st.button("Get Recommendations"): [cite: 57]
        # Text cleaning (spelling match karne ke liye)
        prod_name = product_input.strip()
        
        if prod_name in similarity_df.index:
            # Cosine similarity se top 5 nikalna (iloc[1:6] kyunki 0 par khud wahi product hoga)
            recommendations = similarity_df[prod_name].sort_values(ascending=False).iloc[1:6].index
            
            st.write("### Recommended Products:") [cite: 58]
            # Styled list layout (Card jaisa look dene ke liye)
            for i, item in enumerate(recommendations, 1):
                st.info(f"**{i}. {item}**") [cite: 58]
        else:
            st.error(f"Bhai, '{prod_name}' naam ka koi product data mein nahi mila. Ek baar exact naam check karo!")