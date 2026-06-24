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

    # Inputs ekdum sahi line par hone chahiye
    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, step=10.0)

    # Dhyan se dekho: 'if' line bilkul 'recency' aur 'frequency' ki line ke sath vertically aligned honi chahiye
    if st.button("Predict Segment"):
        # Is block ke andar 4 spaces (ya 1 tab) ka gap hona chahiye
        user_data = np.array([[recency, frequency, monetary]])
        user_scaled = scaler.transform(user_data)
        cluster_pred = kmeans.predict(user_scaled)[0]
        
        cluster_labels = {
            0: "0 High Value Customer (High F, High M, Low R)",
            1: "1 At-Risk Customer (High R, Low F, Low M)",
            2: "2 Regular Customer (Medium F, Medium M)",
            3: "3  At-Risk Customer (High R, Low F, Low M)"
        }
        
        final_label = cluster_labels.get(cluster_pred, f"Cluster {cluster_pred}")
        
        st.subheader("Prediction Result:")
        st.success(f"This customer belongs to: **{final_label}**")
        
        final_label = cluster_labels.get(cluster_pred, f"Cluster {cluster_pred}")
        
        # Display Output [cite: 66]
        st.subheader("Prediction Result:")
        st.success(f"This customer belongs to: **{final_label}**")

# ================= 🎁 RECOMMENDATION MODULE =================
elif page == "🎁 Recommendation":
    st.title("🎯 Product Recommendation Module")
    st.write("Enter a product name to get 5 similar recommendations based on Collaborative Filtering:")

    # Inputs aur Button ekdum vertical line mein hone chahiye
    product_input = st.text_input("Enter Product Name", value="GREEN VINTAGE SPOT BEAKER")

    # Dhyan se dekho: 'if' line bilkul 'product_input' ke sath vertically aligned hai
    if st.button("Get Recommendations"):
        prod_name = product_input.strip()
        
        # Is block ke andar 4 spaces (ya 1 tab) ka gap hai
        if prod_name in similarity_df:
            recommendations = similarity_df[prod_name][:5]
            
            st.write("### Recommended Products:")
            for i, item in enumerate(recommendations, 1):
                st.info(f"**{i}. {item}**")
        else:
            st.error(f"Bhai, '{prod_name}' naam ka koi product data mein nahi mila. Ek baar exact naam check karo!")
