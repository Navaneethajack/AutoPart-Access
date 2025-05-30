import streamlit as st
import ollama
import json
import os
import hashlib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import base64
import random
import urllib.parse

# ⬇️ Your loader function
def show_loader(placeholder):
    placeholder.markdown("""
    <style>
        .loader-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            position: relative;
        }
        .ring {
            border: 10px solid ;
            border-top: 10px solid #4CAF50;
            border-radius: 50%;
            position: absolute;
            animation: spin 2s linear infinite;
        }
        .ring1 {
            width: 120px;
            height: 120px;
        }
        .ring2 {
            width: 80px;
            height: 80px;
            border-top: 10px solid #2196F3;
            animation-duration: 1.5s;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <div class="loader-wrapper">
        <div class="ring ring1"></div>
        <div class="ring ring2"></div>
    </div>
    """, unsafe_allow_html=True)
# 🔍 Query LLaMA 3
def parse_query_llama3(query):
    prompt = f"""
    Extract the automobile part type, automobile part model, vehicle model, and price range from the following query:

    Query: "{query}"

    Respond in JSON format with keys: part_type, vehicle_model, price_range (as a list of two numbers).
    """
    try:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response['message']['content']
        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end+1]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response.")
    except Exception as e:
        print("LLaMA 3 Parsing Error:", e)
        return {"part_type": "", "vehicle_model": "", "price_range": [0, 999999]}

# 🔌 Get Site-specific Search URL
def get_search_url(site, query):
    encoded_query = urllib.parse.quote_plus(query)
    site = site.lower()
    domain_map = {
        "flipkart": f"https://www.flipkart.com/search?q={encoded_query}",
        "amazon": f"https://www.amazon.in/s?k={encoded_query}",
        "snapdeal": f"https://www.snapdeal.com/search?keyword={encoded_query}",
        "ebay": f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}",
        "indiamart": f"https://dir.indiamart.com/search.mp?ss={encoded_query}",
        "boodmo": f"https://boodmo.com/catalog/search/?q={encoded_query}",
        "pricerunner": f"https://www.pricerunner.com/search?q={encoded_query}",
        "gomechanic": f"https://gomechanic.in/spares?q={encoded_query}",
        "cardekho": f"https://www.cardekho.com/cars?q={encoded_query}",
        "autodoc": f"https://www.autodoc.co.uk/search?keyword={encoded_query}",
        "motointegrator": f"https://www.motointegrator.com/search?keyword={encoded_query}",
        "partslink24": f"https://www.partslink24.com/search?q={encoded_query}",
        "tecalliance": f"https://www.tecalliance.com/en/solutions/tecdoc-catalog?q={encoded_query}",
        "camelcamelcamel": f"https://camelcamelcamel.com/search?sq={encoded_query}"
    }

    general_query = f"{site.rstrip('/')}/search?q={encoded_query}"
    return domain_map.get(site, general_query)

# 📦 Scraper with Caching (Simulated)
def scrape_site(query, site):
    filename = f"cache/{hashlib.md5((query + site).encode()).hexdigest()}.json"
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)

    result = [{
        "name": f"{query} - Sample from {site}",
        "price": random.randint(1200, 2000),
        "rating": round(random.uniform(3.8, 4.5), 2),
        "link": get_search_url(site, query)
    }]

    with open(filename, "w") as f:
        json.dump(result, f)

    return result

# 🏆 Optimal Product Selector
def choose_optimal(results):
    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()

    df["norm_price"] = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min() + 1e-6)
    df["norm_rating"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min() + 1e-6)
    df["score"] = (1 - df["norm_price"]) * 0.6 + df["norm_rating"] * 0.4

    return df.sort_values(by="score", ascending=False).head(1)

# 🌐 Supported Sites (by identifier)
supported_sites = [
    "amazon", "ebay", "flipkart", "snapdeal", "indiamart", "boodmo", "pricerunner", "gomechanic",
    "cardekho", "autodoc", "motointegrator", "supercheapauto", "repco", "partslink24", "tecalliance", "camelcamelcamel"
]

# 🖼️ Streamlit UI
st.title("🔧 Auto Part Finder Chatbot")
user_query = st.text_input("Enter your automobile part request:")

if user_query:
    loader_placeholder = st.empty()
    show_loader(loader_placeholder)

    parsed = parse_query_llama3(user_query)
    query = f"{parsed['part_type']} for {parsed['vehicle_model']}"

    all_results = []
    for site in supported_sites:
        all_results.extend(scrape_site(query, site))

    loader_placeholder.empty()

    results_df = pd.DataFrame(all_results)
    optimal_df = choose_optimal(all_results)

    st.write("🔍 **Search Query:**", query)
    st.write("📦 **All Search Results:**")
    st.dataframe(results_df)

    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_data,
        file_name="auto_parts_results.csv",
        mime="text/csv"
    )

    st.write("✅ **Optimal Recommendation:**")
    if not optimal_df.empty:
        st.dataframe(optimal_df[["name", "price", "rating", "link"]])
    else:
        st.warning("No suitable products found.")
