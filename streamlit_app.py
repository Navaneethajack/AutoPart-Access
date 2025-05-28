import streamlit as st
import ollama
import json
import os
import hashlib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import base64

def show_leaf_loader(placeholder):
    image_paths = [
        os.path.join("assets", "loader1.gif"),
        os.path.join("assets", "loader2.gif"),
        os.path.join("assets", "loader3.gif")
    ]
    
    base64_images = []
    for path in image_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                base64_images.append(base64.b64encode(f.read()).decode())
        else:
            st.warning(f"Missing image file: {path}")
            base64_images.append("")  # Placeholder if missing
    
    placeholder.markdown(f"""
    <style>
        .loader-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            position: relative;
        }}
        .outer, .middle, .inner {{
            position: absolute;
            border-radius: 50%;
        }}
        .outer {{
            width: 300px;
            height: 300px;
            animation: rotateClockwise 4s linear infinite;
        }}
        .middle {{
            width: 200px;
            height: 200px;
            animation: rotateAntiClockwise 3s linear infinite;
        }}
        .inner {{
            width: 80px;
            height: 80px;
            animation: rotateClockwise 4s linear infinite;
        }}
        @keyframes rotateClockwise {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes rotateAntiClockwise {{
            0% {{ transform: rotate(360deg); }}
            100% {{ transform: rotate(0deg); }}
        }}
    </style>
    <div class="loader-wrapper">
        <div class="outer">
            {"<img src='data:image/gif;base64," + base64_images[0] + "' width='100%' height='100%' />" if base64_images[0] else ""}
        </div>
        <div class="middle">
            {"<img src='data:image/gif;base64," + base64_images[1] + "' width='100%' height='100%' />" if base64_images[1] else ""}
        </div>
        <div class="inner">
            {"<img src='data:image/gif;base64," + base64_images[2] + "' width='100%' height='100%' />" if base64_images[2] else ""}
        </div>
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

# 🧠 Decision Tree Builder
def build_decision_tree(data):
    X = data[["price", "rating"]]
    y = data["suitability"]
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

# 🕸️ Caching Site Scraper
def scrape_site(query, site):
    filename = f"cache/{hashlib.md5((query + site).encode()).hexdigest()}.json"
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)

    result = [{
        "name": f"{query} - Sample from {site}",
        "price": 1500 + hash(site) % 500,
        "rating": 4.0 + (hash(site) % 10) * 0.01,
        "link": site
    }]

    with open(filename, "w") as f:
        json.dump(result, f)

    return result

# 🏆 Select Best Product
def choose_optimal(results):
    df = pd.DataFrame(results)
    df["suitability"] = [1 if price <= df["price"].min() and rating >= df["rating"].max()
                         else 0 for price, rating in zip(df["price"], df["rating"])]
    clf = build_decision_tree(df)
    df["pred"] = clf.predict(df[["price", "rating"]])
    df_sorted = df[df["pred"] == 1].sort_values(by=["price", "rating"], ascending=[True, False])
    return df_sorted.head(1)

# 🌐 Supported Sites
supported_sites = [
    "https://www.amazon.in/", "https://www.ebay.com/", "https://www.rockauto.com/",
    "https://www.autozone.com/", "https://shop.advanceautoparts.com/", "https://www.napaonline.com/",
    "https://www.summitracing.com/", "https://www.eurocarparts.com/", "https://www.halfords.com/",
    "https://www.autodoc.co.uk/", "https://www.motointegrator.com/", "https://boodmo.com/",
    "https://gomechanic.in/", "https://www.cardekho.com/", "https://www.supercheapauto.com.au/",
    "https://www.repco.com.au/", "https://www.partslink24.com/",
    "https://www.tecalliance.com/en/solutions/tecdoc-catalog", "https://www.pricerunner.com/",
    "https://camelcamelcamel.com/"
]

# 🖼️ Streamlit UI
st.title("🔧 Auto Part Finder Chatbot")
user_query = st.text_input("Enter your automobile part request:")

if user_query:
    loader_placeholder = st.empty()
    show_leaf_loader(loader_placeholder)

    parsed = parse_query_llama3(user_query)
    query = f"{parsed['part_type']} for {parsed['vehicle_model']}"

    all_results = []
    for site in supported_sites:
        all_results.extend(scrape_site(query, site))

    optimal_df = choose_optimal(all_results)
    results_df = pd.DataFrame(all_results)

    loader_placeholder.empty()

    st.write("🔍 **Search Query**:", query)

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
        st.dataframe(optimal_df)
    else:
        st.warning("No suitable products found.")
