from sentence_transformers import SentenceTransformer
import pinecone
import pandas as pd
import streamlit as st
from PIL import Image
import os

# loading env data
from dotenv import load_dotenv
load_dotenv()

# connecting to Pinecone
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Mapping display values to actual index names
index_display_map = {
    # "Asos - Latest Clothes & Fashion": "image-search-asos-metadata",
    "Aldo - Shoes and Accessories": "image-search-aldo-metadata",
    "Hartlauer - Tech products": "image-search-hartlauer-crawled"
}

# Default index name (if exists)
default_index_name = "image-search-aldo-metadata"

# UI elements
st.image('https://upload.wikimedia.org/wikipedia/commons/9/99/Nortal_logo_RGB.svg', width=200)
st.write("""
# Semantic Image Search
""")
st.write("""
Semantic image search uses a *text query* or an *input image* to search a database of images to find images that are semantically similar to the search query.
""")

st.write("""
## Demo
""")

# Create a dropdown with display names
selected_display_name = st.selectbox(
    "Choose an index to query:",
    list(index_display_map.keys()),  # Display values
    index=list(index_display_map.keys()).index("Aldo - Shoes and Accessories")  # Default display name
)

# Get the internal index name for the selected display name
selected_index_name = index_display_map[selected_display_name]

# Initialize the selected Pinecone index
index = pc.Index(selected_index_name)

# st.write(f"Selected index name: {selected_index_name}")


# Load the CLIP model
model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")

# Text query input
query = st.text_input("Enter your search query:")

# Text search
if st.button("Search with text"):
    if query.strip():
        # Use the SentenceTransformer model to embed the query text
        query_embedding = model.encode(query)
        # response_text is a json file that contains the results from the pinecone dabase call
        response_text = index.query(vector=query_embedding.tolist(), top_k=8, include_values=False, include_metadata=True)

        # display the results in a grid
        if response_text.get('matches'):
            matches = response_text['matches']
            
            # Adjust the number of columns you want in each row (currently 4)
            columns_per_row = 4

            # Iterate through the matches
            for i in range(0, len(matches), columns_per_row):
                # Create columns (4 in this case) for each row
                cols = st.columns(columns_per_row)
                
                # Iterate over the matches for this row
                for j in range(columns_per_row):
                    if i + j < len(matches):  # Ensure index is within bounds
                        match = matches[i + j]
                        
                        # Extract metadata safely
                        metadata = match.get('metadata', {})
                        image_url = metadata.get('image_url', None)
                        product_url = metadata.get('product_url', None)
                        product_id = match.get('id', 'Unknown ID')

                        # Use the column object to display the image and link
                        with cols[j]:  # Target the specific column in the row
                            if image_url:
                                # Display the image and product link
                                st.markdown(f'<a href="{image_url}" target="_blank"><img src="{image_url}" width="150"/></a>', unsafe_allow_html=True)
                                st.markdown(f'<a href="{product_url}" target="_blank">Product ID: {product_id}</a>', unsafe_allow_html=True)
                            else:
                                st.write(f"Image URL not found for Product ID: {product_id}.")

        else:
            st.write("No matching images found.")
    else:
        st.write("Please enter a valid search query.")

# Image upload input
uploaded_file = st.file_uploader("Or upload an image to search", type=["jpg", "png", "jpeg"])

# Image search
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Get image embeddings using CLIP model
    image_embeddings = model.encode(image).tolist()
    
    # Query Pinecone with the image embeddings
    response_image = index.query(vector=image_embeddings, top_k=8, include_values=False, include_metadata=True)
    
    # display the results in a grid
    if response_image.get('matches'):
        matches = response_image['matches']
        
        # Adjust the number of columns you want in each row (currently 4)
        columns_per_row = 4

        # Iterate through the matches
        for i in range(0, len(matches), columns_per_row):
            # Create columns (4 in this case) for each row
            cols = st.columns(columns_per_row)
            
            # Iterate over the matches for this row
            for j in range(columns_per_row):
                if i + j < len(matches):  # Ensure index is within bounds
                    match = matches[i + j]
                    
                    # Extract metadata safely
                    metadata = match.get('metadata', {})
                    image_url = metadata.get('image_url', None)
                    product_url = metadata.get('product_url', None)
                    product_id = match.get('id', 'Unknown ID')

                    # Use the column object to display the image and link
                    with cols[j]:  # Target the specific column in the row
                        if image_url:
                            # Display the image and product link
                            st.markdown(f'<a href="{image_url}" target="_blank"><img src="{image_url}" width="150"/></a>', unsafe_allow_html=True)
                            st.markdown(f'<a href="{product_url}" target="_blank">Product ID: {product_id}</a>', unsafe_allow_html=True)
                        else:
                            st.write(f"Image URL not found for Product ID: {product_id}.")
    else:
        st.write("No matching images found.")
