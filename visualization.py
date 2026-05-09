import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np
import streamlit as st

def visualize_vectors_tsne(qdrant_client, collection_name="my_documents"):
    try:
        # Fetch all points from Qdrant (limit to 500 for speed)
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=500,
            with_vectors=True,
            with_payload=True
        )
        
        if not points:
            return None
            
        vectors = [p.vector for p in points]
        texts = [p.payload.get("page_content", "Unknown")[:100] + "..." for p in points]
        
        vectors = np.array(vectors)
        
        # t-SNE reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
        reduced = tsne.fit_transform(vectors)
        
        fig = px.scatter(
            x=reduced[:, 0], y=reduced[:, 1],
            hover_name=texts,
            title="2D Vector Map (t-SNE)",
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            template="plotly_dark"
        )
        return fig
    except Exception as e:
        st.error(f"Visualization Error: {e}")
        return None
