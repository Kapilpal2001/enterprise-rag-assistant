try:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    print("✅ Success! Libraries are installed and working.")
except ImportError as e:
    print(f"❌ Still missing: {e}")
    