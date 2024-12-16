# Replace this line:
documents_dict = [doc.dict() for doc in documents]  # If using Pydantic v1

# With this:
documents_dict = [doc.model_dump() for doc in documents]  # For Pydantic v2