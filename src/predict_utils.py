def extract_prediction(response, is_few_shot=True):
    response = response.lower().strip()
    # Prioritize exact match at the beginning of the response
    if response.startswith("formal"):
        return "formal"
    elif response.startswith("informal"):
        return "informal"
    # Fallback to checking within the response
    elif "formal" in response and "informal" not in response:
        return "formal"
    elif "informal" in response:
        return "informal"
    return "informal" # Default