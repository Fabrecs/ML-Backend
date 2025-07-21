from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Text Vectorization using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load model and tokenizer once at module level for efficiency
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Move model to GPU if available
model = model.to(device)
print(f"✅ Text Vectorization model loaded and moved to {device}")

# Enable evaluation mode for inference
model.eval()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_text_vector(text):
    """
    Convert input text into a vector representation using sentence transformers.
    
    Args:
        text (str): Input text to be vectorized
        
    Returns:
        numpy.ndarray: Vector representation of the input text
    """
    # Tokenize text
    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
    
    # Move inputs to the same device as the model
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    # Convert to numpy array and return (move back to CPU for numpy conversion)
    return sentence_embeddings[0].cpu().numpy()

# # Example usage
# if __name__ == "__main__":
#     test_text = "This is an example sentence"
#     vector = get_text_vector(test_text)
#     print("Vector representation:", vector)
