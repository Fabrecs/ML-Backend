from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import boto3
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
import requests
import torch

# Load environment variables for S3 access
load_dotenv()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load HuggingFace token for private model access
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("❌ Warning: HUGGINGFACE_TOKEN not found in environment variables.")
    print("This may cause issues if the model repository is private.")
else:
    print("✅ HuggingFace token loaded successfully")
    print(f"Token length: {len(hf_token)} characters")
    print(f"Token starts with: {hf_token[:7]}...")
    
    # Try to authenticate with HuggingFace Hub
    try:
        from huggingface_hub import login, whoami
        login(token=hf_token)
        user_info = whoami(token=hf_token)
        print(f"✅ Successfully authenticated as: {user_info.get('name', 'Unknown')}")
    except Exception as auth_error:
        print(f"❌ Authentication failed: {auth_error}")
        print("Please check if your token is valid and has the correct permissions.")

# --- S3 Client Setup ---
# Reusing the pattern from s3_service
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

s3_client = None
if all([aws_access_key_id, aws_secret_access_key, aws_region]):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        print("S3 client initialized successfully for image download.")
    except Exception as e:
        print(f"Warning: Error creating S3 client for image download: {e}. Will attempt direct download.")
else:
    print("Warning: AWS credentials not fully configured for S3 client. Will attempt direct download.")

# Load Image Captioning Model
try:
    if hf_token:
        print("Loading model with HuggingFace token authentication...")
        print(f"Attempting to access repository: rcfg/FashionBLIP-1")
        
        # Try different authentication approaches
        try:
            # Method 1: Using token parameter directly
            processor = BlipProcessor.from_pretrained("rcfg/FashionBLIP-1", token=hf_token)
            model = BlipForConditionalGeneration.from_pretrained("rcfg/FashionBLIP-1", token=hf_token)
        except Exception as token_error:
            print(f"❌ Direct token method failed: {token_error}")
            print("Trying alternative authentication method...")
            
            # Method 2: Using use_auth_token parameter (for older versions)
            try:
                processor = BlipProcessor.from_pretrained("rcfg/FashionBLIP-1", use_auth_token=hf_token)
                model = BlipForConditionalGeneration.from_pretrained("rcfg/FashionBLIP-1", use_auth_token=hf_token)
                print("✅ Alternative authentication method worked!")
            except Exception as alt_error:
                print(f"❌ Alternative method also failed: {alt_error}")
                raise alt_error
    else:
        print("Loading model without authentication (assuming public access)...")
        processor = BlipProcessor.from_pretrained("rcfg/FashionBLIP-1")
        model = BlipForConditionalGeneration.from_pretrained("rcfg/FashionBLIP-1")
    
    print("✅ Processor and Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n🔧 Troubleshooting suggestions:")
    print("1. Check if your HuggingFace token is valid:")
    print("   - Go to https://huggingface.co/settings/tokens")
    print("   - Verify the token has 'Read' permissions")
    print("2. Verify repository access:")
    print("   - Go to https://huggingface.co/rcfg/FashionBLIP-1")
    print("   - Ensure you have access to this private repository")
    print("3. Check your .env file:")
    print("   - Ensure HUGGINGFACE_TOKEN=your_token_here is correctly set")
    print("4. Try regenerating your HuggingFace token")
    print("5. Ensure the repository name 'rcfg/FashionBLIP-1' is correct")
    raise e

# Move model to GPU if available
model = model.to(device)
print(f"✅ Model loaded and moved to {device}")

# Enable evaluation mode for inference
model.eval()

async def generate_caption(image_url: str):
    """Generate an image caption from an image URL (downloads from S3 or public)."""
    try:
        print(f"📷 Received Image URL: {image_url}")

        image_bytes = None

        # Attempt to download using S3 client first
        if s3_client:
            try:
                parsed_url = urlparse(image_url)
                if parsed_url.netloc.endswith('.amazonaws.com'): # Basic check for S3 URL
                    # Assumes bucket name is part of the hostname or path depending on URL format
                    # Example: https://<bucket-name>.s3.<region>.amazonaws.com/<key>
                    # Example: https://s3.<region>.amazonaws.com/<bucket-name>/<key>
                    bucket_name = None
                    object_key = parsed_url.path.lstrip('/')

                    host_parts = parsed_url.netloc.split('.')
                    if len(host_parts) > 3 and host_parts[1] == 's3': # Format: <bucket>.s3... or s3.<region>...<bucket>/key
                        if host_parts[0] != 's3':
                             bucket_name = host_parts[0]
                        else:
                             # Try extracting bucket from path
                             path_parts = object_key.split('/', 1)
                             if len(path_parts) > 1:
                                 bucket_name = path_parts[0]
                                 object_key = path_parts[1]

                    if bucket_name and object_key:
                        print(f"Attempting S3 download: Bucket={bucket_name}, Key={object_key}")
                        s3_response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                        image_bytes = s3_response['Body'].read()
                        print("✅ Image Downloaded via S3")
                    else:
                         print("Could not determine S3 bucket/key from URL, falling back to direct download.")
                else:
                    print("URL does not look like an S3 URL, falling back to direct download.")
            except Exception as s3_error:
                print(f"S3 download failed: {s3_error}. Falling back to direct download.")

        # Fallback: Attempt direct download (for public URLs or if S3 failed)
        if image_bytes is None:
            print("Attempting direct HTTP download...")
            response = requests.get(image_url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            image_bytes = response.content
            print("✅ Image Downloaded via HTTP GET")

        # Process the downloaded image bytes (same as before)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("✅ Image Loaded Successfully")

        # Process image and generate caption (same as before)
        inputs = processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate caption with no gradient computation for efficiency
        with torch.no_grad():
            caption_ids = model.generate(**inputs, max_length=150)
        
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        print("📝 Generated Caption:", caption)

        return {"caption": caption}

    except requests.exceptions.RequestException as http_err:
        print(f"❌ HTTP Error downloading image: {http_err}")
        return {"error": f"Failed to download image from URL: {http_err}"}
    except Exception as e:
        print(f"❌ Error processing image/captioning: {e}")
        # Provide more specific error context if possible
        error_detail = f"Error during processing: {e}"
        if 's3_response' in locals() and 'Error' in s3_response:
             error_detail = f"S3 Error: {s3_response['Error']}"
        return {"error": error_detail}

