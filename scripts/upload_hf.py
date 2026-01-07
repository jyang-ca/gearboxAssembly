import os
from huggingface_hub import HfApi, create_repo

# Configuration
TOKEN = "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"
REPO_ID = "yjsm1203/Galaxea-Gearbox-Assembly-R1-Policies"
FOLDER_PATH = "./submission"

def main():
    print(f"Starting upload to {REPO_ID}...")
    
    # Initialize API
    api = HfApi(token=TOKEN)
    
    # 1. Create Repository
    try:
        url = create_repo(repo_id=REPO_ID, token=TOKEN, private=False, exist_ok=True)
        print(f"Repository ready: {url}")
    except Exception as e:
        print(f"Warning: Could not create repo (might exist or auth error): {e}")

    # 2. Upload Folder
    try:
        future = api.upload_folder(
            folder_path=FOLDER_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload trained policies for Approach, Grasp, and Transport-1"
        )
        print("Upload started/completed.")
        print(future)
        print("\n[SUCCESS] Model artifacts successfully uploaded to Hugging Face!")
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")

if __name__ == "__main__":
    main()
