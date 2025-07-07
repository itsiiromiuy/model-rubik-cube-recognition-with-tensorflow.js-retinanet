#!/usr/bin/env python3
"""
Deploy Rubik's Cube Recognition to Hugging Face Spaces
"""

import os
from huggingface_hub import HfApi, Repository


def deploy_to_huggingface():
    """Deploy the project to Hugging Face Spaces"""

    # Set your token

    # Initialize HF API
    api = HfApi()

    # Repository details
    repo_id = "itsyuimorii/rubiks-cube-recognition"
    repo_type = "space"

    print(f"Deploying to: https://huggingface.co/spaces/{repo_id}")

    try:
        # Create repository if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            exist_ok=True
        )
        print("✅ Repository created/verified")

        # Upload files
        files_to_upload = [
            "README.md",
            "app.py",
            "requirements.txt",
            "app_simple.py"
        ]

        for file_path in files_to_upload:
            if os.path.exists(file_path):
                print(f"📤 Uploading {file_path}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token
                )
                print(f"✅ {file_path} uploaded successfully")
            else:
                print(f"⚠️ {file_path} not found, skipping...")

        print("\n🎉 Deployment completed successfully!")
        print(f"🌐 Visit your Space: https://huggingface.co/spaces/{repo_id}")

    except Exception as e:
        print(f"❌ Error during deployment: {e}")
        return False

    return True


if __name__ == "__main__":
    # Install required package if not already installed
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        import huggingface_hub

    deploy_to_huggingface()
