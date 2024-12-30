import os
import torch
from google.colab import auth

def configure_environment(environment='colab'):
    if environment == 'colab':
        # Authenticate Google Cloud
        auth.authenticate_user()
        os.system('gcloud auth login')
        
        # Set GCP project from user data
        from google.colab import userdata
        project_id = userdata.get('GCP_EEG_PROJECT_ID')
        os.system(f'gcloud config set project {project_id}')

    elif environment == 'local':
        from dotenv import load_dotenv
        load_dotenv(override=True)

        os.system('python3 -m ensurepip --upgrade')
        os.system('python3 -m pip install --upgrade pip')
        os.system('pip install torch')
    
    # Install dependencies from requirements.txt
    os.system('pip install -r requirements.txt -q')

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.system('nvidia-smi')

    return device
