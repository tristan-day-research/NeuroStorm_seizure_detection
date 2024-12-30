import os
import torch
from google.colab import auth

# def configure_environment(environment='colab'):


#     GCP_EEG_PROJECT_ID = userdata.get('GCP_EEG_PROJECT_ID')
#     BUCKET_NAME = os.getenv('GCP_EEG_BUCKET_NAME')

#     # Set Git configuration using Python variables
#     !git config --global user.email "{github_email}"
#     !git config --global user.name "{github_username}"

#     if environment == 'colab':
#         # Authenticate Google Cloud
#         auth.authenticate_user()
#         os.system('gcloud auth login')
        
#         # Set GCP project from user data
#         from google.colab import userdata
#         project_id = userdata.get('GCP_EEG_PROJECT_ID')
#         os.system(f'gcloud config set project {project_id}')

#     elif environment == 'local':
#         from dotenv import load_dotenv
#         load_dotenv(override=True)

#         os.system('python3 -m ensurepip --upgrade')
#         os.system('python3 -m pip install --upgrade pip')
#         os.system('pip install torch')
    
#     # Install dependencies from requirements.txt
#     os.system('pip install -r requirements.txt -q')


import os
from google.colab import userdata

def configure_environment(environment='colab'):
    if environment == 'colab':
        # Retrieve GitHub and GCP credentials
        token = userdata.get('GITHUB_PAT')
        github_email = userdata.get('GITHUB_EMAIL')
        github_username = userdata.get('GITHUB_USER_NAME')
        project_id = userdata.get('GCP_EEG_PROJECT_ID')
        bucket_name = os.getenv('GCP_EEG_BUCKET_NAME')

        # Configure Git with credentials
        os.system(f'git config --global user.email "{github_email}"')
        os.system(f'git config --global user.name "{github_username}"')
        os.system(f'gcloud config set project {project_id}')
        
        print(f"GCP Project Set: {project_id}")
        print("Git configured with your user data.")

    else:
        print("Running in local environment. Ensure .env is configured.")


   
