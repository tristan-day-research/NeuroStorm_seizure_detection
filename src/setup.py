import os
import torch
from google.colab import auth
from google.colab import userdata

def configure_environment(environment='colab'):
    if environment == 'colab':
        # Retrieve GitHub and GCP credentials
        token = userdata.get('GITHUB_PAT')
        github_email = userdata.get('GITHUB_EMAIL')
        github_username = userdata.get('GITHUB_USER_NAME')
        project_id = userdata.get('GCP_EEG_PROJECT_ID')
        gcp_bucket_name = userdata.get('GCP_EEG_BUCKET_NAME')
        gcp_file_prefix = userdata.get('EEG_GCP_FILEPATH')

        # Configure Git with credentials
        os.system(f'git config --global user.email "{github_email}"')
        os.system(f'git config --global user.name "{github_username}"')
        os.system(f'gcloud config set project {project_id}')

        auth.authenticate_user()
        
        print(f"GCP Project Set")
        print("Git configured with your user data.")

    else:
        print("Running in local environment. Ensure .env is configured.")

    return gcp_bucket_name, gcp_file_prefix, project_id


   
