1. Write Python Script
Ensure that the Python script can save its output data to Cloud Storage, BigQuery, or other locations.

2. Deploy Your Python Script:
Deploy the script as a Cloud Function or on Cloud Run. For Cloud Functions: bash: gcloud functions deploy your-function-name --runtime python3.X --trigger-http
OR
Use the Cloud Function console. 
! 1st gen and 2nd gen have different settings.


3. Schedule and Run Python Scripts Daily
Cloud Scheduler:
Schedule your Python script to run at specific intervals (e.g., daily) using cron expressions.
Cloud Scheduler can trigger HTTP endpoints, Pub/Sub messages, and more.

4. Store Output Data on Google Cloud
Cloud Storage:
If your Python scripts generate output files (CSV files), save them locally and upload to GCS OR directly to Cloud Storage (GCS).
GCS is a scalable and cost-effective object storage service.

5. Visulization
Google Looker Studio Dashboard Demo: 
https://lookerstudio.google.com/reporting/08affbf4-7a19-4c4e-9449-e2bc480cf61a



### Do not put api key in the deployed script 

import os
DEVELOPER_KEY = os.getenv("DEVELOPER_KEY")
