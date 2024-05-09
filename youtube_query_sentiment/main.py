from video_data import search_videos
from sentiment_analysis import vader_videos_sentiment
from google.cloud import storage
import datetime
import base64



""" 2nd gen cloud functions - pub/sub trigger"""
import functions_framework

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    # Print out the data from Pub/Sub, to prove that it worked
    print(base64.b64decode(cloud_event.data["message"]["data"]))


    """
    #1st gen 
    def hello_pubsub(event, context):
        '''Triggered from a message on a Cloud Pub/Sub topic.
        Args:
            event (dict): Event payload.
            context (google.cloud.functions.Context): Metadata for the event.
        '''
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        print(pubsub_message)
    """


def youtube_query_sentiment(query='Bitcoin', max_videos_count=100):
    
    df = search_videos(query, max_videos_count)
    df_sentitment = vader_videos_sentiment(df)
    
    output = df_sentitment[['title', 'upload_date', 'views', 'likes', 'vader_compound', 'vader_sentiment']]
    
    current_date =  datetime.datetime.now().strftime('%Y-%m-%d')
    file_name = f'youtube_{query}_sentiment_{current_date}.csv'
    local_path = f'/tmp/{file_name}'  # cloud function /tmp dir

    output.to_csv(local_path, encoding='utf-8', header=True, index=None)
   
    return file_name, local_path


# Upload files from local file_path to Cloud Storage
def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {bucket}.")


if __name__ == "__main__":
    
    file_name, local_path = youtube_query_sentiment('Bitcoin', 100) # save a output file to local path
    bucket_name = 'youtube_videos_sentiment'
    destination_file_name = file_name  # Optional: include folders in the path

    # Upload the processed file 
    upload_to_gcs(local_path, bucket_name, destination_file_name)


