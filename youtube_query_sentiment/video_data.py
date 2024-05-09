# Sample Python code for youtube.commentThreads.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

# Google cloud API https://console.cloud.google.com/apis/credentials?project=ivory-amplifier-421507&supportedpurview=project
# Python package: https://github.com/googleapis/google-api-python-client/tree/main/googleapiclient


import googleapiclient.discovery
import pandas as pd
import datetime


# Function to get video statistics
def get_video_details(youtube, video_ids):
    videos = []
    response = youtube.videos().list(
        part="snippet,statistics",
        id=','.join(video_ids)
    ).execute()
    
    for item in response['items']:
        snippet = item['snippet']
        stats = item['statistics']
        video_info = {
            'title': snippet['title'],
            'description': snippet.get('description', ''),
            'upload_date': snippet['publishedAt'],
            'views': stats.get('viewCount', '0'),
            'likes': stats.get('likeCount', '0')
        }
        videos.append(video_info)
    
    return videos

# Function to search for videos based on a query
def search_videos(query, max_results=100):

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "YOUR_KEY"

    start_date = (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat() + 'Z'
    end_date = datetime.datetime.now().isoformat() + 'Z'

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    search_response = youtube.search().list(
        q = query,
        type = "video",
        part = "id,snippet",
        maxResults = max_results,
        publishedAfter = start_date,
        publishedBefore = end_date
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]
    
    # Fetch the detailed information of each video
    videos_list = get_video_details(youtube, video_ids)
    videos_df = pd.DataFrame(videos_list)
    # videos_df.to_csv(f'data/youtube_{query}_videos_{max_results}.csv', encoding='utf-8', header=True, index=None)
    
    return videos_df



