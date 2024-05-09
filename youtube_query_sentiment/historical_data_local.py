'''
Historical data pull for youtube sentiment research 
1. all-in-one script for ease of use
2. Save daily output in csv files locally
3. How many days could be pulled depends on the daily limit of google API token usage
'''

import pandas as pd
import datetime
import googleapiclient.discovery
from youtube_video_sentiment_analysis import vader_videos_sentiment
from youtube_video_info import get_video_details 


# define google api args
video_id = 'CzhkXYdeh-c'
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyCpHdX77lnBSBRRP5K71XvR9_EtynfYFXE"
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = DEVELOPER_KEY)

query = 'Bitcoin'


for i in range(120):
    # i days ago from today
    start_date = (datetime.datetime.now() - datetime.timedelta(days=i+1)).isoformat() + 'Z'
    end_date = (datetime.datetime.now() - datetime.timedelta(days=i)).isoformat() + 'Z'

    search_response = youtube.search().list(
        q = query,
        type = "video",
        part = "id,snippet",
        maxResults = 100,
        publishedAfter = start_date,
        publishedBefore = end_date
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]

    # Fetch the detailed information of each video
    videos_list = get_video_details(youtube, video_ids)
    videos_df = pd.DataFrame(videos_list)

    # add sentiment scores
    df_sentitment = vader_videos_sentiment(videos_df)

    # final output dataframe    
    output = df_sentitment[['title', 'upload_date', 'views', 'likes', 'vader_compound', 'vader_sentiment']]

    # Save the data to local machine
    date_name = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
    output.to_csv(f'data/youtube_{query}_sentiment_{date_name}.csv', encoding='utf-8', header=True, index=None)
    print(i)
