# Sample Python code for youtube.commentThreads.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

# Google cloud API https://console.cloud.google.com/apis/credentials?project=ivory-amplifier-421507&supportedpurview=project
# Python package: https://github.com/googleapis/google-api-python-client/tree/main/googleapiclient



import googleapiclient.discovery
import pandas as pd
import json
# import sys
import os


# json parsing
def process_comments(response_items):
    comments = []
    for comment in response_items:
            author = comment['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment_text = comment['snippet']['topLevelComment']['snippet']['textOriginal']
            publish_time = comment['snippet']['topLevelComment']['snippet']['publishedAt']
            comment_info = {'author': author, 
                            'comment': comment_text, 
                            'published_at': publish_time
                            }
            comments.append(comment_info)
    # print(f'Finished processing {len(comments)} comments.')
    return comments


def run_youtube_etl():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    video_id = 'CzhkXYdeh-c'
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyCpHdX77lnBSBRRP5K71XvR9_EtynfYFXE"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part = "snippet, replies",
        videoId = video_id   
    )
    response = request.execute()
    
    # save an example for the comments json
    # with open('youtube_comments_data.json', 'w') as f:
    #     json.dump(response, f, indent=4)

    comments_list = process_comments(response['items'])

    # comments have multiple pages. look for the nextPageToken in each response to retrieve all comments. 
    # Whenever another page of comments was available, the response would include a nextPageToken.
    while response.get('nextPageToken', None):
        request = youtube.commentThreads().list(
            part='id,replies,snippet',
            videoId=video_id,
            pageToken=response['nextPageToken']
        )
        response = request.execute()
        comments_list.extend(process_comments(response['items']))


    comments_df = pd.DataFrame(comments_list)
    # comments_df.to_csv(f's3://airflow-youtube-comments-test/youtube_comments_{video_id}.csv')
    comments_df.to_csv(f'youtube_comments_{video_id}.csv', encoding='utf-8', header=True, index=None)
    # print(f'This video has {len(comments_df)} comments.')


# if __name__ == "__main__":
#     main(VIDEO_ID)
    # # Check if any arguments were passed (first argument is the script name)
    # if len(sys.argv) > 1:
    #     VIDEO_ID = sys.argv[1]  # The first command line argument
    #     print(f'The Youtube video is {VIDEO_ID}')
    #     main(VIDEO_ID)
    # else:
    #     print("No video provided.")
    #     # Exit the script with an exit code of 1, indicating an error or issue
    #     sys.exit(1)

    



