from youtube_transcript_api import YouTubeTranscriptApi as yta
import re

# text = " hello world"
# match = re.search(r"world", text)
# if match:
#    print("Matched:", match.group())

def extract_video_id(url):
    # Regular expression to extract video ID from various YouTube URL formats
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id):
    ytt_api = yta()
    transcript = ytt_api.fetch(video_id, languages=['ko', 'en']).to_raw_data()
    full_text = "\n".join( [item['text'] for item in transcript] )
    return full_text

if __name__ == "__main__":
    # print("자동샐행합니다.")
    youtube_url = input("유튜브 영상 URL을 입력하세요: ")
    print("입력된 URL:", youtube_url)
    video_id = extract_video_id(youtube_url)
    if video_id:
        print("추출된 비디오 ID:", video_id)
        transcript_text = get_transcript(video_id)
        print("추출된 자막:\n", transcript_text)
    else:
        print("유효한 YouTube URL이 아닙니다.")
