from dotenv import load_dotenv
import os
from  openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel # DTO와 유사한 클래스 생성
import json

# response = client.responses.create(
#     model = "gpt-4o-mini",
#     input = "서울에 갈 만한 여행지 추천해줘"
        
# )
    # response = client.responses.create(
    #     model="gpt-4o-mini",
    #     input=[
    #         {
    #             "role": "system",
    #             "content": [
    #                 {
    #                     "type": "input_text",
    #                     "text": "너는 json 생성기다. 반드시 json형식으로만 출력 해줘. 두개 컬럼 '제목', '내용'을 포함하고 5개만 출력"
    #                 }
    #             ]
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "input_text",
    #                     "text": requst.content
    #                 }
    #             ]
    #         }
    #     ]
    # )
load_dotenv(override=True)
app = FastAPI(title = "hak api")
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

class UserRequest(BaseModel):
    content:str

@app.get("/")
def root():
    return { "message" : "hi"}

@app.post("/generate")
def openapi(req:UserRequest):
    prompt = f"""
{req.content}
아래 json 배열 형식으로만 하루 3끼 응답해 줘.
[
    {{
        "meal":"아침/점심/저녁",
        "type":"관광/자연/음식",
        "desc":"간단설명",
        "trans":"대중표통팁"
    }}
]
    """
# 아래 json 배열 형식으로만 10개를 응답해 줘.
# [
#     {{
#         "name":"장소명",
#         "type":"관광/자연/음식",
#         "desc":"간단설명",
#         "trans":"대중표통팁"
#     }},
#     {{
#         "name":"장소명",
#         "type":"관광/자연/음식",
#         "desc":"간단설명",
#         "trans":"대중표통팁"
#     }}
# ]    

    # 표준적인 Chat Completion 호출 방식 사용
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "너는 JSON 생성기다."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        # 모델이 JSON 응답을 보내도록 강제 (지원되는 모델에서만 사용 가능)
        response_format={"type": "json_object"}
    )
    output_text = response.choices[0].message.content
    
    try:
        parsed_output = json.loads(output_text)
        return {"message": parsed_output}
    except json.JSONDecodeError:
        return {"error": "JSON 파싱에 실패했습니다.", "raw_content": output_text}
