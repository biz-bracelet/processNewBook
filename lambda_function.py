import json
import boto3
import io
import os
from pypdf import PdfReader

def extract_text_from_s3(bucket_name, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    file_ext = key.lower().split('.')[-1]

    if file_ext == 'txt':
        text = obj['Body'].read().decode('utf-8')
        return text

    elif file_ext == 'pdf':
        pdf_bytes = obj['Body'].read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    else:
        raise ValueError("지원하지 않는 파일 형식입니다. (pdf, txt만 지원)")

def analyze_book_with_bedrock(extracted_text, fileKey):
    bedrock_runtime = boto3.client('bedrock-runtime')
    bedrock_prompt = ""  # 나중에 수정
    bedrock_prompt = f"""
            Here is a book text. Please analyze it and extract the following information in a structured JSON format:
            1.  **"title"**: The title of the book.
            2.  **"author"**: The author of the book.
            3.  **"prologue_summary"**: A concise summary of the book's beginning or setup (around 100-200 words).
            4.  **"episode_summaries"**: An array of 3 to 5 key plot points or major episodes. Each episode should have:
                * "episode_num": (integer)
                * "summary": A brief description of the key event (around 50-100 words).
            5.  **"ending_summary"**: A concise summary of how the story concludes (around 100-200 words).
            6.  **"book_overview"**: A general overview of the entire book, its genre, and main themes (around 200-300 words).

            If a piece of information is not clearly available, use "N/A" or "Not found".

            Make sure your response is a valid JSON object.

            Book Text: {extracted_text}
        """
    bedrock_response = bedrock_runtime.invoke_model(
                body=json.dumps({
                    "prompt": bedrock_prompt,
                    "max_tokens_to_sample": 2000, # Bedrock 응답 최대 길이 (분석 결과가 길어질 수 있으므로 충분히 줍니다)
                    "temperature": 1, # 창의성 (낮을수록 일관적, 높을수록 창의적)
                    "top_p": 0.9 # 샘플링 다양성
                }),
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                contentType="application/json",
                accept="application/json"
            )
    response_body = json.loads(bedrock_response['body'].read())
    ai_analysis_raw_text = response_body.get('completion', response_body.get('generations', [{}])[0].get('text', '')) # 모델 응답 형식에 따라 달라질 수 있음

    # AI 응답 파싱 (JSON 형식으로 출력되도록 프롬프트 설계했으므로 JSON 파싱 시도)
    try:
        book_analysis_data = json.loads(ai_analysis_raw_text)
    except json.JSONDecodeError as e:
        # 파싱 실패 시, 최소한의 정보라도 저장하기 위해 설정
        book_analysis_data = {
            "title": os.path.basename(fileKey),
            "author": "N/A",
            "prologue_summary": "AI analysis failed to parse. Raw text: " + ai_analysis_raw_text[:200],
            "episode_summaries": [],
            "ending_summary": "N/A",
            "book_overview": "N/A"
        }

    return book_analysis_data
