import json
import boto3
import io
import os
import base64
import re
from pypdf import PdfReader
import logging
import datetime 

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO) # INFO 레벨로 설정하여 상세 로그를 볼 수 있습니다.

# AWS 서비스 클라이언트 초기화 (전역으로 한 번만 초기화)
s3_client = boto3.client('s3')
dynamodb_resource = boto3.resource('dynamodb') # Resource 객체로 변경 (테이블 객체 얻기 위함)
bedrock_runtime_client = boto3.client('bedrock-runtime') # 클라이언트 이름 변경

# 설정값
BOOK_META_TABLE_NAME = '' #실제 AWS 리소스 이름으로 변경
PROCESSED_TEXT_BUCKET_NAME = '' #실제 AWS 리소스 이름으로 변경
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# DynamoDB 테이블 객체 초기화 (전역으로 한 번만 초기화)
book_meta_table = dynamodb_resource.Table(BOOK_META_TABLE_NAME)

# Bedrock 토큰 제한 (사용하는 모델에 따라 조절)
MAX_BEDROCK_INPUT_LENGTH = 100000 
MAX_BEDROCK_OUTPUT_TOKENS = 3000

def download_and_extract_text_from_s3(bucket_name, key):
    """
    S3에서 파일을 다운로드하고, 파일 형식에 따라 텍스트를 추출합니다.
    (PDF: pypdf, TXT: 디코딩)
    """
    logger.info(f"Attempting to download and extract text from s3://{bucket_name}/{key}")
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        file_ext = key.lower().split('.')[-1]
        
        file_content_bytes = obj['Body'].read()

        if file_ext == 'txt':
            text = file_content_bytes.decode('utf-8')
            logger.info(f"Successfully extracted text from TXT file: {key}")
            return text

        elif file_ext == 'pdf':
            reader = PdfReader(io.BytesIO(file_content_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or "" # None 반환 방지
            logger.info(f"Successfully extracted text from PDF file: {key}")
            return text

        else:
            logger.error(f"Unsupported file type for {key}. Only PDF and TXT are supported.")
            raise ValueError(f"Unsupported file type: {file_ext}. (pdf, txt만 지원)")

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"File not found in S3: s3://{bucket_name}/{key}")
        raise FileNotFoundError(f"File not found: s3://{bucket_name}/{key}")
    except Exception as e:
        logger.error(f"Error downloading or extracting text from {key}: {e}", exc_info=True)
        raise

def analyze_book_with_bedrock(extracted_text, book_id):
    """
    Bedrock Messages API를 사용하여 추출된 텍스트를 분석하고 구조화된 JSON을 반환합니다.
    """
    logger.info(f"Sending text for Bedrock analysis using Messages API for book ID: {book_id}")

    # Messages API 형식에 맞는 요청 본문 (body) 구성
    # system, messages 필드가 필수로 들어갑니다.
    messages_api_body = {
        "anthropic_version": "bedrock-2023-05-31", # Messages API 버전
        "max_tokens": MAX_BEDROCK_OUTPUT_TOKENS,
        "temperature": 0.7,
        "system": """You are an expert book analyst. Your task is to analyze the provided book text and extract key information in a structured JSON format. Ensure the JSON is valid and comprehensive. The response should be ONLY a valid JSON object, without any preamble or additional text outside the JSON. Do not include markdown (```json) tags.""", # 시스템 프롬프트
        "messages": [
            {
                "role": "user", # 사용자 메시지 시작
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Based on the following text, please provide detailed and vivid explanations for each of the following aspects. 
For the protagonist, describe in rich detail including name, estimated age, gender, clothing style, hair color and style, eye color, and any other distinctive physical features. 
Also include personality, background, and motivations. 
The goal is to provide enough visual and narrative detail to create an AI-generated text game and it's storyline.

Please express the worldbuilding and the temporal/spatial settings as vividly and specifically as possible.
For the key events, provide a detailed summary of the most significant events in the story, including their impact on the protagonist and the world.
key events will be used in building the main storyline of the game.

The total maximum token limit is 3000, so be concise but thorough.
1.  **"title"**: The title of the book.
2.  **"author"**: The author of the book.
3.  **"genre"**: The genre of the book.
4.  **"protagonist"**: The main character or protagonist of the book.
    * "name": character name or nickname.
    * "age": estimated age.
    * "gender": (string)
    * "personality": personality summary.
    * "background": background summary.
    * "appearance": physical appearance.
    * "summary": Analyze the following story and extract a concise visual description of the main protagonist.
        Output it as **a single line**, using comma-separated phrases only — no full sentences or bullet points.

        Format:
        [Estimated Age], [Gender], [Skin Tone], [Eye Color], [Hair Color], [General Clothing Style and Era or Region], [Optional: Personality expression or visual vibe]

        Guidelines:
        - Avoid sensitive regional, racial, or political labels. Use descriptive, neutral terms (e.g., "desert region" instead of "Middle Eastern", "19th-century rural attire" instead of "American working-class").
        - If any detail is missing, infer plausibly from context. Do not leave blanks.
        - Ensure the description is compact and under 300 characters.
        - Do not mention the word “style”, “tone”, or “era” explicitly — describe visually instead.

        Example:
        Late 20s, Male, Sun-kissed skin, Hazel eyes, Black wavy hair, Simple tunic and scarf from a dry, rural land, Calm but alert expression
5.  **"worldbuilding"**: environment, atmosphere, rules, tone, etc.
6.  **"temporal_spatial_setting"**: historical context, geography, culture, architecture, etc.
7.  **"plot_summary"**: A concise summary of the book's beginning or setup (around 100-200 words).
8.  **"key_events"**: An array of 5 to 10 key plot points or major episodes. Each episode should have:
    * "episode_num": (integer)
    * "event_summary": Detailed descriptions of the key event.
9.  **"ending_summary"**: A concise summary of how the story concludes.
10.  **"book_overview"**: A general overview of the entire book, its genre, and main themes (around 200-300 words).

If a piece of information is not clearly available, use "N/A" or "Not found".

Book Text:
<text_to_analyze>{extracted_text}</text_to_analyze>
"""
                    }
                ]
            }
        ]
    }

    try:
        # invoke_model API는 동일하지만, body의 형식이 완전히 변경됩니다.
        bedrock_response = bedrock_runtime_client.invoke_model(
            body=json.dumps(messages_api_body), # 새로운 body 사용
            modelId=BEDROCK_MODEL_ID, # BEDROCK_MODEL_ID는 동일
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(bedrock_response['body'].read())
        
        # Messages API 응답 형식에서 AI 텍스트 추출 방식 변경
        ai_analysis_raw_text = ""
        for content_block in response_body.get('content', []):
            if content_block.get('type') == 'text':
                ai_analysis_raw_text += content_block.get('text', '')

        logger.info(f"Received raw AI analysis response for {book_id}: {ai_analysis_raw_text[:500]}...")

        # AI 응답 파싱
        book_analysis_data = json.loads(ai_analysis_raw_text)
        logger.info(f"AI response parsed as JSON successfully for book ID: {book_id}.")
        return book_analysis_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON for book ID {book_id}: {e}. Raw AI text: {ai_analysis_raw_text}", exc_info=True)
        # 파싱 실패 시, 최소한의 정보라도 저장하기 위해 설정
        return {
            "title": os.path.basename(book_id),
            "author": "N/A",
            "prologue_summary": "AI analysis failed to parse. Raw text: " + ai_analysis_raw_text[:200],
            "episode_summaries": [],
            "ending_summary": "N/A",
            "book_overview": "AI analysis failed to provide a valid JSON output."
        }
    except Exception as e:
        logger.error(f"Error during Bedrock analysis for book ID {book_id}: {e}", exc_info=True)
        raise # Bedrock 통신 오류는 상위에서 처리하도록 다시 발생


def save_metadata_to_dynamodb(
    book_id,
    analysis_data,
    original_s3_key,
    processed_s3_key
):
    """
    분석된 책 메타데이터를 DynamoDB 테이블에 저장합니다.
    """
    logger.info(f"Saving metadata to DynamoDB for bookId: {book_id}")
    
    # AI 분석 데이터에 제목/저자가 없으면 파일 이름에서 유추
    if analysis_data.get('title', 'N/A') == 'N/A':
        analysis_data['title'] = book_id # book_id는 파일 이름에서 확장자 제거한 것이므로 제목으로 사용
    if analysis_data.get('author', 'N/A') == 'N/A' and 'author' not in analysis_data:
        analysis_data['author'] = "Unknown"

    item_to_put = {
        'bookId': book_id,
        'title': analysis_data.get('title', 'Unknown Title'),
        'author': analysis_data.get('author', 'Unknown Author'),
        'genre': analysis_data.get('genre', 'Unknown'),
        'originalS3Key': original_s3_key,
        'processedS3Key': processed_s3_key,
        'protagonist': analysis_data.get('protagonist', []),
        'worldbuilding': analysis_data.get('worldbuilding', ''),
        'temporalSpatialSetting': analysis_data.get('temporal_spatial_setting', ''),
        'plotSummary': analysis_data.get('plot_summary', ''),
        'keyEvents': analysis_data.get('key_events', []),
        'endingSummary': analysis_data.get('ending_summary', ''),
        'bookOverview': analysis_data.get('book_overview', ''),
        'status': 'PROCESSED',
        'lastProcessedDate': int(datetime.datetime.now().timestamp() * 1000) # 현재 시간 기록
    }
    book_meta_table.put_item(Item=item_to_put)
    logger.info(f"Book metadata saved successfully for {book_id}.")

def save_processed_text_to_s3(
    bucket_name,
    output_key,
    text_content
):
    """
    추출된 텍스트를 지정된 S3 버킷에 저장합니다.
    """
    logger.info(f"Saving extracted text to s3://{bucket_name}/{output_key}")
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=text_content.encode('utf-8')
        )
        logger.info(f"Extracted text saved to s3://{bucket_name}/{output_key} successfully.")
    except Exception as e:
        logger.error(f"Failed to save processed text to S3: s3://{bucket_name}/{output_key}: {e}", exc_info=True)
        raise # S3 저장 실패는 중요한 오류이므로 다시 발생

def handle_processing_error(book_id, file_key, error_message):
    """
    파일 처리 중 오류 발생 시 DynamoDB에 FAILED 상태를 기록합니다.
    """
    logger.error(f"Error encountered for book ID {book_id} (file: {file_key}): {error_message}", exc_info=True)
    
    book_meta_table.put_item(
        Item={
            'bookId': book_id,
            'status': 'FAILED',
            'errorMessage': error_message,
            'originalS3Key': file_key,
            'lastProcessedDate': int(datetime.datetime.now().timestamp() * 1000)
        }
    )
    logger.info(f"Error status 'FAILED' recorded for bookId: {book_id}")

def generate_images_from_prompt(prompt_text, samples=1):
    payload = {
        "text_prompts": [{"text": prompt_text}],
        "cfg_scale": 10,
        "seed": 42,
        "steps": 50,
        "style_preset": "photographic",
        "samples": samples
    }

    response = bedrock_runtime_client.invoke_model(
        modelId="stability.stable-diffusion-xl-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    result = json.loads(response['body'].read())
    images = []

    for i, artifact in enumerate(result.get("artifacts", [])):
        image_b64 = artifact.get("base64")
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            images.append(image_bytes)

    return images

def extract_tags_from_prompt(prompt):
    return list(set(re.findall(r'\b[a-zA-Z]{3,}\b', prompt.lower())))

def save_images_and_tags_to_s3(images, tags, s3_prefix, episode_id):
    for idx, image_bytes in enumerate(images):
        image_key = f"{s3_prefix}/episode_{episode_id}/image_{idx + 1}.png"
        s3_client.put_object(Bucket=PROCESSED_TEXT_BUCKET_NAME, Key=image_key, Body=image_bytes)

    tag_key = f"{s3_prefix}/episode_{episode_id}/tags.json"
    s3_client.put_object(
        Bucket=PROCESSED_TEXT_BUCKET_NAME,
        Key=tag_key,
        Body=json.dumps({"tags": tags}).encode("utf-8"),
        ContentType="application/json"
    )



def lambda_handler(event, context):
    """
    S3 파일 업로드 이벤트를 처리하는 Lambda 함수의 메인 핸들러.
    """
    logger.info(f"Lambda function triggered by S3 event. Event details: {json.dumps(event)}")

    # 모든 레코드를 성공적으로 처리했는지 여부를 추적 (부분 성공 가능)
    overall_status = {'statusCode': 200, 'body': 'Processing initiated for all records. Check logs for details.'}

    for record in event['Records']:
        source_bucket_name = record['s3']['bucket']['name']
        file_key = record['s3']['object']['key']
        book_id = os.path.splitext(os.path.basename(file_key))[0] # 예: raw/my_book.pdf -> my_book
        
        logger.info(f"--- Starting processing for book ID: {book_id} (File: {file_key}) ---")

        try:
            # 1. S3에서 파일 내용 다운로드 및 텍스트 추출
            extracted_text = download_and_extract_text_from_s3(source_bucket_name, file_key)
            
            # 2. Bedrock으로 텍스트 분석
            book_analysis_data = analyze_book_with_bedrock(extracted_text, book_id) 
            
            # 3. DynamoDB에 메타데이터 저장
            processed_s3_key = f"processed_texts/{book_id}.txt"
            save_metadata_to_dynamodb(
                book_id,
                book_analysis_data,
                file_key,
                processed_s3_key
            )

            # 4. 추출된 전체 텍스트를 다른 S3 버킷에 저장
            save_processed_text_to_s3(
                PROCESSED_TEXT_BUCKET_NAME,
                processed_s3_key,
                extracted_text
            )
            logger.info(f"--- Successfully processed book ID: {book_id} ---")
            
        except FileNotFoundError as fnfe:
            # S3에서 파일을 찾지 못한 경우
            handle_processing_error(book_id, file_key, f"File not found: {fnfe}")
            overall_status['statusCode'] = 202 # 일부 실패 의미
        except ValueError as ve: 
            # 지원하지 않는 파일 형식 등 예상된 입력 오류
            handle_processing_error(book_id, file_key, f"Data validation error: {ve}")
            overall_status['statusCode'] = 202
        except Exception as e:
            # 기타 예상치 못한 오류
            handle_processing_error(book_id, file_key, f"An unexpected error occurred during processing: {e}")
            overall_status['statusCode'] = 500 # 심각한 오류
        
        # 각 레코드 처리 후 다음 레코드로 넘어감 (continue는 try 블록 안에서만 동작)

    return overall_status
