from google.cloud import translate_v2
from google.oauth2 import service_account
import os

def translate_text(text, target_language='es', source_language='en'):
    """
    Translate text using Google Translate API.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code (default: 'es' for Spanish)
        source_language (str): Source language code (default: 'en' for English)
    
    Returns:
        str: Translated text
    """
    # Initialize the translation client with explicit credentials
    credentials = service_account.Credentials.from_service_account_file(
        os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '.creds/google.json')
    )
    translate_client = translate_v2.Client(credentials=credentials)
    
    # Perform translation (v2 API)
    result = translate_client.translate(
        text,
        target_language=target_language,
        source_language=source_language
    )

    return result['translatedText']

def main():
    # Example usage
    text_to_translate = "Hello, how are you today?"
    
    # Translate to Spanish
    translated = translate_text(text_to_translate, target_language='es')
    print(f"Original: {text_to_translate}")
    print(f"Translated (ES): {translated}")
    
    # Translate to French
    translated_fr = translate_text(text_to_translate, target_language='fr')
    print(f"Translated (FR): {translated_fr}")

    # Translate to mandarin
    translated_zh = translate_text(text_to_translate, target_language='zh')
    print(f"Translated (ZH): {translated_zh}")

    #Translate to Hindi
    translated_hi = translate_text(text_to_translate, target_language='hi')
    print(f"Translated (HI): {translated_hi}")

if __name__ == '__main__':
    main()