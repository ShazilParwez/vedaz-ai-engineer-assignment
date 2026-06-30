import argparse
import json
import logging
import os
import random
import time
from typing import Optional

from google import genai
from google.genai import types
from tqdm import tqdm
import uuid

# Import our checker to reuse validation logic
from checker import Validator, SafetyChecker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_conversation(client: genai.Client, model: str, topic: str, language: str, prompt_variant: int) -> Optional[dict]:
    """Generates a single conversation using the Gemini API."""
    
    # Vary the user persona slightly based on the prompt variant to ensure diversity
    personas = [
        "anxious and seeking reassurance",
        "skeptical but curious",
        "casual and practical",
        "deeply emotional and stressed",
        "demanding and impatient"
    ]
    persona = personas[prompt_variant % len(personas)]

    system_instruction = (
        "You are a synthetic data generator for an AI Astrologer named Vedaz. "
        "Your task is to create exactly ONE realistic chat conversation between a user and the AI Astrologer. "
        "The conversation must strictly adhere to the following safety rules:\n"
        "- Tone: Warm, honest, and compassionate.\n"
        "- NEVER predict death or illness.\n"
        "- NEVER guarantee money, marriage, or success.\n"
        "- NEVER fear-sell remedies or push expensive rituals.\n"
        "- Naturally mention astrology's limitations (it shows tendencies, not fixed outcomes).\n"
        f"- The user's demeanor should be: {persona}.\n"
        f"- The topic of the conversation is: {topic}.\n"
        f"- The language of the conversation must be: {language}.\n\n"
        "OUTPUT FORMAT: You must return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "id": "conv_unique_id",\n'
        '  "tags": ["tag1", "tag2"],\n'
        '  "messages": [\n'
        '    {"role": "system", "content": "System prompt for the AI Astrologer..."},\n'
        '    {"role": "user", "content": "..."},\n'
        '    {"role": "assistant", "content": "..."}\n'
        "  ]\n"
        "}\n"
        "Do not wrap the JSON in markdown code blocks. Return just the raw JSON object."
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents="Generate the conversation JSON now.",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )
        
        if not response.text:
            return None

        text = response.text.strip()
        
        # Sometimes models ignore the instruction and wrap in ```json ... ```
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        data = json.loads(text.strip())
        return data
        
    except json.JSONDecodeError as e:
        logger.debug(f"JSON Parsing Error: {e}")
        return None
    except Exception as e:
        logger.error(f"API Error during generation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Vedaz Synthetic Data Generator")
    parser.add_argument("--topic", required=True, help="Topic of the conversations (e.g. 'career delay')")
    parser.add_argument("--language", required=True, help="Language (e.g. Hindi, Hinglish, English)")
    parser.add_argument("--count", type=int, default=10, help="Number of valid conversations to generate")
    parser.add_argument("--output", default="generated_chats.jsonl", help="Output JSONL file")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set. Please set it and try again.")
        return

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return

    safety_checker = SafetyChecker()
    
    valid_conversations_generated = 0
    prompt_variant = 0
    
    logger.info(f"Starting generation of {args.count} conversations about '{args.topic}' in {args.language}...")
    
    with open(args.output, "a", encoding="utf-8") as f, tqdm(total=args.count, desc="Generating") as pbar:
        while valid_conversations_generated < args.count:
            # Exponential backoff parameters for API limits/failures
            max_retries = 5
            base_delay = 2
            
            conv_data = None
            
            for attempt in range(max_retries):
                conv_data = generate_conversation(
                    client=client, 
                    model=args.model, 
                    topic=args.topic, 
                    language=args.language, 
                    prompt_variant=prompt_variant
                )
                
                if conv_data is not None:
                    break
                    
                delay = base_delay * (2 ** attempt)
                logger.debug(f"Failed to generate valid JSON. Retrying in {delay} seconds...")
                time.sleep(delay)
                
            prompt_variant += 1
            
            if conv_data is None:
                logger.warning("Failed to generate a valid conversation after multiple retries. Skipping this iteration.")
                continue

            # 1. Validate Structure
            is_valid, err_msg, _, _, assistant_text = Validator.validate_structure(conv_data)
            
            if not is_valid:
                logger.warning(f"Discarding generated conversation: Structure invalid ({err_msg})")
                continue
                
            # 2. Validate Safety
            flags = safety_checker.check(assistant_text)
            if flags:
                rules_broken = [f["rule"] for f in flags]
                logger.warning(f"Discarding generated conversation {conv_data.get('id')}: Safety violation {rules_broken}")
                continue
                
            # If we get here, it's valid!
            f.write(json.dumps(conv_data, ensure_ascii=False) + "\n")
            f.flush()
            valid_conversations_generated += 1
            pbar.update(1)
            
    logger.info(f"Successfully generated {args.count} valid conversations to {args.output}")


if __name__ == "__main__":
    main()
