# app/services/llm_service.py

import logging
from typing import Dict
import google.generativeai as genai
import json
import re
from app.core.config import settings
from app.core.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)


class LLMService:
    """Service to handle all LLM interactions with Gemini."""

    def __init__(self):
        self.model = None
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini client."""
        try:
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not configured")
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("✅ Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise LLMGenerationError(f"Gemini initialization failed: {e}") from e

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> str:
        """Generate a response using Gemini."""
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            if not response.candidates:
                raise LLMGenerationError("Gemini returned no candidates")

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise LLMGenerationError(
                    f"Gemini candidate has no content (finish_reason={candidate.finish_reason})"
                )

            result = "".join(part.text for part in candidate.content.parts if part.text)
            if not result.strip():
                raise LLMGenerationError(
                    f"Gemini returned empty text (finish_reason={candidate.finish_reason})"
                )
            return result

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise LLMGenerationError(f"Gemini generation failed: {e}") from e

    def generate_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> Dict:
        """Generate JSON response safely."""
        try:
            json_system_prompt = system_prompt + "\n\nReturn ONLY valid JSON format. Do not include any other text or markdown."

            response = self.generate_response(
                system_prompt=json_system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not response or not response.strip():
                return {"error": "Empty response from model"}

            # Remove markdown code blocks if present
            clean_response = re.sub(r"```json\s*|\s*```", "", response).strip()
            if not clean_response:
                return {"error": "No valid JSON content in response"}

            parsed_json = json.loads(clean_response)
            return parsed_json

        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response: {e}\nResponse: {response}"}
        except Exception as e:
            return {"error": f"JSON generation failed: {e}"}


# Create global instance
llm_service = LLMService()
