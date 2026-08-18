import os
import logging
import json
from typing import List, Dict, Any
from app.core.config import settings
from google import genai
from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
            """
            # ROLE AND OBJECTIVE
            You are the official automated representative for Synkro, an AI-native product and growth studio. Your sole objective is to answer user inquiries accurately and professionally using ONLY the information provided in the [KNOWLEDGE BASE] below.
            # STRICT INSTRUCTIONS
            1. ZERO HALLUCINATION: You must base your answers exclusively on the provided [KNOWLEDGE BASE]. Do not infer, guess, or synthesize outside information. Do not use your pre-trained knowledge.
            2. NO CREATIVITY: Present facts exactly as they are described in the source data. Do not use metaphors, creative language, or embellishments. Keep responses concise, direct, and professional.
            3. LANGUAGE MATCHING: The knowledge base contains both English (`text_en`, `answer_en`) and Arabic (`text_ar`, `answer_ar`) data. You must respond in the exact language the user uses to ask the question.
            4. OUT OF SCOPE QUERIES: If a user asks a question that cannot be explicitly answered by the [KNOWLEDGE BASE], you must refuse to answer. Use the following exact fallback phrases:
                - English: I apologize, but I do not have information regarding that topic. Please contact Synkro directly for more details.
                - Arabic: أعتذر، ولكن ليس لدي معلومات حول هذا الموضوع. يرجى التواصل مع سينكرو مباشرة لمزيد من التفاصيل.
            5. FORMATTING: Respond in PLAIN TEXT only. Do NOT use markdown syntax of any kind:
                - No **bold**, no _italics_, no headers (#).
                - No bullet points using -, *, or •.
                - No numbered lists using 1) or 1.
                If listing multiple items, write them as a single flowing sentence separated by commas, or use simple line breaks with plain dashes replaced by full sentences (e.g. "This includes X, Y, and Z.").
            # KNOWLEDGE BASE
            {knowledge_base}
            """
        )

class ResponseGenerator:
    """LLM Answer Generator incorporating retrieved source context."""

    def __init__(self, openai_model: str = None, gemini_model: str = None, groq_api_key: str = None, gemini_api_key: str = None):
        self.groq_api_key = settings.GROQ_API_KEY 
        self.gemini_api_key = settings.GEMINI_API_KEY 
        self.openai_model = settings.OPENAI_MODEL 
        self.gemini_model = settings.GEMINI_MODEL 

        self.use_groq = bool(self.groq_api_key)
        self.use_gemini = bool(self.gemini_api_key)

        # Initialize Groq client
        self.groq_client = None
        if self.use_groq:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info(f"✅ Groq client initialized | model={self.openai_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                self.use_groq = False

        # Initialize Gemini client (google-genai >= 2.18.1 uses genai.Client)
        self.gemini_client = None
        if self.use_gemini:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info(f"✅ Gemini client initialized | model={self.gemini_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self.use_gemini = False

        if not self.use_groq and not self.use_gemini:
            logger.warning("⚠️ No LLM API keys configured. All responses will use deterministic fallback.")

        self.load_knowlege_base_file = self._load_knowledge_base()
        self.knowledge_base = json.dumps(self.load_knowlege_base_file, ensure_ascii=False, indent=2)

    def _load_knowledge_base(self, path: Any = None) -> Any:
        """Loads the Synkro knowledge base JSON from disk."""
        target_path = Path(path) if path else settings.CHUNKS_PATH
        try:
            with open(target_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found at '{target_path}'.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Knowledge base JSON is malformed: {e}")
            return []

        
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None,
    ) -> str:
        """Generates answer using Groq (primary), Gemini (fallback), or
        template fallback (final resort) if both LLM calls fail."""
        formatted_context = self._format_context(context_chunks)
        system_prompt = SYSTEM_PROMPT.format(knowledge_base=self.knowledge_base)
        user_content = f"Question: {query}\n\nRetrieved Knowledge Context:\n{formatted_context}"

        # 1. Try Groq first
        if self.use_groq and self.groq_client:
            try:
                logger.info(f"Generating response using Groq | model={self.openai_model} | query='{query[:80]}...'")
                response = self._call_groq(system_prompt, user_content, history)
                logger.info("✅ Response successfully generated by Groq.")
                return response
            except Exception as e:
                logger.warning(f"Groq completion call failed: {e}. Falling back to Gemini.")

        # 2. Fall back to Gemini
        if self.use_gemini and self.gemini_client:
            try:
                logger.info(f"Generating response using Gemini | model={self.gemini_model} | query='{query[:80]}...'")
                response = self._call_gemini(system_prompt, user_content, history)
                logger.info("✅ Response successfully generated by Gemini.")
                return response
            except Exception as e:
                logger.warning(f"Gemini completion call failed: {e}. Using template fallback.")

        # 3. Final fallback: deterministic template built from context_chunks
        logger.warning("⚠️ No LLM available (Groq & Gemini both failed or unconfigured). Using deterministic fallback generator.")
        return self._fallback_generator(query, context_chunks)

    def _call_groq(
        self, system_prompt: str, user_content: str, history: List[Dict[str, str]] = None
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_content})

        response = self.groq_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()


    def _call_gemini(
        self, system_prompt: str, user_content: str, history: List[Dict[str, str]] = None
    ) -> str:
        # google-genai >= 2.18.1 uses genai.Client with contents list
        contents = []

        # Prepend history turns
        if history:
            for msg in history:
                role = "model" if msg.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        # Add current user turn
        contents.append({"role": "user", "parts": [{"text": user_content}]})

        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.1,
            },
        )
        return response.text.strip()
    

    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "No relevant documents found."

        formatted = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Unknown Document")
            content = chunk.get("content", "").strip()
            formatted.append(f"--- Document [{i}]: {source} ---\n{content}\n")

        return "\n".join(formatted)
    

    def _fallback_generator(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generates structured response from retrieved context when both
        Groq and Gemini are unavailable."""
        if not context_chunks:
            return "I couldn't find any relevant company document information to answer your question."

        top_chunk = context_chunks[0]
        sources_list = ", ".join(sorted({c.get("source", "doc") for c in context_chunks}))

        answer = (
            f"Based on the official company documents ({sources_list}):\n\n"
            f"{top_chunk.get('content', '')}\n\n"
        )

        if len(context_chunks) > 1:
            answer += "Additional related context:\n"
            for c in context_chunks[1:]:
                answer += f"• [{c.get('source')}]: {c.get('content', '')[:150]}...\n"

        return answer