#!/usr/bin/env python3
"""
Gemini-powered Chatbot Service for Medical Questionnaire and Analysis
Provides intelligent conversational interface with dynamic system prompts based on patient data.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class ChatMessage(BaseModel):
    """Chat message model"""
    id: str
    type: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime

class PatientContext(BaseModel):
    """Patient context for system prompt generation"""
    demographics: Optional[Dict] = None
    symptoms: Optional[Dict] = None
    medical_history: Optional[Dict] = None
    scan_results: Optional[Dict] = None
    current_question_index: int = 0
    questionnaire_complete: bool = False

class QuestionnaireFormData(BaseModel):
    """Form-based questionnaire data"""
    demographics: Dict
    symptoms: Dict
    medical_history: Dict

class GeminiChatbotService:
    """Gemini-powered chatbot service for medical questionnaire"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini REST API
        self.model_name = 'gemini-2.0-flash'  # Use Gemini 2.0 Flash
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        # Questionnaire structure
        self.questions = [
            {
                "id": "age",
                "text": "What is your age?",
                "type": "number",
                "field": "demographics.age",
                "validation": {"min": 18, "max": 100}
            },
            {
                "id": "sex",
                "text": "What is your biological sex?",
                "type": "choice",
                "field": "demographics.sex",
                "options": ["Female", "Male"]
            },
            {
                "id": "ethnicity",
                "text": "What is your ethnicity? (Optional)",
                "type": "text",
                "field": "demographics.ethnicity",
                "optional": True
            },
            {
                "id": "heavy_periods",
                "text": "Do you experience heavy menstrual bleeding?",
                "type": "choice",
                "field": "symptoms.heavy_bleeding",
                "options": ["Yes", "No"],
                "condition": lambda ctx: ctx.get("demographics", {}).get("sex") == "Female"
            },
            {
                "id": "pelvic_pain",
                "text": "Do you experience pelvic pain or pressure?",
                "type": "choice",
                "field": "symptoms.pelvic_pain",
                "options": ["Yes", "No"]
            },
            {
                "id": "family_history",
                "text": "Do you have a family history of uterine fibroids?",
                "type": "choice",
                "field": "medical_history.family_history_fibroids",
                "options": ["Yes", "No", "Not sure"]
            }
        ]
    
    def _build_system_prompt(self, patient_context: PatientContext, scan_results: Optional[Dict] = None) -> str:
        """Build dynamic system prompt based on patient context and scan results"""
        
        base_prompt = """You are a compassionate and knowledgeable medical AI assistant specializing in women's health and uterine fibroids. Your role is to:

1. Guide patients through a health questionnaire with empathy and professionalism
2. Provide educational information about uterine fibroids and related conditions
3. Explain scan results in understandable terms
4. Offer appropriate recommendations while emphasizing the importance of professional medical consultation

IMPORTANT GUIDELINES:
- Always maintain a warm, supportive, and professional tone
- Explain medical terms in simple language
- Never provide definitive diagnoses - always recommend consulting healthcare providers
- Be sensitive to patient concerns and emotions
- Provide accurate, evidence-based information
- Respect patient privacy and confidentiality

"""
        
        # Add scan results context if available
        if scan_results:
            if scan_results.get('fibroidDetected'):
                fibroid_count = scan_results.get('fibroidCount', 0)
                confidence = scan_results.get('confidence', 0)
                
                base_prompt += f"""
SCAN RESULTS CONTEXT:
- Fibroids detected: {fibroid_count} fibroid(s) found
- AI confidence level: {confidence:.1%}
- This information should be communicated sensitively and with appropriate context
"""
            else:
                base_prompt += """
SCAN RESULTS CONTEXT:
- No fibroids detected in the current scan
- This is generally positive news, but symptoms should still be discussed
"""
        
        # Add patient context if available
        if patient_context.demographics:
            age = patient_context.demographics.get('age')
            sex = patient_context.demographics.get('sex')
            if age and sex:
                base_prompt += f"""
PATIENT CONTEXT:
- Age: {age} years old
- Sex: {sex}
"""
        
        # Add current questionnaire status
        current_q = patient_context.current_question_index
        total_q = len(self.questions)
        
        base_prompt += f"""
QUESTIONNAIRE STATUS:
- Current question: {current_q + 1} of {total_q}
- Questions completed: {current_q}/{total_q}

Your current task is to ask the next question in the questionnaire while being conversational and supportive. After each answer, acknowledge the response appropriately before moving to the next question.
"""
        
        return base_prompt

    def _build_dynamic_system_prompt(self, form_data: QuestionnaireFormData, scan_results: Optional[Dict] = None) -> str:
        """Build dynamic system prompt based on form data and scan results"""

        # Extract patient information
        age = form_data.demographics.get('age', 'unknown age')
        sex = form_data.demographics.get('sex', 'unknown sex')
        ethnicity = form_data.demographics.get('ethnicity', 'unspecified ethnicity')

        heavy_bleeding = form_data.symptoms.get('heavy_bleeding', 'No')
        pelvic_pain = form_data.symptoms.get('pelvic_pain', 'No')
        other_symptoms = form_data.symptoms.get('other_symptoms', '')

        family_history = form_data.medical_history.get('family_history_fibroids', 'No')
        previous_pregnancies = form_data.medical_history.get('previous_pregnancies', 'Not specified')
        hormonal_medications = form_data.medical_history.get('hormonal_medications', 'Not specified')

        # Build fibroid information from scan results
        fibroid_info = ""
        if scan_results:
            if scan_results.get('fibroidDetected'):
                fibroid_count = scan_results.get('fibroidCount', 0)
                confidence = scan_results.get('confidence', 0)

                # Determine size description based on confidence or other metrics
                if confidence > 0.8:
                    size_desc = "clearly visible"
                elif confidence > 0.6:
                    size_desc = "moderately sized"
                else:
                    size_desc = "small"

                if fibroid_count == 1:
                    fibroid_info = f"has 1 fibroid which is {size_desc}"
                else:
                    fibroid_info = f"has {fibroid_count} fibroids which are {size_desc}"
            else:
                fibroid_info = "shows no detectable fibroids in the current scan"

        # Build symptom description
        symptom_list = []
        if heavy_bleeding == 'Yes':
            symptom_list.append("heavy menstrual bleeding")
        if pelvic_pain == 'Yes':
            symptom_list.append("pelvic pain")
        if other_symptoms:
            symptom_list.append(f"additional symptoms: {other_symptoms}")

        symptom_desc = ", ".join(symptom_list) if symptom_list else "no specific symptoms reported"

        # Build the dynamic system prompt
        system_prompt = f"""You are a compassionate and knowledgeable medical AI assistant specializing in women's health and uterine fibroids.

PATIENT PROFILE:
You are helping educate a {age}-year-old {sex.lower()} patient of {ethnicity} origin who is concerned about uterine fibroid detection. The patient {fibroid_info}. The patient experiences {symptom_desc}.

ADDITIONAL CONTEXT:
- Family history of fibroids: {family_history}
- Previous pregnancies: {previous_pregnancies}
- Hormonal medications: {hormonal_medications}

YOUR ROLE:
1. Provide personalized education about uterine fibroids based on their specific situation
2. Address their concerns with empathy and understanding
3. Explain their scan results in context of their symptoms and medical history
4. Offer appropriate lifestyle and medical recommendations
5. Encourage appropriate medical follow-up

IMPORTANT GUIDELINES:
- Always maintain a warm, supportive, and professional tone
- Explain medical terms in simple language appropriate for their background
- Never provide definitive diagnoses - always recommend consulting healthcare providers
- Be sensitive to their concerns and emotions about their condition
- Provide accurate, evidence-based information tailored to their specific situation
- Respect patient privacy and confidentiality
- Consider their demographic background when providing culturally sensitive advice

The patient has completed a comprehensive questionnaire and is now ready to discuss their results and ask questions. Be ready to provide personalized guidance based on their specific profile and concerns."""

        return system_prompt

    async def start_form_based_conversation(self, form_data: QuestionnaireFormData, scan_results: Optional[Dict] = None) -> Dict:
        """Start a new conversation based on completed form data"""

        # Create patient context from form data
        patient_context = PatientContext(
            demographics=form_data.demographics,
            symptoms=form_data.symptoms,
            medical_history=form_data.medical_history,
            scan_results=scan_results,
            questionnaire_complete=True
        )

        # Build dynamic system prompt
        system_prompt = self._build_dynamic_system_prompt(form_data, scan_results)

        # Create personalized greeting based on their information
        age = form_data.demographics.get('age')
        fibroid_detected = scan_results and scan_results.get('fibroidDetected', False)

        if fibroid_detected:
            fibroid_count = scan_results.get('fibroidCount', 0)
            if fibroid_count == 1:
                initial_message = f"""Hello! I'm here to help you understand your scan results and provide personalized guidance about uterine fibroids.

I can see from your questionnaire that you're {age} years old and your ultrasound scan has detected 1 fibroid. I understand this news might feel overwhelming, but I want you to know that uterine fibroids are very common - affecting up to 80% of women by age 50.

I'm here to help you understand what this means for your health, answer any questions you might have, and provide guidance tailored specifically to your situation.

What would you like to know about your results or fibroids in general?"""
            else:
                initial_message = f"""Hello! I'm here to help you understand your scan results and provide personalized guidance about uterine fibroids.

I can see from your questionnaire that you're {age} years old and your ultrasound scan has detected {fibroid_count} fibroids. I understand this news might feel concerning, but I want you to know that uterine fibroids are very common - affecting up to 80% of women by age 50.

I'm here to help you understand what this means for your health, answer any questions you might have, and provide guidance tailored specifically to your situation.

What would you like to know about your results or fibroids in general?"""
        else:
            initial_message = f"""Hello! I'm here to help you understand your scan results and provide personalized health guidance.

Great news! Your ultrasound scan didn't detect any uterine fibroids. This is wonderful news and means your uterus appears healthy in this regard.

However, I noticed from your questionnaire that you may be experiencing some symptoms. I'm here to help you understand what might be causing these symptoms and provide guidance on maintaining your reproductive health.

What questions do you have about your results or your health in general?"""

        return {
            'message': initial_message,
            'patient_context': patient_context.dict(),
            'conversation_id': f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'system_prompt': system_prompt
        }

    def _get_next_question(self, patient_context: PatientContext) -> Optional[Dict]:
        """Get the next appropriate question based on patient context"""
        current_index = patient_context.current_question_index

        # Check if questionnaire is complete
        if current_index >= len(self.questions):
            return None

        question = self.questions[current_index].copy()  # Make a copy to avoid modifying original

        # Check if question has conditions
        if 'condition' in question:
            # Convert patient context to dict for condition evaluation
            context_dict = {
                'demographics': patient_context.demographics or {},
                'symptoms': patient_context.symptoms or {},
                'medical_history': patient_context.medical_history or {}
            }

            if not question['condition'](context_dict):
                # Skip this question and get the next one
                patient_context.current_question_index += 1
                return self._get_next_question(patient_context)

        # Remove the condition function before returning (not serializable)
        if 'condition' in question:
            del question['condition']

        return question
    
    async def start_conversation(self, scan_results: Optional[Dict] = None) -> Dict:
        """Start a new conversation with initial greeting"""
        patient_context = PatientContext()
        
        system_prompt = self._build_system_prompt(patient_context, scan_results)
        
        # Create initial greeting based on scan results
        if scan_results and scan_results.get('fibroidDetected'):
            initial_message = """Hello! I'm here to help you understand your scan results and gather some important health information. 

I can see from your ultrasound scan that we have some findings to discuss. Don't worry - I'm here to guide you through this process step by step and help you understand what this means for your health.

Let's start by gathering some basic information about you. This will help me provide you with the most relevant and personalized guidance."""
        else:
            initial_message = """Hello! I'm here to help you understand your scan results and gather some health information.

Good news - your scan didn't show any signs of uterine fibroids! However, I'd still like to ask you a few questions about your health to provide you with comprehensive guidance and recommendations.

Let's start with some basic information."""
        
        # Get first question
        first_question = self._get_next_question(patient_context)
        if first_question:
            initial_message += f"\n\n{first_question['text']}"
        
        return {
            'message': initial_message,
            'patient_context': patient_context.dict(),
            'current_question': first_question,
            'conversation_id': f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    async def process_user_response(
        self,
        user_message: str,
        patient_context: PatientContext,
        conversation_history: List[ChatMessage],
        scan_results: Optional[Dict] = None
    ) -> Dict:
        """Process user response and generate next message"""

        try:
            # Check if questionnaire is already complete (form-based approach)
            if patient_context.questionnaire_complete:
                return await self._process_form_based_conversation(
                    user_message, patient_context, conversation_history, scan_results
                )

            # Legacy questionnaire-based approach (for backward compatibility)
            # Get current question
            current_question = self._get_next_question(patient_context)

            if not current_question:
                # Questionnaire complete
                return await self._complete_questionnaire(patient_context, scan_results)

            # Validate and process the response
            processed_response = self._process_response(user_message, current_question)

            if 'error' in processed_response:
                return {
                    'message': processed_response['error'] + " Please try again.",
                    'patient_context': patient_context.dict(),
                    'current_question': current_question,
                    'error': True
                }

            # Update patient context
            self._update_patient_context(patient_context, current_question, processed_response['value'])

            # Move to next question
            patient_context.current_question_index += 1
            next_question = self._get_next_question(patient_context)

            # Generate response using Gemini
            system_prompt = self._build_system_prompt(patient_context, scan_results)

            # Build conversation context for Gemini
            conversation_context = self._build_conversation_context(
                conversation_history,
                user_message,
                current_question,
                next_question,
                patient_context
            )

            # Get response from Gemini
            response = await self._get_gemini_response(system_prompt, conversation_context)

            return {
                'message': response,
                'patient_context': patient_context.dict(),
                'current_question': next_question,
                'questionnaire_complete': next_question is None
            }

        except Exception as e:
            print(f"Error processing user response: {e}")
            return {
                'message': "I apologize, but I encountered an issue processing your response. Could you please try again?",
                'patient_context': patient_context.dict(),
                'current_question': None,
                'error': True
            }

    async def _process_form_based_conversation(
        self,
        user_message: str,
        patient_context: PatientContext,
        conversation_history: List[ChatMessage],
        scan_results: Optional[Dict] = None
    ) -> Dict:
        """Process conversation for form-based chatbot (questionnaire already complete)"""

        try:
            # Build dynamic system prompt based on patient context
            system_prompt = self._build_dynamic_system_prompt_from_context(patient_context, scan_results)

            # Build conversation context for educational chat
            conversation_context = self._build_educational_conversation_context(
                conversation_history, user_message, patient_context
            )

            # Get response from Gemini (or fallback)
            response = await self._get_gemini_response(system_prompt, conversation_context)

            return {
                'message': response,
                'patient_context': patient_context.dict(),
                'questionnaire_complete': True,
                'current_question': None
            }

        except Exception as e:
            print(f"Error processing form-based conversation: {e}")
            return {
                'message': "I apologize, but I encountered an issue processing your message. Could you please try again?",
                'patient_context': patient_context.dict(),
                'questionnaire_complete': True,
                'current_question': None,
                'error': True
            }

    def _build_dynamic_system_prompt_from_context(self, patient_context: PatientContext, scan_results: Optional[Dict] = None) -> str:
        """Build dynamic system prompt from existing patient context"""

        # Extract patient information from context
        demographics = patient_context.demographics or {}
        symptoms = patient_context.symptoms or {}
        medical_history = patient_context.medical_history or {}

        age = demographics.get('age', 'unknown age')
        sex = demographics.get('sex', 'unknown sex')
        ethnicity = demographics.get('ethnicity', 'unspecified ethnicity')

        heavy_bleeding = symptoms.get('heavy_bleeding', 'No')
        pelvic_pain = symptoms.get('pelvic_pain', 'No')
        other_symptoms = symptoms.get('other_symptoms', '')

        family_history = medical_history.get('family_history_fibroids', 'No')
        previous_pregnancies = medical_history.get('previous_pregnancies', 'Not specified')
        hormonal_medications = medical_history.get('hormonal_medications', 'Not specified')

        # Build fibroid information from scan results
        fibroid_info = ""
        if scan_results:
            if scan_results.get('fibroidDetected'):
                fibroid_count = scan_results.get('fibroidCount', 0)
                confidence = scan_results.get('confidence', 0)

                # Determine size description based on confidence or other metrics
                if confidence > 0.8:
                    size_desc = "clearly visible"
                elif confidence > 0.6:
                    size_desc = "moderately sized"
                else:
                    size_desc = "small"

                if fibroid_count == 1:
                    fibroid_info = f"has 1 fibroid which is {size_desc}"
                else:
                    fibroid_info = f"has {fibroid_count} fibroids which are {size_desc}"
            else:
                fibroid_info = "shows no detectable fibroids in the current scan"

        # Build symptom description
        symptom_list = []
        if heavy_bleeding == 'Yes':
            symptom_list.append("heavy menstrual bleeding")
        if pelvic_pain == 'Yes':
            symptom_list.append("pelvic pain")
        if other_symptoms:
            symptom_list.append(f"additional symptoms: {other_symptoms}")

        symptom_desc = ", ".join(symptom_list) if symptom_list else "no specific symptoms reported"

        # Build the dynamic system prompt
        system_prompt = f"""You are a compassionate and knowledgeable medical AI assistant specializing in women's health and uterine fibroids.

PATIENT PROFILE:
You are helping educate a {age}-year-old {sex.lower()} patient of {ethnicity} origin who is concerned about uterine fibroid detection. The patient {fibroid_info}. The patient experiences {symptom_desc}.

ADDITIONAL CONTEXT:
- Family history of fibroids: {family_history}
- Previous pregnancies: {previous_pregnancies}
- Hormonal medications: {hormonal_medications}

YOUR ROLE:
1. Provide personalized education about uterine fibroids based on their specific situation
2. Address their concerns with empathy and understanding
3. Explain their scan results in context of their symptoms and medical history
4. Offer appropriate lifestyle and medical recommendations
5. Encourage appropriate medical follow-up
6. Answer questions about their diagnosis, treatment options, and health management

IMPORTANT GUIDELINES:
- Always maintain a warm, supportive, and professional tone
- Explain medical terms in simple language appropriate for their background
- Never provide definitive diagnoses - always recommend consulting healthcare providers
- Be sensitive to their concerns and emotions about their condition
- Provide accurate, evidence-based information tailored to their specific situation
- Respect patient privacy and confidentiality
- Consider their demographic background when providing culturally sensitive advice

The patient has completed a comprehensive questionnaire and is now ready to discuss their results and ask questions. Be ready to provide personalized guidance based on their specific profile and concerns."""

        return system_prompt

    def _build_educational_conversation_context(
        self,
        conversation_history: List[ChatMessage],
        current_user_message: str,
        patient_context: PatientContext
    ) -> str:
        """Build conversation context for educational chat"""

        context = "CONVERSATION HISTORY:\n"

        # Add recent conversation history (last 10 messages)
        recent_messages = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        for msg in recent_messages:
            role = "User" if msg.type == "user" else "Assistant"
            context += f"{role}: {msg.content}\n"

        # Add current user message
        context += f"User: {current_user_message}\n\n"

        context += """INSTRUCTIONS:
Please respond to the user's question or message in a helpful, educational, and supportive manner.
Use the patient profile information to provide personalized guidance.
Be conversational and empathetic while maintaining medical accuracy.
If the user asks about their diagnosis, explain their scan results in the context of their symptoms and profile.
Always encourage them to follow up with healthcare providers for medical decisions."""

        return context

    def _process_response(self, user_message: str, question: Dict) -> Dict:
        """Process and validate user response based on question type"""
        user_message = user_message.strip()
        
        if question['type'] == 'number':
            try:
                value = int(user_message)
                validation = question.get('validation', {})
                
                if 'min' in validation and value < validation['min']:
                    return {'error': f"Please enter a number greater than or equal to {validation['min']}"}
                if 'max' in validation and value > validation['max']:
                    return {'error': f"Please enter a number less than or equal to {validation['max']}"}
                
                return {'value': value}
            except ValueError:
                return {'error': "Please enter a valid number"}
        
        elif question['type'] == 'choice':
            options = question.get('options', [])
            # Check if user message matches any option (case insensitive)
            for option in options:
                if user_message.lower() == option.lower():
                    return {'value': option}
            
            return {'error': f"Please choose one of: {', '.join(options)}"}
        
        elif question['type'] == 'text':
            if not user_message and not question.get('optional', False):
                return {'error': "Please provide an answer"}
            return {'value': user_message if user_message else None}
        
        return {'value': user_message}

    def _update_patient_context(self, patient_context: PatientContext, question: Dict, value: Any):
        """Update patient context with new response"""
        field_path = question['field'].split('.')

        # Initialize nested dictionaries if needed
        if field_path[0] == 'demographics':
            if not patient_context.demographics:
                patient_context.demographics = {}
            target = patient_context.demographics
        elif field_path[0] == 'symptoms':
            if not patient_context.symptoms:
                patient_context.symptoms = {}
            target = patient_context.symptoms
        elif field_path[0] == 'medical_history':
            if not patient_context.medical_history:
                patient_context.medical_history = {}
            target = patient_context.medical_history
        else:
            return

        # Set the value
        if len(field_path) > 1:
            target[field_path[1]] = value
        else:
            target[field_path[0]] = value

    def _build_conversation_context(
        self,
        conversation_history: List[ChatMessage],
        current_user_message: str,
        current_question: Dict,
        next_question: Optional[Dict],
        patient_context: PatientContext
    ) -> str:
        """Build conversation context for Gemini"""

        context = "CONVERSATION HISTORY:\n"

        # Add recent conversation history (last 6 messages)
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        for msg in recent_messages:
            role = "User" if msg.type == "user" else "Assistant"
            context += f"{role}: {msg.content}\n"

        # Add current user message
        context += f"User: {current_user_message}\n\n"

        # Add current question context
        context += f"CURRENT QUESTION ANSWERED: {current_question['text']}\n"
        context += f"USER'S ANSWER: {current_user_message}\n\n"

        # Add next question if available
        if next_question:
            context += f"NEXT QUESTION TO ASK: {next_question['text']}\n"
            if next_question.get('options'):
                context += f"Available options: {', '.join(next_question['options'])}\n"
        else:
            context += "QUESTIONNAIRE COMPLETE - Provide summary and next steps\n"

        context += "\nPlease respond naturally and conversationally, acknowledging their answer and then asking the next question (if any). Be empathetic and supportive."

        return context

    async def _get_gemini_response(self, system_prompt: str, conversation_context: str) -> str:
        """Get response from Gemini API using REST API"""
        try:
            # Combine system prompt and conversation context
            full_prompt = f"{system_prompt}\n\n{conversation_context}"

            # Prepare the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ]
            }

            headers = {
                'Content-Type': 'application/json'
            }

            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Extract the text from the response
                        if (data.get('candidates') and
                            len(data['candidates']) > 0 and
                            data['candidates'][0].get('content') and
                            data['candidates'][0]['content'].get('parts') and
                            len(data['candidates'][0]['content']['parts']) > 0):

                            text = data['candidates'][0]['content']['parts'][0].get('text', '')
                            if text:
                                return text.strip()

                        print("Gemini API returned empty response, using fallback")
                        return self._generate_intelligent_fallback(system_prompt, conversation_context)
                    else:
                        error_text = await response.text()
                        print(f"Gemini API error {response.status}: {error_text}")
                        return self._generate_intelligent_fallback(system_prompt, conversation_context)

        except Exception as e:
            print(f"Error getting Gemini response: {e}")
            print("Using intelligent fallback response")
            return self._generate_intelligent_fallback(system_prompt, conversation_context)

    def _generate_intelligent_fallback(self, system_prompt: str, conversation_context: str) -> str:
        """Generate intelligent fallback responses based on context"""

        try:
            # Check if this is a form-based conversation (educational chat)
            if "INSTRUCTIONS:" in conversation_context and "educational" in conversation_context.lower():
                response = self._generate_educational_fallback(system_prompt, conversation_context)
                return response if response else "I'm here to help you understand your health results. What would you like to know?"

            # Legacy questionnaire-based fallback
            response = self._generate_questionnaire_fallback(conversation_context)
            return response if response else "Thank you for your response. How can I help you today?"

        except Exception as e:
            print(f"Error in fallback generation: {e}")
            return "I'm here to help you with your health questions. What would you like to know?"

    def _generate_educational_fallback(self, system_prompt: str, conversation_context: str) -> str:
        """Generate educational responses for form-based conversations"""

        # Extract the user's latest message
        lines = conversation_context.split('\n')
        user_message = ""

        for line in lines:
            if line.startswith("User: "):
                user_message = line.replace("User: ", "").strip().lower()

        # Extract patient information from system prompt
        fibroid_info = ""
        patient_age = ""
        symptoms = ""

        if "has 1 fibroid" in system_prompt:
            fibroid_info = "1 fibroid"
        elif "has 2 fibroids" in system_prompt:
            fibroid_info = "2 fibroids"
        elif "has 3 fibroids" in system_prompt:
            fibroid_info = "3 fibroids"
        elif "shows no detectable fibroids" in system_prompt:
            fibroid_info = "no fibroids detected"

        if "-year-old" in system_prompt:
            # Extract age from system prompt
            import re
            age_match = re.search(r'(\d+)-year-old', system_prompt)
            if age_match:
                patient_age = age_match.group(1)

        # Generate contextual responses based on user questions
        if any(word in user_message for word in ["results", "result", "mean", "diagnosis", "what"]):
            if "no fibroids detected" in fibroid_info:
                return f"""Based on your ultrasound scan, I have good news to share! Your scan shows no detectable uterine fibroids, which means your uterus appears healthy in this regard.

This is a positive result that indicates:
• No abnormal growths were found in your uterine tissue
• Your uterine structure appears normal on the imaging
• The symptoms you may be experiencing are likely due to other causes

However, I noticed from your questionnaire that you may be experiencing some symptoms. Even without fibroids, there can be other reasons for symptoms like heavy bleeding or pelvic discomfort, such as:
• Hormonal imbalances
• Other benign conditions
• Normal variations in menstrual cycles

I'd recommend discussing these results and any ongoing symptoms with your healthcare provider to ensure comprehensive care and address any concerns you may have."""

            elif fibroid_info:
                return f"""Based on your ultrasound scan analysis, your results show that you have {fibroid_info} detected in your uterus.

Here's what this means:
• Uterine fibroids are non-cancerous growths in the uterine muscle
• They are very common - affecting up to 80% of women by age 50
• Your scan detected {fibroid_info}, which our AI analysis has identified

Understanding your specific situation:
• Given your age ({patient_age} years old) and the symptoms you've reported, this finding helps explain what you may be experiencing
• Fibroids can cause symptoms like heavy menstrual bleeding, pelvic pressure, or pain
• The size and location of fibroids determine what symptoms they might cause

Next steps:
• These results should be reviewed with your healthcare provider
• They can discuss treatment options if your symptoms are bothersome
• Many fibroids can be managed effectively with various treatment approaches
• Regular monitoring may be all that's needed if symptoms are mild

Remember, having fibroids is very common and there are many effective treatment options available."""

        elif any(word in user_message for word in ["treatment", "treat", "options", "what can i do"]):
            return """There are several treatment options available for managing uterine fibroids, depending on your symptoms and individual situation:

**Non-surgical options:**
• Hormonal medications to reduce heavy bleeding
• Pain management strategies
• Iron supplements if you have anemia from heavy bleeding
• Lifestyle modifications (diet, exercise, stress management)

**Minimally invasive procedures:**
• Uterine artery embolization (UAE)
• MRI-guided focused ultrasound
• Laparoscopic procedures

**Surgical options:**
• Myomectomy (removal of fibroids while preserving the uterus)
• Hysterectomy (in severe cases)

**Important:** The best treatment approach depends on:
• Size and location of your fibroids
• Severity of your symptoms
• Your age and family planning goals
• Your overall health

I strongly recommend discussing these options with your healthcare provider, who can evaluate your specific situation and recommend the most appropriate treatment plan for you."""

        elif any(word in user_message for word in ["symptoms", "feel", "experience", "pain", "bleeding"]):
            return """Uterine fibroids can cause various symptoms, though some women have no symptoms at all. Common symptoms include:

**Menstrual-related symptoms:**
• Heavy menstrual bleeding
• Prolonged periods (lasting more than 7 days)
• Bleeding between periods
• Severe menstrual cramps

**Pressure-related symptoms:**
• Pelvic pain or pressure
• Frequent urination
• Difficulty emptying the bladder
• Constipation
• Backache or leg pain

**Other symptoms:**
• Enlarged abdomen
• Pain during intercourse
• Fatigue (often due to anemia from heavy bleeding)

Based on your questionnaire responses, it sounds like you may be experiencing some of these symptoms. The good news is that these symptoms can often be effectively managed with appropriate treatment.

If your symptoms are significantly impacting your quality of life, I encourage you to discuss them with your healthcare provider. They can help determine the best approach to manage your symptoms and improve your comfort."""

        else:
            return """I'm here to help you understand your scan results and provide guidance about uterine fibroids. Feel free to ask me about:

• What your specific results mean
• Symptoms you might be experiencing
• Treatment options available
• Lifestyle recommendations
• When to see a healthcare provider
• Any other questions about fibroids or women's health

What would you like to know more about?"""

    def _generate_questionnaire_fallback(self, conversation_context: str) -> str:
        """Generate fallback responses for legacy questionnaire-based conversations"""

        # Extract the user's latest message from conversation context
        lines = conversation_context.split('\n')
        user_message = ""
        current_question = ""

        for line in lines:
            if line.startswith("User: "):
                user_message = line.replace("User: ", "").strip()
            elif line.startswith("CURRENT QUESTION ANSWERED: "):
                current_question = line.replace("CURRENT QUESTION ANSWERED: ", "").strip()
            elif line.startswith("NEXT QUESTION TO ASK: "):
                next_question = line.replace("NEXT QUESTION TO ASK: ", "").strip()

                # Generate contextual response based on the question type
                if "age" in current_question.lower():
                    age = user_message
                    return f"Thank you for sharing that you're {age} years old. That's helpful information for understanding your health profile. {next_question}"

                elif "biological sex" in current_question.lower() or "sex" in current_question.lower():
                    sex = user_message
                    if sex.lower() == "female":
                        return f"Thank you for confirming you're female. This helps me ask more relevant questions about your reproductive health. {next_question}"
                    else:
                        return f"Thank you for that information. {next_question}"

                elif "ethnicity" in current_question.lower():
                    if user_message.lower() in ["skip", "prefer not to say", ""]:
                        return f"That's perfectly fine - ethnicity information is optional. {next_question}"
                    else:
                        return f"Thank you for sharing that information. {next_question}"

                elif "heavy" in current_question.lower() and "bleeding" in current_question.lower():
                    if user_message.lower() == "yes":
                        return f"I understand you experience heavy menstrual bleeding. This is an important symptom that we'll consider in your overall health assessment. {next_question}"
                    else:
                        return f"Thank you for that information. {next_question}"

                elif "pelvic pain" in current_question.lower():
                    if user_message.lower() == "yes":
                        return f"I note that you experience pelvic pain or pressure. This symptom, along with your scan results, will help us provide better guidance. {next_question}"
                    else:
                        return f"Thank you for letting me know. {next_question}"

                elif "family history" in current_question.lower():
                    if user_message.lower() == "yes":
                        return f"Thank you for sharing your family history. Having relatives with uterine fibroids can be a risk factor. {next_question}"
                    elif user_message.lower() == "not sure":
                        return f"That's understandable - family medical history isn't always well-known. {next_question}"
                    else:
                        return f"Thank you for that information. {next_question}"

                else:
                    return f"Thank you for your response. {next_question}"

        # If we can't parse the context, provide a generic but helpful response
        return "Thank you for your response. Let me ask you the next question to better understand your health profile."

    async def _complete_questionnaire(self, patient_context: PatientContext, scan_results: Optional[Dict] = None) -> Dict:
        """Complete the questionnaire and provide summary"""
        patient_context.questionnaire_complete = True

        # Build comprehensive summary prompt
        summary_prompt = self._build_system_prompt(patient_context, scan_results)
        summary_prompt += """
TASK: The questionnaire is now complete. Please provide a comprehensive, empathetic summary that includes:

1. Acknowledgment of their participation
2. Summary of key health information gathered
3. Explanation of scan results in context of their symptoms/history
4. Personalized recommendations based on their profile
5. Clear next steps for medical care
6. Reassurance and support

Be warm, professional, and ensure they understand the importance of following up with healthcare providers.
"""

        # Build patient summary for context
        patient_summary = self._build_patient_summary(patient_context)
        summary_prompt += f"\n\nPATIENT SUMMARY:\n{patient_summary}"

        try:
            response = await self._get_gemini_response(summary_prompt, "Please provide the final summary and recommendations.")

            return {
                'message': response,
                'patient_context': patient_context.dict(),
                'questionnaire_complete': True,
                'summary': self._generate_structured_summary(patient_context, scan_results)
            }

        except Exception as e:
            print(f"Error generating questionnaire summary: {e}")

            return {
                'message': "Thank you for completing the health questionnaire. I apologize, but I encountered an issue generating your summary. Please proceed to discuss your results.",
                'patient_context': patient_context.dict(),
                'questionnaire_complete': True,
                'summary': self._generate_structured_summary(patient_context, scan_results)
            }

    def _build_patient_summary(self, patient_context: PatientContext) -> str:
        """Build a summary of patient information for context"""
        summary = []

        if patient_context.demographics:
            demo = patient_context.demographics
            if demo.get('age'):
                summary.append(f"Age: {demo['age']}")
            if demo.get('sex'):
                summary.append(f"Sex: {demo['sex']}")
            if demo.get('ethnicity'):
                summary.append(f"Ethnicity: {demo['ethnicity']}")

        if patient_context.symptoms:
            symptoms = patient_context.symptoms
            symptom_list = []
            if symptoms.get('heavy_bleeding') == 'Yes':
                symptom_list.append("heavy menstrual bleeding")
            if symptoms.get('pelvic_pain') == 'Yes':
                symptom_list.append("pelvic pain")

            if symptom_list:
                summary.append(f"Reported symptoms: {', '.join(symptom_list)}")

        if patient_context.medical_history:
            history = patient_context.medical_history
            if history.get('family_history_fibroids'):
                summary.append(f"Family history of fibroids: {history['family_history_fibroids']}")

        return "\n".join(summary) if summary else "Limited information provided"

    def _generate_structured_summary(self, patient_context: PatientContext, scan_results: Optional[Dict] = None) -> Dict:
        """Generate structured summary for API response"""
        summary = {
            'patient_profile': {
                'demographics': patient_context.demographics or {},
                'symptoms': patient_context.symptoms or {},
                'medical_history': patient_context.medical_history or {}
            },
            'scan_results': scan_results or {},
            'recommendations': [],
            'follow_up_required': True,
            'risk_factors': []
        }

        # Analyze risk factors
        if patient_context.medical_history and patient_context.medical_history.get('family_history_fibroids') == 'Yes':
            summary['risk_factors'].append('Family history of uterine fibroids')

        if patient_context.symptoms:
            if patient_context.symptoms.get('heavy_bleeding') == 'Yes':
                summary['risk_factors'].append('Heavy menstrual bleeding')
            if patient_context.symptoms.get('pelvic_pain') == 'Yes':
                summary['risk_factors'].append('Pelvic pain or pressure')

        # Generate recommendations based on findings
        if scan_results and scan_results.get('fibroidDetected'):
            summary['recommendations'].extend([
                'Schedule consultation with gynecologist',
                'Discuss treatment options based on symptoms',
                'Consider follow-up imaging as recommended by physician'
            ])
        else:
            summary['recommendations'].extend([
                'Continue routine gynecological care',
                'Monitor symptoms and report changes to healthcare provider',
                'Maintain healthy lifestyle habits'
            ])

        return summary



# Global instance
chatbot_service = None

def get_chatbot_service() -> GeminiChatbotService:
    """Get or create chatbot service instance"""
    global chatbot_service
    if chatbot_service is None:
        chatbot_service = GeminiChatbotService()
    return chatbot_service
