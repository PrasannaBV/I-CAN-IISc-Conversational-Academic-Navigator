import re
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import os

class ContextAnalyzer:
    def __init__(self, config_path: str = "data/academic_calendar.json"):
        self.config_path = config_path
        self.academic_calendar = self._load_academic_calendar()
        
        # Intent classification keywords
        self.intent_keywords = {
            "admission": ["admission", "apply", "application", "eligibility", "qualify", "entrance", "selection"],
            "fees": ["fee", "cost", "payment", "scholarship", "financial", "tuition", "money"],
            "courses": ["course", "subject", "curriculum", "syllabus", "credits", "elective"],
            "deadlines": ["deadline", "last date", "due date", "when", "timeline", "schedule"],
            "documents": ["document", "certificate", "transcript", "proof", "upload", "submit"],
            "registration": ["registration", "enroll", "register", "seat", "confirm"],
            "academic_calendar": ["semester", "exam", "holiday", "break", "calendar", "timetable"],
            "thesis": ["thesis", "project", "research", "dissertation", "guide", "supervisor"],
            "graduation": ["graduation", "degree", "completion", "requirements", "convocation"]
        }
        
        # Vague query patterns that need clarification
        self.vague_patterns = [
            r"what about.*",
            r"tell me about.*",
            r"how.*work",
            r"^(fees?|admission|course)$",
            r"need help",
            r"don't understand",
            r"confused about"
        ]
        
        # Program/stream keywords
        self.program_keywords = {
            "mtech": ["mtech", "m.tech", "masters in technology"],
            "data_science": ["data science", "ds", "analytics"],
            "ai": ["artificial intelligence", "ai", "machine learning", "ml"],
            "ece": ["electronics", "communication", "ece"],
            "cse": ["computer science", "cse", "computing"]
        }

    def _load_academic_calendar(self) -> Dict:
        """Load academic calendar data"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load academic calendar: {e}")
        
        # Default fallback calendar
        return {
            "current_semester": "Spring 2024",
            "academic_year": "2024-25",
            "important_dates": {
                "admission_deadline": "2024-04-01",
                "class_start": "2024-08-01",
                "registration_period": "2024-07-15 to 2024-07-30",
                "mid_sem_exams": "2024-09-15 to 2024-09-25",
                "end_sem_exams": "2024-12-01 to 2024-12-15"
            },
            "current_phase": "admission_period"
        }

    def analyze_query_intent(self, query: str) -> List[str]:
        """
        Classify the intent of the user query
        Returns: List of detected intents (can be multiple)
        """
        query_lower = query.lower()
        detected_intents = []
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Default to general if no specific intent detected
        if not detected_intents:
            detected_intents.append("general")
            
        return detected_intents

    def identify_program_stream(self, query: str) -> Optional[str]:
        """
        Identify which program/stream the user is asking about
        """
        query_lower = query.lower()
        
        for program, keywords in self.program_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return program
        
        return None

    def identify_clarification_needs(self, query: str) -> Dict[str, any]:
        """
        Detect if query is vague and needs clarification
        Returns: Dict with clarification info
        """
        query_lower = query.lower().strip()
        
        clarification_needed = False
        clarification_type = None
        suggested_questions = []
        
        # Check for vague patterns
        for pattern in self.vague_patterns:
            if re.search(pattern, query_lower):
                clarification_needed = True
                break
        
        # Check for single word queries
        if len(query_lower.split()) <= 2 and query_lower in ["fees", "admission", "courses", "deadline"]:
            clarification_needed = True
            clarification_type = "too_generic"
        
        # Generate appropriate clarification questions based on intent
        if clarification_needed:
            intents = self.analyze_query_intent(query)
            suggested_questions = self._generate_clarification_questions(intents[0] if intents else "general")
        
        return {
            "needs_clarification": clarification_needed,
            "type": clarification_type,
            "suggested_questions": suggested_questions,
            "confidence_score": 0.3 if clarification_needed else 0.8
        }

    def _generate_clarification_questions(self, intent: str) -> List[str]:
        """Generate clarifying questions based on intent"""
        questions = {
            "admission": [
                "Are you asking about eligibility criteria or application process?",
                "Which program are you interested in? (M.Tech Data Science, AI, etc.)",
                "Do you need information about deadlines or required documents?"
            ],
            "fees": [
                "Do you want to know about tuition fees or other charges?",
                "Are you looking for scholarship information?",
                "Which program's fee structure do you need?"
            ],
            "courses": [
                "Are you looking for course curriculum or specific subject details?",
                "Do you want core courses or elective options?",
                "Which semester's courses are you interested in?"
            ],
            "deadlines": [
                "Which deadline are you asking about? (Application, registration, fee payment)",
                "For which academic year or semester?",
                "Do you need current deadlines or future ones?"
            ]
        }
        
        return questions.get(intent, [
            "Could you be more specific about what information you need?",
            "Are you asking about admission, courses, or something else?",
            "Which aspect would you like to know more about?"
        ])

    def get_academic_context(self) -> Dict[str, any]:
        """
        Get current academic context (semester, phase, important upcoming dates)
        """
        current_date = datetime.now().date()
        
        # Determine current academic phase
        phase = self._determine_current_phase(current_date)
        
        # Get upcoming important dates
        upcoming_dates = self._get_upcoming_dates(current_date)
        
        return {
            "current_semester": self.academic_calendar.get("current_semester"),
            "academic_year": self.academic_calendar.get("academic_year"),
            "current_phase": phase,
            "current_date": current_date.isoformat(),
            "upcoming_dates": upcoming_dates,
            "is_admission_period": phase == "admission_period",
            "is_academic_session": phase == "academic_session"
        }

    def _determine_current_phase(self, current_date: date) -> str:
        """Determine current academic phase based on date"""
        try:
            important_dates = self.academic_calendar.get("important_dates", {})
            
            # Simple phase determination logic
            admission_deadline = datetime.strptime(important_dates.get("admission_deadline", "2024-04-01"), "%Y-%m-%d").date()
            class_start = datetime.strptime(important_dates.get("class_start", "2024-08-01"), "%Y-%m-%d").date()
            
            if current_date < admission_deadline:
                return "admission_period"
            elif current_date < class_start:
                return "pre_academic_period"
            else:
                return "academic_session"
                
        except Exception:
            return "academic_session"

    def _get_upcoming_dates(self, current_date: date, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming important dates within specified days"""
        upcoming = []
        
        try:
            important_dates = self.academic_calendar.get("important_dates", {})
            
            for event, date_str in important_dates.items():
                try:
                    event_date = datetime.strptime(date_str.split(" to ")[0], "%Y-%m-%d").date()
                    days_until = (event_date - current_date).days
                    
                    if 0 <= days_until <= days_ahead:
                        upcoming.append({
                            "event": event.replace("_", " ").title(),
                            "date": event_date.isoformat(),
                            "days_until": days_until
                        })
                except (ValueError, AttributeError):
                    continue
            
            # Sort by date
            upcoming.sort(key=lambda x: x["days_until"])
            
        except Exception as e:
            print(f"Error getting upcoming dates: {e}")
        
        return upcoming

    def get_query_urgency(self, query: str) -> str:
        """
        Determine urgency level of the query based on content and timing
        """
        urgent_keywords = ["urgent", "asap", "immediately", "deadline", "last date", "today", "tomorrow"]
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in urgent_keywords):
            return "high"
        
        # Check if query is about upcoming deadlines
        context = self.get_academic_context()
        upcoming_dates = context.get("upcoming_dates", [])
        
        if upcoming_dates and any(date_info["days_until"] <= 7 for date_info in upcoming_dates):
            intents = self.analyze_query_intent(query)
            if any(intent in ["deadlines", "registration", "documents"] for intent in intents):
                return "high"
        
        return "normal"
    
    def analyze_query_context(self, query: str) -> Dict[str, any]:
        """
        Wrapper method to provide a simplified interface
        for context analysis to be used in prompt_manager.
        """
        # You can customize this to return what prompt_manager expects.

        return {
            "primary_intent": self.analyze_query_intent(query)[0],
            "intents": self.analyze_query_intent(query),
            "program_stream": self.identify_program_stream(query),
            "clarification": self.identify_clarification_needs(query),
            "academic_context": self.get_academic_context(),
            "urgency": self.get_query_urgency(query)
    }


    def analyze_complete_context(self, query: str) -> Dict[str, any]:
        """
        Complete context analysis combining all methods
        """
        return {
            "intents": self.analyze_query_intent(query),
            "program_stream": self.identify_program_stream(query),
            "clarification": self.identify_clarification_needs(query),
            "academic_context": self.get_academic_context(),
            "urgency": self.get_query_urgency(query),
            "query_complexity": "simple" if len(query.split()) <= 5 else "complex"
        }