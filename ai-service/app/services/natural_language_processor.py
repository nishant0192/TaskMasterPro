# app/services/natural_language_processor.py
import re
import spacy
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass
import dateutil.parser
import pytz
from word2number import w2n

from app.models.schemas import (
    ParsedTask, ExtractedEntity, TaskPriority, NLPRequest
)
from app.core.exceptions import InsufficientDataException

logger = logging.getLogger(__name__)

@dataclass
class NLPResult:
    parsed_task: ParsedTask
    extracted_entities: List[ExtractedEntity]
    confidence_score: float
    suggestions: List[str]
    alternative_interpretations: List[ParsedTask]

class PersonalizedNLPProcessor:
    """Personalized NLP that learns user's language patterns"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_vocabulary: Dict[str, str] = {}  # User-specific terms
        self.user_patterns: Dict[str, Any] = {}    # User's common patterns
        self.user_shortcuts: Dict[str, str] = {}   # User-defined shortcuts
        
    def learn_user_pattern(self, text: str, parsed_result: ParsedTask):
        """Learn from user's input patterns"""
        # Extract common phrases and their meanings
        text_lower = text.lower()
        
        # Learn time expressions
        if parsed_result.due_date:
            time_phrases = self._extract_time_phrases(text_lower)
            for phrase in time_phrases:
                self.user_patterns[phrase] = parsed_result.due_date
        
        # Learn priority expressions
        if parsed_result.priority:
            priority_phrases = self._extract_priority_phrases(text_lower)
            for phrase in priority_phrases:
                self.user_patterns[phrase] = parsed_result.priority
                
        # Learn category/tag patterns
        if parsed_result.category:
            category_phrases = self._extract_category_phrases(text_lower)
            for phrase in category_phrases:
                self.user_patterns[phrase] = parsed_result.category

class NaturalLanguageProcessor:
    def __init__(self):
        self.nlp = None
        self.is_initialized = False
        self.user_processors: Dict[str, PersonalizedNLPProcessor] = {}
        
        # Common patterns
        self.time_patterns = [
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(in|after)\s+(\d+)\s+(minutes?|hours?|days?|weeks?|months?)\b',
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            r'\b(\d{1,2})\s*(am|pm)\b',
            r'\b(next|this)\s+(week|month|year)\b',
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-]?(\d{2,4})?\b',  # dates
        ]
        
        self.priority_patterns = [
            r'\b(urgent|asap|critical|important|high priority)\b',
            r'\b(low priority|not urgent|whenever|someday)\b',
            r'\b(priority|pri)\s*[:\-]?\s*([1-5]|high|medium|low)\b',
            r'\b(!!|!!!)\b',  # Exclamation marks for urgency
        ]
        
        self.duration_patterns = [
            r'\b(\d+)\s*(minutes?|mins?|hours?|hrs?|h)\b',
            r'\btakes?\s+(\d+)\s*(minutes?|hours?)\b',
            r'\b(\d+)\s*[-]\s*(\d+)\s*(minutes?|hours?)\b',  # range
        ]
        
        self.category_keywords = {
            'work': ['work', 'office', 'meeting', 'project', 'deadline', 'client', 'boss'],
            'personal': ['personal', 'home', 'family', 'friend', 'self'],
            'health': ['doctor', 'gym', 'exercise', 'medication', 'health', 'workout'],
            'shopping': ['buy', 'shop', 'purchase', 'store', 'mall', 'order'],
            'learning': ['study', 'learn', 'read', 'course', 'tutorial', 'research'],
            'social': ['call', 'text', 'meet', 'visit', 'party', 'dinner', 'coffee'],
            'finance': ['pay', 'bill', 'bank', 'money', 'budget', 'expense'],
            'travel': ['book', 'flight', 'hotel', 'trip', 'vacation', 'travel']
        }

    async def initialize(self):
        """Initialize the NLP service"""
        try:
            logger.info("Initializing Natural Language Processing Service...")
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic NLP")
                self.nlp = None
            
            self.is_initialized = True
            logger.info("Natural Language Processing Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP Service: {e}")
            raise

    async def parse_task_input(self, text: str, context: Dict[str, Any], 
                             user_timezone: str = "UTC", user_id: str = None) -> NLPResult:
        """Main method to parse natural language task input"""
        if not self.is_initialized:
            raise Exception("NLP Service not initialized")
        
        try:
            # Get or create personalized processor
            if user_id:
                if user_id not in self.user_processors:
                    self.user_processors[user_id] = PersonalizedNLPProcessor(user_id)
                user_processor = self.user_processors[user_id]
            else:
                user_processor = None
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Extract entities
            entities = await self._extract_entities(cleaned_text, user_timezone, user_processor)
            
            # Parse into structured task
            parsed_task = await self._parse_to_task(cleaned_text, entities, context, user_processor)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(text, entities, parsed_task)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(text, parsed_task, entities)
            
            # Generate alternative interpretations
            alternatives = await self._generate_alternatives(text, entities, context)
            
            # Learn from this interaction if user is identified
            if user_processor and parsed_task:
                user_processor.learn_user_pattern(text, parsed_task)
            
            return NLPResult(
                parsed_task=parsed_task,
                extracted_entities=entities,
                confidence_score=confidence,
                suggestions=suggestions,
                alternative_interpretations=alternatives
            )
            
        except Exception as e:
            logger.error(f"NLP parsing failed: {e}")
            # Return fallback result
            return NLPResult(
                parsed_task=ParsedTask(title=text.strip()),
                extracted_entities=[],
                confidence_score=0.3,
                suggestions=["Unable to parse input, using as task title"],
                alternative_interpretations=[]
            )

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Expand common abbreviations
        abbreviations = {
            r'\btmrw\b': 'tomorrow',
            r'\btdy\b': 'today',
            r'\bpls\b': 'please',
            r'\basap\b': 'as soon as possible',
            r'\bfyi\b': 'for your information',
            r'\beod\b': 'end of day',
            r'\bcob\b': 'close of business',
        }
        
        for abbr, expansion in abbreviations.items():
            text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)
        
        return text

    async def _extract_entities(self, text: str, user_timezone: str, 
                               user_processor: Optional[PersonalizedNLPProcessor]) -> List[ExtractedEntity]:
        """Extract entities from text"""
        entities = []
        text_lower = text.lower()
        
        # Extract time entities
        time_entities = self._extract_time_entities(text, user_timezone, user_processor)
        entities.extend(time_entities)
        
        # Extract priority entities
        priority_entities = self._extract_priority_entities(text, user_processor)
        entities.extend(priority_entities)
        
        # Extract duration entities
        duration_entities = self._extract_duration_entities(text)
        entities.extend(duration_entities)
        
        # Extract person entities (if spaCy available)
        if self.nlp:
            person_entities = self._extract_person_entities(text)
            entities.extend(person_entities)
        
        # Extract location entities (if spaCy available)
        if self.nlp:
            location_entities = self._extract_location_entities(text)
            entities.extend(location_entities)
        
        # Extract category entities
        category_entities = self._extract_category_entities(text, user_processor)
        entities.extend(category_entities)
        
        return entities

    def _extract_time_entities(self, text: str, user_timezone: str, 
                              user_processor: Optional[PersonalizedNLPProcessor]) -> List[ExtractedEntity]:
        """Extract time-related entities"""
        entities = []
        text_lower = text.lower()
        
        # Check user patterns first
        if user_processor:
            for pattern, time_value in user_processor.user_patterns.items():
                if pattern in text_lower and isinstance(time_value, datetime):
                    entities.append(ExtractedEntity(
                        entity_type="datetime",
                        value=time_value.isoformat(),
                        confidence=0.9,
                        start_pos=text_lower.find(pattern),
                        end_pos=text_lower.find(pattern) + len(pattern)
                    ))
        
        # Relative dates
        now = datetime.now(pytz.timezone(user_timezone))
        
        if 'today' in text_lower:
            entities.append(ExtractedEntity(
                entity_type="date",
                value=now.date().isoformat(),
                confidence=0.95,
                start_pos=text_lower.find('today'),
                end_pos=text_lower.find('today') + 5
            ))
        
        if 'tomorrow' in text_lower:
            tomorrow = now + timedelta(days=1)
            entities.append(ExtractedEntity(
                entity_type="date",
                value=tomorrow.date().isoformat(),
                confidence=0.95,
                start_pos=text_lower.find('tomorrow'),
                end_pos=text_lower.find('tomorrow') + 8
            ))
        
        # Day of week
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days):
            if day in text_lower:
                # Find next occurrence of this day
                days_ahead = (i - now.weekday()) % 7
                if days_ahead == 0:  # Today is the day
                    days_ahead = 7  # Next week
                target_date = now + timedelta(days=days_ahead)
                
                entities.append(ExtractedEntity(
                    entity_type="date",
                    value=target_date.date().isoformat(),
                    confidence=0.8,
                    start_pos=text_lower.find(day),
                    end_pos=text_lower.find(day) + len(day)
                ))
        
        # Time patterns (e.g., "3pm", "15:30")
        time_matches = re.finditer(r'\b(\d{1,2}):?(\d{2})?\s*(am|pm)?\b', text_lower)
        for match in time_matches:
            hour_str = match.group(1)
            minute_str = match.group(2) or "00"
            ampm = match.group(3)
            
            try:
                hour = int(hour_str)
                minute = int(minute_str)
                
                if ampm:
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                
                time_obj = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                entities.append(ExtractedEntity(
                    entity_type="time",
                    value=time_obj.time().isoformat(),
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
            except ValueError:
                continue
        
        # Duration patterns (e.g., "in 2 hours", "after 30 minutes")
        duration_matches = re.finditer(r'\b(?:in|after)\s+(\d+)\s+(minutes?|hours?|days?|weeks?)\b', text_lower)
        for match in duration_matches:
            amount = int(match.group(1))
            unit = match.group(2)
            
            if 'minute' in unit:
                target_time = now + timedelta(minutes=amount)
            elif 'hour' in unit:
                target_time = now + timedelta(hours=amount)
            elif 'day' in unit:
                target_time = now + timedelta(days=amount)
            elif 'week' in unit:
                target_time = now + timedelta(weeks=amount)
            else:
                continue
            
            entities.append(ExtractedEntity(
                entity_type="datetime",
                value=target_time.isoformat(),
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        return entities

    def _extract_priority_entities(self, text: str, 
                                  user_processor: Optional[PersonalizedNLPProcessor]) -> List[ExtractedEntity]:
        """Extract priority-related entities"""
        entities = []
        text_lower = text.lower()
        
        # Check user patterns first
        if user_processor:
            for pattern, priority_value in user_processor.user_patterns.items():
                if pattern in text_lower and isinstance(priority_value, TaskPriority):
                    entities.append(ExtractedEntity(
                        entity_type="priority",
                        value=str(priority_value.value),
                        confidence=0.9,
                        start_pos=text_lower.find(pattern),
                        end_pos=text_lower.find(pattern) + len(pattern)
                    ))
        
        # High priority indicators
        high_priority_terms = ['urgent', 'asap', 'critical', 'important', 'high priority', '!!!', '!!']
        for term in high_priority_terms:
            if term in text_lower:
                entities.append(ExtractedEntity(
                    entity_type="priority",
                    value="5",  # Critical
                    confidence=0.8,
                    start_pos=text_lower.find(term),
                    end_pos=text_lower.find(term) + len(term)
                ))
        
        # Low priority indicators
        low_priority_terms = ['low priority', 'not urgent', 'whenever', 'someday', 'eventually']
        for term in low_priority_terms:
            if term in text_lower:
                entities.append(ExtractedEntity(
                    entity_type="priority",
                    value="1",  # Very Low
                    confidence=0.8,
                    start_pos=text_lower.find(term),
                    end_pos=text_lower.find(term) + len(term)
                ))
        
        # Explicit priority numbers
        priority_matches = re.finditer(r'\b(?:priority|pri)\s*[:\-]?\s*([1-5])\b', text_lower)
        for match in priority_matches:
            priority_value = match.group(1)
            entities.append(ExtractedEntity(
                entity_type="priority",
                value=priority_value,
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        return entities

    def _extract_duration_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract duration estimates"""
        entities = []
        
        # Duration patterns
        duration_matches = re.finditer(r'\b(\d+)\s*(minutes?|mins?|hours?|hrs?|h)\b', text.lower())
        for match in duration_matches:
            amount = int(match.group(1))
            unit = match.group(2)
            
            # Convert to minutes
            if 'hour' in unit or unit == 'h':
                duration_minutes = amount * 60
            else:
                duration_minutes = amount
            
            entities.append(ExtractedEntity(
                entity_type="duration",
                value=str(duration_minutes),
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Implied duration patterns
        quick_tasks = ['quick', 'briefly', 'short', 'fast']
        for term in quick_tasks:
            if term in text.lower():
                entities.append(ExtractedEntity(
                    entity_type="duration",
                    value="15",  # 15 minutes
                    confidence=0.5,
                    start_pos=text.lower().find(term),
                    end_pos=text.lower().find(term) + len(term)
                ))
        
        return entities

    def _extract_person_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract person names using spaCy"""
        entities = []
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(ExtractedEntity(
                    entity_type="person",
                    value=ent.text,
                    confidence=0.7,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                ))
        
        return entities

    def _extract_location_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract location names using spaCy"""
        entities = []
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:  # Geopolitical, Location, Facility
                entities.append(ExtractedEntity(
                    entity_type="location",
                    value=ent.text,
                    confidence=0.7,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                ))
        
        return entities

    def _extract_category_entities(self, text: str, 
                                  user_processor: Optional[PersonalizedNLPProcessor]) -> List[ExtractedEntity]:
        """Extract task category based on keywords"""
        entities = []
        text_lower = text.lower()
        
        # Check user patterns first
        if user_processor:
            for pattern, category_value in user_processor.user_patterns.items():
                if pattern in text_lower and isinstance(category_value, str):
                    entities.append(ExtractedEntity(
                        entity_type="category",
                        value=category_value,
                        confidence=0.9,
                        start_pos=text_lower.find(pattern),
                        end_pos=text_lower.find(pattern) + len(pattern)
                    ))
        
        # Check predefined categories
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append(ExtractedEntity(
                        entity_type="category",
                        value=category,
                        confidence=0.6,
                        start_pos=text_lower.find(keyword),
                        end_pos=text_lower.find(keyword) + len(keyword)
                    ))
                    break  # Only one category per keyword match
        
        return entities

    async def _parse_to_task(self, text: str, entities: List[ExtractedEntity], 
                           context: Dict[str, Any], 
                           user_processor: Optional[PersonalizedNLPProcessor]) -> ParsedTask:
        """Parse extracted entities into a structured task"""
        
        # Start with the full text as title
        title = text.strip()
        description = None
        due_date = None
        priority = None
        estimated_duration = None
        category = None
        tags = []
        reminder_time = None
        
        # Process entities
        date_entity = None
        time_entity = None
        
        for entity in entities:
            if entity.entity_type == "date":
                date_entity = entity
            elif entity.entity_type == "time":
                time_entity = entity
            elif entity.entity_type == "datetime":
                try:
                    due_date = datetime.fromisoformat(entity.value.replace('Z', '+00:00'))
                except:
                    pass
            elif entity.entity_type == "priority":
                try:
                    priority_value = int(entity.value)
                    priority = TaskPriority(priority_value)
                except:
                    pass
            elif entity.entity_type == "duration":
                try:
                    estimated_duration = int(entity.value)
                except:
                    pass
            elif entity.entity_type == "category":
                category = entity.value
            elif entity.entity_type == "person":
                tags.append(f"person:{entity.value}")
            elif entity.entity_type == "location":
                tags.append(f"location:{entity.value}")
        
        # Combine date and time if both present
        if date_entity and time_entity and not due_date:
            try:
                date_part = datetime.fromisoformat(date_entity.value).date()
                time_part = datetime.fromisoformat(f"2000-01-01T{time_entity.value}").time()
                due_date = datetime.combine(date_part, time_part)
            except:
                pass
        elif date_entity and not due_date:
            try:
                due_date = datetime.fromisoformat(date_entity.value)
            except:
                pass
        
        # Clean title by removing recognized entity text
        cleaned_title = self._clean_title(title, entities)
        
        # Set reminder if due date exists
        if due_date and not reminder_time:
            # Default: remind 1 hour before for same-day tasks, 1 day before for future tasks
            if due_date.date() == datetime.now().date():
                reminder_time = due_date - timedelta(hours=1)
            else:
                reminder_time = due_date - timedelta(days=1)
        
        return ParsedTask(
            title=cleaned_title or title,  # Fallback to original if cleaned is empty
            description=description,
            due_date=due_date,
            priority=priority,
            estimated_duration=estimated_duration,
            category=category,
            tags=tags,
            reminder_time=reminder_time
        )

    def _clean_title(self, title: str, entities: List[ExtractedEntity]) -> str:
        """Remove entity text from title to clean it up"""
        cleaned = title
        
        # Sort entities by position (reverse order to maintain indices)
        sorted_entities = sorted(entities, key=lambda x: x.start_pos, reverse=True)
        
        for entity in sorted_entities:
            # Only remove certain types of entities from title
            if entity.entity_type in ["datetime", "date", "time", "priority", "duration"]:
                # Remove the entity text
                before = cleaned[:entity.start_pos]
                after = cleaned[entity.end_pos:]
                cleaned = (before + after).strip()
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def _calculate_confidence(self, text: str, entities: List[ExtractedEntity], 
                            parsed_task: ParsedTask) -> float:
        """Calculate confidence score for the parsing"""
        base_confidence = 0.3
        
        # Boost confidence based on successfully extracted entities
        if entities:
            base_confidence += min(0.4, len(entities) * 0.1)
        
        # Boost confidence for specific entity types
        entity_types = [e.entity_type for e in entities]
        if "datetime" in entity_types or ("date" in entity_types and "time" in entity_types):
            base_confidence += 0.2
        if "priority" in entity_types:
            base_confidence += 0.1
        if "duration" in entity_types:
            base_confidence += 0.1
        
        # Penalize if title is too short or unclear
        if len(parsed_task.title.split()) < 2:
            base_confidence -= 0.1
        
        return min(1.0, max(0.1, base_confidence))

    async def _generate_suggestions(self, text: str, parsed_task: ParsedTask, 
                                  entities: List[ExtractedEntity]) -> List[str]:
        """Generate helpful suggestions for the user"""
        suggestions = []
        
        # Suggest adding missing information
        if not parsed_task.due_date:
            suggestions.append("Consider adding a due date (e.g., 'by Friday', 'tomorrow at 3pm')")
        
        if not parsed_task.priority:
            suggestions.append("You can specify priority (e.g., 'high priority', 'urgent', or 'priority 3')")
        
        if not parsed_task.estimated_duration:
            suggestions.append("Adding time estimate helps with scheduling (e.g., '30 minutes', '2 hours')")
        
        if not parsed_task.category:
            suggestions.append("Try adding context keywords to auto-categorize (e.g., 'work meeting', 'doctor appointment')")
        
        # Suggest improvements based on text analysis
        if len(text.split()) < 3:
            suggestions.append("More descriptive titles help with organization and searching")
        
        return suggestions[:3]  # Limit to top 3 suggestions

    async def _generate_alternatives(self, text: str, entities: List[ExtractedEntity], 
                                   context: Dict[str, Any]) -> List[ParsedTask]:
        """Generate alternative interpretations"""
        alternatives = []
        
        # Alternative 1: Different priority interpretation
        if any(e.entity_type == "priority" for e in entities):
            alt_task = await self._parse_to_task(text, entities, context, None)
            # Modify priority
            if alt_task.priority and alt_task.priority.value > 1:
                alt_task.priority = TaskPriority(alt_task.priority.value - 1)
                alternatives.append(alt_task)
        
        # Alternative 2: Different time interpretation
        time_entities = [e for e in entities if e.entity_type in ["date", "time", "datetime"]]
        if time_entities:
            # Create version with different time interpretation
            # (This would be more sophisticated in practice)
            pass
        
        return alternatives[:2]  # Limit alternatives

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Natural Language Processing Service...")
        self.user_processors.clear()