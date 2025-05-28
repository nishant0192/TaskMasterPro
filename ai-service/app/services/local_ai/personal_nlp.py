# ai-service/app/services/local_ai/personal_nlp.py
import re
import spacy
import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import dateutil.parser
import pytz
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.core.database import get_async_session
from app.models.database import UserVocabulary, TaskEmbedding, UserBehaviorPattern
from app.models.schemas import ParsedTask, ExtractedEntity, TaskPriority, NLPRequest
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class PersonalizedNLPResult:
    parsed_task: ParsedTask
    extracted_entities: List[ExtractedEntity]
    confidence_score: float
    suggestions: List[str]
    alternative_interpretations: List[ParsedTask]
    personalization_applied: bool

class PersonalizedNLP:
    """Personalized Natural Language Processing for task creation"""
    
    def __init__(self, user_id: str, model_manager: ModelManager):
        self.user_id = user_id
        self.model_manager = model_manager
        self.nlp = None
        self.is_initialized = False
        
        # User-specific vocabulary and patterns
        self.user_vocabulary: Dict[str, Dict[str, str]] = {}
        self.user_patterns: Dict[str, Any] = {}
        self.user_shortcuts: Dict[str, str] = {}
        self.category_keywords: Dict[str, List[str]] = {}
        
        # NLP patterns
        self._compile_regex_patterns()
        
    def _compile_regex_patterns(self):
        """Compile regex patterns for entity extraction"""
        self.time_patterns = [
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(in|after)\s+(\d+)\s+(minutes?|hours?|days?|weeks?|months?)\b',
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            r'\b(\d{1,2})\s*(am|pm)\b',
            r'\b(next|this)\s+(week|month|year)\b',
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-]?(\d{2,4})?\b',
        ]
        
        self.priority_patterns = [
            r'\b(urgent|asap|critical|important|high priority)\b',
            r'\b(low priority|not urgent|whenever|someday)\b',
            r'\b(priority|pri)\s*[:\-]?\s*([1-5]|high|medium|low)\b',
            r'\b(!!|!!!)\b',
        ]
        
        self.duration_patterns = [
            r'\b(\d+)\s*(minutes?|mins?|hours?|hrs?|h)\b',
            r'\btakes?\s+(\d+)\s*(minutes?|hours?)\b',
            r'\b(\d+)\s*[-]\s*(\d+)\s*(minutes?|hours?)\b',
        ]

    async def initialize(self):
        """Initialize the personalized NLP processor"""
        try:
            logger.info(f"Initializing personalized NLP for user {self.user_id}")
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic NLP")
                self.nlp = None
            
            # Load user-specific vocabulary and patterns
            await self._load_user_vocabulary()
            await self._load_user_patterns()
            await self._load_category_keywords()
            
            self.is_initialized = True
            logger.info(f"Personalized NLP initialized for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize personalized NLP: {e}")
            raise

    async def _load_user_vocabulary(self):
        """Load user-specific vocabulary from database"""
        try:
            async with get_async_session() as session:
                query = select(UserVocabulary).where(
                    UserVocabulary.user_id == self.user_id
                ).order_by(UserVocabulary.frequency.desc())
                
                result = await session.execute(query)
                vocab_records = result.scalars().all()
                
                for record in vocab_records:
                    term_type = record.term_type
                    if term_type not in self.user_vocabulary:
                        self.user_vocabulary[term_type] = {}
                    
                    self.user_vocabulary[term_type][record.term.lower()] = {
                        'normalized_form': record.normalized_form,
                        'context': record.context,
                        'frequency': record.frequency,
                        'confidence': record.confidence
                    }
                
                logger.debug(f"Loaded {len(vocab_records)} vocabulary terms for user {self.user_id}")
                
        except Exception as e:
            logger.error(f"Error loading user vocabulary: {e}")

    async def _load_user_patterns(self):
        """Load user behavior patterns for NLP personalization"""
        try:
            async with get_async_session() as session:
                query = select(UserBehaviorPattern).where(
                    and_(
                        UserBehaviorPattern.user_id == self.user_id,
                        UserBehaviorPattern.pattern_type.in_([
                            'language_patterns', 'time_expressions', 'priority_expressions'
                        ])
                    )
                )
                
                result = await session.execute(query)
                patterns = result.scalars().all()
                
                for pattern in patterns:
                    self.user_patterns[pattern.pattern_type] = {
                        'data': pattern.pattern_data,
                        'confidence': pattern.confidence_score,
                        'frequency': pattern.frequency_count
                    }
                
                logger.debug(f"Loaded {len(patterns)} behavior patterns for user {self.user_id}")
                
        except Exception as e:
            logger.error(f"Error loading user patterns: {e}")

    async def _load_category_keywords(self):
        """Load user-specific category keywords"""
        try:
            async with get_async_session() as session:
                # Get task embeddings to analyze category patterns
                query = select(TaskEmbedding).where(
                    and_(
                        TaskEmbedding.user_id == self.user_id,
                        TaskEmbedding.category.isnot(None)
                    )
                ).limit(1000)
                
                result = await session.execute(query)
                task_records = result.scalars().all()
                
                # Build category keywords from user's actual tasks
                category_keywords = {}
                for record in task_records:
                    if record.category:
                        if record.category not in category_keywords:
                            category_keywords[record.category] = set()
                        
                        # Extract keywords from embedding or use basic approach
                        # For now, use simple word extraction
                        # In production, you'd use the actual embeddings
                        
                # Default categories if no user data
                if not category_keywords:
                    category_keywords = {
                        'work': ['work', 'office', 'meeting', 'project', 'deadline', 'client'],
                        'personal': ['personal', 'home', 'family', 'friend', 'self'],
                        'health': ['doctor', 'gym', 'exercise', 'medication', 'health'],
                        'shopping': ['buy', 'shop', 'purchase', 'store', 'mall', 'order'],
                        'learning': ['study', 'learn', 'read', 'course', 'tutorial'],
                        'social': ['call', 'text', 'meet', 'visit', 'party', 'dinner'],
                        'finance': ['pay', 'bill', 'bank', 'money', 'budget'],
                        'travel': ['book', 'flight', 'hotel', 'trip', 'vacation']
                    }
                
                self.category_keywords = {k: list(v) for k, v in category_keywords.items()}
                
        except Exception as e:
            logger.error(f"Error loading category keywords: {e}")

    async def parse_task_input(self, text: str, context: Dict[str, Any] = None,
                             user_timezone: str = "UTC") -> PersonalizedNLPResult:
        """Parse natural language task input with personalization"""
        if not self.is_initialized:
            await self.initialize()
        
        context = context or {}
        
        try:
            # Preprocess text with user-specific patterns
            preprocessed_text = await self._preprocess_with_personalization(text)
            
            # Extract entities
            entities = await self._extract_entities_personalized(
                preprocessed_text, user_timezone
            )
            
            # Parse to structured task
            parsed_task = await self._parse_to_task_personalized(
                preprocessed_text, entities, context
            )
            
            # Calculate confidence score
            confidence = self._calculate_personalized_confidence(
                text, entities, parsed_task
            )
            
            # Generate personalized suggestions
            suggestions = await self._generate_personalized_suggestions(
                text, parsed_task, entities
            )
            
            # Generate alternative interpretations
            alternatives = await self._generate_alternatives_personalized(
                text, entities, context
            )
            
            # Learn from this interaction
            await self._learn_from_input(text, parsed_task)
            
            return PersonalizedNLPResult(
                parsed_task=parsed_task,
                extracted_entities=entities,
                confidence_score=confidence,
                suggestions=suggestions,
                alternative_interpretations=alternatives,
                personalization_applied=True
            )
            
        except Exception as e:
            logger.error(f"Error in personalized NLP parsing: {e}")
            
            # Fallback to basic parsing
            return PersonalizedNLPResult(
                parsed_task=ParsedTask(title=text.strip()),
                extracted_entities=[],
                confidence_score=0.3,
                suggestions=["Unable to parse input with personalization"],
                alternative_interpretations=[],
                personalization_applied=False
            )

    async def _preprocess_with_personalization(self, text: str) -> str:
        """Preprocess text using user-specific vocabulary"""
        processed_text = text.lower().strip()
        
        # Apply user abbreviations
        if 'abbreviation' in self.user_vocabulary:
            for abbr, data in self.user_vocabulary['abbreviation'].items():
                processed_text = re.sub(
                    r'\b' + re.escape(abbr) + r'\b',
                    data['normalized_form'],
                    processed_text,
                    flags=re.IGNORECASE
                )
        
        # Apply user synonyms
        if 'synonym' in self.user_vocabulary:
            for synonym, data in self.user_vocabulary['synonym'].items():
                processed_text = re.sub(
                    r'\b' + re.escape(synonym) + r'\b',
                    data['normalized_form'],
                    processed_text,
                    flags=re.IGNORECASE
                )
        
        # Apply user shortcuts
        if 'shortcut' in self.user_vocabulary:
            for shortcut, data in self.user_vocabulary['shortcut'].items():
                if shortcut in processed_text.lower():
                    processed_text = processed_text.replace(
                        shortcut, data['normalized_form']
                    )
        
        return processed_text

    async def _extract_entities_personalized(self, text: str, 
                                           user_timezone: str) -> List[ExtractedEntity]:
        """Extract entities with user personalization"""
        entities = []
        
        # Extract time entities with user patterns
        time_entities = await self._extract_time_entities_personalized(text, user_timezone)
        entities.extend(time_entities)
        
        # Extract priority entities with user preferences
        priority_entities = await self._extract_priority_entities_personalized(text)
        entities.extend(priority_entities)
        
        # Extract duration entities
        duration_entities = self._extract_duration_entities(text)
        entities.extend(duration_entities)
        
        # Extract category entities with user-specific keywords
        category_entities = self._extract_category_entities_personalized(text)
        entities.extend(category_entities)
        
        # Extract person entities if spaCy is available
        if self.nlp:
            person_entities = self._extract_person_entities(text)
            entities.extend(person_entities)
        
        return entities

    async def _extract_time_entities_personalized(self, text: str, 
                                                user_timezone: str) -> List[ExtractedEntity]:
        """Extract time entities using user patterns"""
        entities = []
        text_lower = text.lower()
        
        # Check user-specific time patterns first
        time_patterns = self.user_patterns.get('time_expressions', {}).get('data', {})
        for pattern, time_value in time_patterns.items():
            if pattern in text_lower:
                try:
                    if isinstance(time_value, str):
                        parsed_time = dateutil.parser.parse(time_value)
                    else:
                        parsed_time = datetime.fromisoformat(time_value)
                    
                    entities.append(ExtractedEntity(
                        entity_type="datetime",
                        value=parsed_time.isoformat(),
                        confidence=0.9,
                        start_pos=text_lower.find(pattern),
                        end_pos=text_lower.find(pattern) + len(pattern)
                    ))
                except Exception:
                    continue
        
        # Standard time extraction
        now = datetime.now(pytz.timezone(user_timezone))
        
        # Relative dates
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
        
        # Day of week extraction
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days):
            if day in text_lower:
                days_ahead = (i - now.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_date = now + timedelta(days=days_ahead)
                
                entities.append(ExtractedEntity(
                    entity_type="date",
                    value=target_date.date().isoformat(),
                    confidence=0.8,
                    start_pos=text_lower.find(day),
                    end_pos=text_lower.find(day) + len(day)
                ))
        
        # Time patterns (3pm, 15:30, etc.)
        time_matches = re.finditer(r'\b(\d{1,2}):?(\d{2})?\s*(am|pm)?\b', text_lower)
        for match in time_matches:
            try:
                hour_str = match.group(1)
                minute_str = match.group(2) or "00"
                ampm = match.group(3)
                
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
        
        return entities

    async def _extract_priority_entities_personalized(self, text: str) -> List[ExtractedEntity]:
        """Extract priority entities using user preferences"""
        entities = []
        text_lower = text.lower()
        
        # Check user-specific priority patterns
        priority_patterns = self.user_patterns.get('priority_expressions', {}).get('data', {})
        for pattern, priority_value in priority_patterns.items():
            if pattern in text_lower:
                entities.append(ExtractedEntity(
                    entity_type="priority",
                    value=str(priority_value),
                    confidence=0.9,
                    start_pos=text_lower.find(pattern),
                    end_pos=text_lower.find(pattern) + len(pattern)
                ))
        
        # Standard priority extraction
        high_priority_terms = ['urgent', 'asap', 'critical', 'important', 'high priority', '!!!', '!!']
        for term in high_priority_terms:
            if term in text_lower:
                entities.append(ExtractedEntity(
                    entity_type="priority",
                    value="5",
                    confidence=0.8,
                    start_pos=text_lower.find(term),
                    end_pos=text_lower.find(term) + len(term)
                ))
        
        low_priority_terms = ['low priority', 'not urgent', 'whenever', 'someday', 'eventually']
        for term in low_priority_terms:
            if term in text_lower:
                entities.append(ExtractedEntity(
                    entity_type="priority",
                    value="1",
                    confidence=0.8,
                    start_pos=text_lower.find(term),
                    end_pos=text_lower.find(term) + len(term)
                ))
        
        return entities

    def _extract_duration_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract duration estimates"""
        entities = []
        
        duration_matches = re.finditer(r'\b(\d+)\s*(minutes?|mins?|hours?|hrs?|h)\b', text.lower())
        for match in duration_matches:
            amount = int(match.group(1))
            unit = match.group(2)
            
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
        
        return entities

    def _extract_category_entities_personalized(self, text: str) -> List[ExtractedEntity]:
        """Extract category entities using user-specific keywords"""
        entities = []
        text_lower = text.lower()
        
        # Check user-specific categories first
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    entities.append(ExtractedEntity(
                        entity_type="category",
                        value=category,
                        confidence=0.7,
                        start_pos=text_lower.find(keyword.lower()),
                        end_pos=text_lower.find(keyword.lower()) + len(keyword)
                    ))
                    break  # Only one category per match
        
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

    async def _parse_to_task_personalized(self, text: str, entities: List[ExtractedEntity],
                                        context: Dict[str, Any]) -> ParsedTask:
        """Parse entities into structured task with personalization"""
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
        
        # Combine date and time
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
        
        # Apply user-specific defaults
        if not priority and 'default_priority' in context:
            try:
                priority = TaskPriority(context['default_priority'])
            except:
                pass
        
        if not estimated_duration and 'default_duration' in context:
            estimated_duration = context['default_duration']
        
        # Clean title by removing entity text
        cleaned_title = self._clean_title_personalized(title, entities)
        
        # Set reminder based on user preferences
        if due_date and not reminder_time:
            reminder_preferences = self.user_patterns.get('reminder_preferences', {}).get('data', {})
            default_reminder_offset = reminder_preferences.get('default_offset_hours', 24)
            reminder_time = due_date - timedelta(hours=default_reminder_offset)
        
        return ParsedTask(
            title=cleaned_title or title,
            description=description,
            due_date=due_date,
            priority=priority,
            estimated_duration=estimated_duration,
            category=category,
            tags=tags,
            reminder_time=reminder_time
        )

    def _clean_title_personalized(self, title: str, entities: List[ExtractedEntity]) -> str:
        """Clean title by removing entity text with user-specific rules"""
        cleaned = title
        
        # Sort entities by position (reverse order)
        sorted_entities = sorted(entities, key=lambda x: x.start_pos, reverse=True)
        
        for entity in sorted_entities:
            # More aggressive cleaning for personalized users
            if entity.entity_type in ["datetime", "date", "time", "priority", "duration"]:
                before = cleaned[:entity.start_pos]
                after = cleaned[entity.end_pos:]
                cleaned = (before + after).strip()
        
        # User-specific title cleaning rules
        cleaning_patterns = self.user_patterns.get('title_cleaning', {}).get('data', {})
        for pattern, replacement in cleaning_patterns.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _calculate_personalized_confidence(self, text: str, entities: List[ExtractedEntity],
                                         parsed_task: ParsedTask) -> float:
        """Calculate confidence score with personalization factors"""
        base_confidence = 0.4  # Higher base for personalized processing
        
        # Boost for successful entity extraction
        if entities:
            base_confidence += min(0.3, len(entities) * 0.08)
        
        # Boost for user vocabulary matches
        vocab_matches = 0
        text_lower = text.lower()
        for term_type, terms in self.user_vocabulary.items():
            for term in terms:
                if term in text_lower:
                    vocab_matches += 1
        
        if vocab_matches > 0:
            base_confidence += min(0.2, vocab_matches * 0.05)
        
        # Boost for pattern matches
        pattern_matches = 0
        for pattern_type, pattern_data in self.user_patterns.items():
            patterns = pattern_data.get('data', {})
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    pattern_matches += 1
        
        if pattern_matches > 0:
            base_confidence += min(0.15, pattern_matches * 0.03)
        
        # Penalize if title is too short
        if len(parsed_task.title.split()) < 2:
            base_confidence -= 0.1
        
        return min(1.0, max(0.1, base_confidence))

    async def _generate_personalized_suggestions(self, text: str, parsed_task: ParsedTask,
                                               entities: List[ExtractedEntity]) -> List[str]:
        """Generate personalized suggestions"""
        suggestions = []
        
        # Suggest based on user patterns
        if not parsed_task.due_date:
            if 'typical_due_date_offset' in self.user_patterns:
                offset_days = self.user_patterns['typical_due_date_offset'].get('data', {}).get('days', 7)
                suggestions.append(f"Consider adding a due date (typically you set tasks {offset_days} days out)")
            else:
                suggestions.append("Consider adding a due date (e.g., 'by Friday', 'tomorrow at 3pm')")
        
        if not parsed_task.priority:
            user_priority_prefs = self.user_patterns.get('priority_preferences', {}).get('data', {})
            common_priority = user_priority_prefs.get('most_common', 3)
            suggestions.append(f"You often use priority {common_priority} - consider setting priority")
        
        if not parsed_task.estimated_duration:
            if parsed_task.category:
                # Use category-specific duration suggestions
                category_durations = self.user_patterns.get('category_durations', {}).get('data', {})
                typical_duration = category_durations.get(parsed_task.category, 60)
                suggestions.append(f"Similar {parsed_task.category} tasks typically take {typical_duration} minutes")
            else:
                suggestions.append("Adding time estimate helps with scheduling")
        
        # Suggest similar tasks
        similar_suggestions = await self._get_similar_task_suggestions(text)
        suggestions.extend(similar_suggestions[:2])  # Limit to 2 similar task suggestions
        
        return suggestions[:4]  # Limit total suggestions

    async def _get_similar_task_suggestions(self, text: str) -> List[str]:
        """Get suggestions based on similar past tasks"""
        try:
            # Get text embedding
            embeddings = await self.model_manager.get_embeddings([text])
            if len(embeddings) == 0:
                return []
            
            query_embedding = embeddings[0]
            
            # Find similar tasks (simplified - in production use vector similarity)
            async with get_async_session() as session:
                query = select(TaskEmbedding).where(
                    TaskEmbedding.user_id == self.user_id
                ).limit(100)
                
                result = await session.execute(query)
                task_records = result.scalars().all()
                
                suggestions = []
                for record in task_records:
                    # Simple text similarity check (replace with vector similarity)
                    if any(word in text.lower() for word in record.category.lower().split() if record.category):
                        if record.estimated_duration:
                            suggestions.append(f"Similar tasks usually take {record.estimated_duration} minutes")
                        if record.priority:
                            suggestions.append(f"You typically set priority {record.priority} for {record.category} tasks")
                
                return list(set(suggestions))  # Remove duplicates
                
        except Exception as e:
            logger.error(f"Error getting similar task suggestions: {e}")
            return []

    async def _generate_alternatives_personalized(self, text: str, entities: List[ExtractedEntity],
                                                context: Dict[str, Any]) -> List[ParsedTask]:
        """Generate alternative interpretations with personalization"""
        alternatives = []
        
        # Alternative with different priority based on user patterns
        if any(e.entity_type == "priority" for e in entities):
            alt_task = await self._parse_to_task_personalized(text, entities, context)
            if alt_task.priority and alt_task.priority.value > 1:
                alt_task.priority = TaskPriority(alt_task.priority.value - 1)
                alternatives.append(alt_task)
        
        # Alternative with user's typical duration for category
        category_entity = next((e for e in entities if e.entity_type == "category"), None)
        if category_entity:
            category_durations = self.user_patterns.get('category_durations', {}).get('data', {})
            typical_duration = category_durations.get(category_entity.value)
            if typical_duration:
                alt_task = await self._parse_to_task_personalized(text, entities, context)
                alt_task.estimated_duration = typical_duration
                alternatives.append(alt_task)
        
        return alternatives[:2]  # Limit alternatives

    async def _learn_from_input(self, text: str, parsed_task: ParsedTask):
        """Learn from user input to improve personalization"""
        try:
            # Extract new vocabulary terms
            await self._extract_and_store_vocabulary(text, parsed_task)
            
            # Update behavior patterns
            await self._update_behavior_patterns(text, parsed_task)
            
        except Exception as e:
            logger.error(f"Error learning from input: {e}")

    async def _extract_and_store_vocabulary(self, text: str, parsed_task: ParsedTask):
        """Extract and store new vocabulary terms"""
        try:
            words = text.lower().split()
            
            # Look for potential abbreviations (short words that map to longer concepts)
            potential_abbrevs = [word for word in words if len(word) <= 4 and word.isalpha()]
            
            # Look for potential category indicators
            if parsed_task.category:
                category_words = [word for word in words 
                                if word.lower() not in ['a', 'an', 'the', 'and', 'or', 'but']]
                
                for word in category_words:
                    await self._store_vocabulary_term(
                        word, parsed_task.category, 'category_indicator', 
                        context=f"indicates {parsed_task.category} tasks"
                    )
            
        except Exception as e:
            logger.error(f"Error extracting vocabulary: {e}")

    async def _store_vocabulary_term(self, term: str, normalized_form: str, 
                                   term_type: str, context: str = None):
        """Store a vocabulary term in the database"""
        try:
            async with get_async_session() as session:
                # Check if term already exists
                query = select(UserVocabulary).where(
                    and_(
                        UserVocabulary.user_id == self.user_id,
                        UserVocabulary.term == term.lower(),
                        UserVocabulary.term_type == term_type
                    )
                )
                
                result = await session.execute(query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update frequency
                    existing.frequency += 1
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new term
                    vocab_term = UserVocabulary(
                        user_id=self.user_id,
                        term=term.lower(),
                        normalized_form=normalized_form,
                        term_type=term_type,
                        context=context,
                        frequency=1,
                        confidence=0.6
                    )
                    session.add(vocab_term)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error storing vocabulary term: {e}")

    async def _update_behavior_patterns(self, text: str, parsed_task: ParsedTask):
        """Update user behavior patterns based on input"""
        try:
            current_time = datetime.utcnow()
            
            # Update time expression patterns
            if parsed_task.due_date:
                time_pattern = self._extract_time_pattern(text, parsed_task.due_date)
                if time_pattern:
                    await self._update_pattern('time_expressions', time_pattern, parsed_task.due_date)
            
            # Update priority expression patterns
            if parsed_task.priority:
                priority_pattern = self._extract_priority_pattern(text, parsed_task.priority)
                if priority_pattern:
                    await self._update_pattern('priority_expressions', priority_pattern, parsed_task.priority.value)
            
        except Exception as e:
            logger.error(f"Error updating behavior patterns: {e}")

    def _extract_time_pattern(self, text: str, due_date: datetime) -> Optional[str]:
        """Extract time pattern from text"""
        text_lower = text.lower()
        
        # Simple pattern extraction (could be enhanced)
        time_words = ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 
                     'thursday', 'friday', 'saturday', 'sunday', 'next week']
        
        for word in time_words:
            if word in text_lower:
                return word
        
        return None

    def _extract_priority_pattern(self, text: str, priority: TaskPriority) -> Optional[str]:
        """Extract priority pattern from text"""
        text_lower = text.lower()
        
        priority_words = ['urgent', 'important', 'asap', 'critical', 'low priority', 
                         'high priority', '!!', '!!!']
        
        for word in priority_words:
            if word in text_lower:
                return word
        
        return None

    async def _update_pattern(self, pattern_type: str, pattern: str, value: Any):
        """Update behavior pattern in database"""
        try:
            async with get_async_session() as session:
                query = select(UserBehaviorPattern).where(
                    and_(
                        UserBehaviorPattern.user_id == self.user_id,
                        UserBehaviorPattern.pattern_type == pattern_type
                    )
                )
                
                result = await session.execute(query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing pattern
                    pattern_data = existing.pattern_data or {}
                    pattern_data[pattern] = value
                    existing.pattern_data = pattern_data
                    existing.frequency_count += 1
                    existing.confidence_score = min(1.0, existing.confidence_score + 0.05)
                    existing.last_observed = datetime.utcnow()
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new pattern
                    new_pattern = UserBehaviorPattern(
                        user_id=self.user_id,
                        pattern_type=pattern_type,
                        pattern_data={pattern: value},
                        confidence_score=0.6,
                        frequency_count=1
                    )
                    session.add(new_pattern)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error updating pattern: {e}")

    async def get_personalization_stats(self) -> Dict[str, Any]:
        """Get personalization statistics"""
        return {
            'vocabulary_terms': sum(len(terms) for terms in self.user_vocabulary.values()),
            'behavior_patterns': len(self.user_patterns),
            'category_keywords': len(self.category_keywords),
            'is_initialized': self.is_initialized
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.debug(f"Cleaning up personalized NLP for user {self.user_id}")
        self.user_vocabulary.clear()
        self.user_patterns.clear()
        self.user_shortcuts.clear()
        self.category_keywords.clear()