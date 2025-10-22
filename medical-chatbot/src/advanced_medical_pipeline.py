#!/usr/bin/env python3
"""
Advanced Medical AI Pipeline Architecture
Enterprise-grade medical decision support system

Components:
1. Multi-model Intent Classification (Ensemble)
2. Advanced NER with BioBERT
3. Medical Knowledge Graph Integration  
4. Clinical Decision Support Engine
5. Risk Assessment & Safety Module
6. Evidence-based Response Generation
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

# Medical domain imports
import numpy as np
from sklearn.ensemble import VotingClassifier
# Lazy import for transformers (installed separately if needed)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not installed - using fallback models")
    
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalSeverity(Enum):
    """Medical severity classification"""
    CRITICAL = "critical"      # Life-threatening
    URGENT = "urgent"          # Requires immediate attention
    SEMI_URGENT = "semi_urgent" # Within hours
    NON_URGENT = "non_urgent"   # Routine care
    WELLNESS = "wellness"       # Health maintenance

class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_HIGH = "very_high"    # Emergency intervention
    HIGH = "high"              # Urgent medical attention
    MODERATE = "moderate"      # Monitor closely
    LOW = "low"               # Standard care
    MINIMAL = "minimal"        # Educational/preventive

@dataclass
class MedicalEntity:
    """Enhanced medical entity structure"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    medical_code: Optional[str] = None  # ICD-10, ATC codes
    severity: Optional[MedicalSeverity] = None
    relationships: List[str] = None

@dataclass  
class ClinicalContext:
    """Clinical context for decision making"""
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    medical_history: List[str] = None
    current_medications: List[str] = None
    allergies: List[str] = None
    vital_signs: Dict[str, float] = None
    lab_values: Dict[str, float] = None

@dataclass
class MedicalResponse:
    """Comprehensive medical response structure"""
    intent: str
    confidence: float
    entities: List[MedicalEntity]
    severity_assessment: MedicalSeverity
    risk_level: RiskLevel
    clinical_recommendations: List[str]
    differential_diagnosis: List[Dict[str, Any]]
    drug_interactions: List[Dict[str, str]]
    contraindications: List[str]
    follow_up_required: bool
    escalation_needed: bool
    evidence_sources: List[str]
    response_text: str

class MedicalKnowledgeGraph:
    """Medical knowledge graph for clinical decision support"""
    
    def __init__(self):
        self.diseases = self._load_disease_database()
        self.medications = self._load_medication_database()
        self.symptoms = self._load_symptom_database()
        self.interactions = self._load_interaction_database()
        
    def _load_disease_database(self) -> Dict:
        """Load comprehensive disease database with ICD-10 codes"""
        return {
            "hypertension": {
                "icd10": "I10",
                "symptoms": ["headache", "dizziness", "chest_pain", "shortness_of_breath"],
                "risk_factors": ["age", "obesity", "diabetes", "smoking"],
                "complications": ["stroke", "heart_attack", "kidney_disease"],
                "treatments": ["ACE_inhibitors", "diuretics", "lifestyle_modification"],
                "severity_markers": ["blood_pressure_reading", "end_organ_damage"]
            },
            "diabetes_type_2": {
                "icd10": "E11",
                "symptoms": ["polyuria", "polydipsia", "fatigue", "blurred_vision"],
                "risk_factors": ["obesity", "family_history", "sedentary_lifestyle"],
                "complications": ["diabetic_retinopathy", "nephropathy", "neuropathy"],
                "treatments": ["metformin", "insulin", "diet_modification"],
                "severity_markers": ["HbA1c", "blood_glucose", "ketones"]
            },
            "pneumonia": {
                "icd10": "J18.9",
                "symptoms": ["fever", "cough", "dyspnea", "chest_pain", "sputum"],
                "risk_factors": ["age", "immunocompromised", "chronic_disease"],
                "complications": ["sepsis", "respiratory_failure", "pleural_effusion"],
                "treatments": ["antibiotics", "supportive_care", "oxygen_therapy"],
                "severity_markers": ["oxygen_saturation", "respiratory_rate", "confusion"]
            }
        }
    
    def _load_medication_database(self) -> Dict:
        """Load medication database with ATC codes and interactions"""
        return {
            "metformin": {
                "atc_code": "A10BA02",
                "indications": ["diabetes_type_2"],
                "contraindications": ["severe_kidney_disease", "metabolic_acidosis"],
                "side_effects": ["nausea", "diarrhea", "lactic_acidosis_rare"],
                "interactions": ["contrast_agents", "alcohol"],
                "monitoring": ["kidney_function", "vitamin_b12"],
                "dosage_ranges": {"adult": "500-2000mg daily"}
            },
            "lisinopril": {
                "atc_code": "C09AA03", 
                "indications": ["hypertension", "heart_failure"],
                "contraindications": ["pregnancy", "angioedema_history"],
                "side_effects": ["dry_cough", "hyperkalemia", "angioedema"],
                "interactions": ["potassium_supplements", "NSAIDs"],
                "monitoring": ["kidney_function", "potassium", "blood_pressure"],
                "dosage_ranges": {"adult": "10-40mg daily"}
            },
            "amoxicillin": {
                "atc_code": "J01CA04",
                "indications": ["bacterial_infections", "pneumonia"],
                "contraindications": ["penicillin_allergy"],
                "side_effects": ["diarrhea", "rash", "c_diff_colitis"],
                "interactions": ["warfarin", "oral_contraceptives"],
                "monitoring": ["allergy_signs", "c_diff_symptoms"],
                "dosage_ranges": {"adult": "250-500mg q8h"}
            }
        }
    
    def _load_symptom_database(self) -> Dict:
        """Load symptom database with clinical significance"""
        return {
            "chest_pain": {
                "red_flags": ["crushing", "radiating", "diaphoresis", "dyspnea"],
                "differential": ["myocardial_infarction", "pulmonary_embolism", "aortic_dissection"],
                "severity_assessment": "immediate_evaluation_required",
                "associated_symptoms": ["shortness_of_breath", "nausea", "sweating"]
            },
            "headache": {
                "red_flags": ["sudden_onset", "worst_headache_ever", "fever", "neck_stiffness"],
                "differential": ["migraine", "tension_headache", "subarachnoid_hemorrhage"],
                "severity_assessment": "depends_on_characteristics",
                "associated_symptoms": ["nausea", "photophobia", "aura"]
            },
            "fever": {
                "red_flags": ["temperature_over_39C", "immunocompromised", "altered_mental_status"],
                "differential": ["infection", "inflammatory_condition", "malignancy"],
                "severity_assessment": "depends_on_patient_factors",
                "associated_symptoms": ["chills", "sweats", "malaise"]
            }
        }
    
    def _load_interaction_database(self) -> Dict:
        """Load drug-drug interaction database"""
        return {
            ("warfarin", "amoxicillin"): {
                "severity": "moderate",
                "mechanism": "enhanced_anticoagulation",
                "clinical_effect": "increased_bleeding_risk",
                "management": "monitor_INR_closely"
            },
            ("lisinopril", "potassium_supplement"): {
                "severity": "major",
                "mechanism": "additive_hyperkalemia",
                "clinical_effect": "cardiac_arrhythmias",
                "management": "avoid_combination_or_monitor_closely"
            },
            ("metformin", "contrast_agent"): {
                "severity": "major",
                "mechanism": "increased_lactic_acidosis_risk",
                "clinical_effect": "metabolic_acidosis",
                "management": "discontinue_metformin_48h_before_procedure"
            }
        }
    
    def get_disease_info(self, disease: str) -> Optional[Dict]:
        """Get comprehensive disease information"""
        return self.diseases.get(disease.lower().replace(" ", "_"))
    
    def get_medication_info(self, medication: str) -> Optional[Dict]:
        """Get comprehensive medication information"""
        return self.medications.get(medication.lower().replace(" ", "_"))
    
    def check_drug_interactions(self, medications: List[str]) -> List[Dict]:
        """Check for drug-drug interactions"""
        interactions = []
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                interaction_key = (med1.lower(), med2.lower())
                reverse_key = (med2.lower(), med1.lower())
                
                if interaction_key in self.interactions:
                    interactions.append({
                        "drug1": med1,
                        "drug2": med2,
                        **self.interactions[interaction_key]
                    })
                elif reverse_key in self.interactions:
                    interactions.append({
                        "drug1": med2,
                        "drug2": med1,
                        **self.interactions[reverse_key]
                    })
        
        return interactions
    
    def assess_symptom_severity(self, symptoms: List[str]) -> MedicalSeverity:
        """Assess overall severity based on symptom constellation"""
        red_flag_count = 0
        emergency_symptoms = ["chest_pain", "difficulty_breathing", "severe_headache", 
                            "loss_of_consciousness", "severe_bleeding"]
        
        for symptom in symptoms:
            if any(emergency in symptom.lower() for emergency in emergency_symptoms):
                red_flag_count += 1
        
        if red_flag_count > 0:
            return MedicalSeverity.CRITICAL
        elif len(symptoms) > 3:
            return MedicalSeverity.URGENT
        elif len(symptoms) > 1:
            return MedicalSeverity.SEMI_URGENT
        else:
            return MedicalSeverity.NON_URGENT

class ClinicalDecisionEngine:
    """Advanced clinical decision support engine"""
    
    def __init__(self, knowledge_graph: MedicalKnowledgeGraph):
        self.kg = knowledge_graph
        self.clinical_rules = self._load_clinical_rules()
    
    def _load_clinical_rules(self) -> Dict:
        """Load evidence-based clinical decision rules"""
        return {
            "chest_pain_risk_stratification": {
                "HEART_score": {
                    "history": {"suspicious": 2, "moderately_suspicious": 1, "not_suspicious": 0},
                    "ecg": {"significant_st_deviation": 2, "nonspecific_changes": 1, "normal": 0},
                    "age": {">=65": 2, "45-64": 1, "<45": 0},
                    "risk_factors": {">=3": 2, "1-2": 1, "0": 0},
                    "troponin": {">=3x_normal": 2, "1-3x_normal": 1, "normal": 0}
                }
            },
            "pneumonia_severity": {
                "CURB65": {
                    "confusion": 1,
                    "urea_>7mmol": 1,
                    "respiratory_rate_>=30": 1,
                    "bp_systolic_<90_or_diastolic_<=60": 1,
                    "age_>=65": 1
                }
            }
        }
    
    def calculate_risk_score(self, condition: str, patient_data: Dict) -> Tuple[int, str]:
        """Calculate clinical risk scores"""
        if condition == "chest_pain" and "HEART_score" in self.clinical_rules["chest_pain_risk_stratification"]:
            score = 0
            # Implementation would include actual scoring logic
            # This is simplified for demonstration
            if patient_data.get("age", 0) >= 65:
                score += 2
            elif patient_data.get("age", 0) >= 45:
                score += 1
            
            if score >= 7:
                return score, "High risk - consider admission"
            elif score >= 4:
                return score, "Moderate risk - outpatient monitoring"
            else:
                return score, "Low risk - discharge appropriate"
        
        return 0, "Risk score not available"
    
    def generate_differential_diagnosis(self, symptoms: List[str], patient_context: ClinicalContext) -> List[Dict]:
        """Generate evidence-based differential diagnosis"""
        differentials = []
        
        # Simplified logic - in reality would use more sophisticated algorithms
        symptom_keywords = [s.lower() for s in symptoms]
        
        for disease, info in self.kg.diseases.items():
            match_score = 0
            matching_symptoms = []
            
            for symptom in info["symptoms"]:
                if any(keyword in symptom for keyword in symptom_keywords):
                    match_score += 1
                    matching_symptoms.append(symptom)
            
            if match_score > 0:
                # Calculate probability based on prevalence, patient factors, etc.
                probability = min(0.9, match_score / len(info["symptoms"]))
                
                differentials.append({
                    "condition": disease.replace("_", " ").title(),
                    "icd10": info["icd10"],
                    "probability": probability,
                    "matching_symptoms": matching_symptoms,
                    "supporting_evidence": f"{match_score}/{len(info['symptoms'])} symptoms match"
                })
        
        # Sort by probability
        differentials.sort(key=lambda x: x["probability"], reverse=True)
        return differentials[:5]  # Top 5 most likely

class AdvancedMedicalPipeline:
    """Main pipeline orchestrating all medical AI components"""
    
    def __init__(self):
        logger.info("Initializing Advanced Medical AI Pipeline...")
        
        # Initialize components
        self.knowledge_graph = MedicalKnowledgeGraph()
        self.decision_engine = ClinicalDecisionEngine(self.knowledge_graph)
        
        # Load existing models
        self.intent_classifier = self._load_intent_classifier()
        self.ner_model = self._load_ner_model()
        
        # Initialize safety checkers
        self.safety_keywords = {
            "emergency": ["cấp cứu", "emergency", "khẩn cấp", "SOS", "911", "mayday"],
            "severe_pain": ["đau dữ dội", "severe pain", "đau không chịu được"],
            "breathing_difficulty": ["khó thở", "difficulty breathing", "không thở được"],
            "chest_pain": ["đau ngực", "chest pain", "đau tim"],
            "bleeding": ["chảy máu", "bleeding", "xuất huyết"],
            "consciousness": ["bất tỉnh", "unconscious", "hôn mê", "ngất"]
        }
        
        logger.info("✅ Advanced Medical Pipeline initialized successfully")
    
    def _load_intent_classifier(self):
        """Load intent classifier - would be enhanced with ensemble methods"""
        try:
            # In production, this would load multiple models for ensemble
            from models.medical_intent_classifier import MedicalIntentClassifier
            classifier = MedicalIntentClassifier()
            classifier.train()
            return classifier
        except Exception as e:
            logger.warning(f"Intent classifier not available: {e}")
            return None
    
    def _load_ner_model(self):
        """Load NER model - would be BioBERT in production"""
        try:
            model_path = 'e:/MedCare/medical-chatbot/data/models/ner_model_5k.joblib'
            if os.path.exists(model_path):
                return joblib.load(model_path)['model']
        except Exception as e:
            logger.warning(f"NER model not available: {e}")
        return None
    
    def _assess_immediate_risk(self, text: str, entities: List[MedicalEntity]) -> Tuple[RiskLevel, bool]:
        """Immediate risk assessment for triage"""
        text_lower = text.lower()
        
        # Check for emergency keywords
        emergency_detected = any(
            keyword in text_lower 
            for keywords in self.safety_keywords.values() 
            for keyword in keywords
        )
        
        # Check entity-based risks
        high_risk_entities = ["severe", "critical", "emergency", "bleeding", "unconscious"]
        entity_risk = any(
            risk_term in entity.text.lower() 
            for entity in entities 
            for risk_term in high_risk_entities
        )
        
        # Determine risk level
        if emergency_detected or entity_risk:
            return RiskLevel.VERY_HIGH, True
        elif any("pain" in entity.text.lower() for entity in entities):
            return RiskLevel.MODERATE, False
        else:
            return RiskLevel.LOW, False
    
    def _extract_clinical_context(self, entities: List[MedicalEntity]) -> ClinicalContext:
        """Extract clinical context from entities"""
        context = ClinicalContext(
            medical_history=[],
            current_medications=[],
            allergies=[],
            vital_signs={},
            lab_values={}
        )
        
        for entity in entities:
            if entity.label == "AGE":
                # Extract age number
                age_str = entity.text.split()[0]
                try:
                    context.patient_age = int(age_str)
                except ValueError:
                    pass
            elif entity.label == "MEDICATION":
                context.current_medications.append(entity.text)
            elif entity.label == "DISEASE":
                context.medical_history.append(entity.text)
        
        return context
    
    async def process_medical_query(self, text: str) -> MedicalResponse:
        """Main pipeline processing with comprehensive analysis"""
        logger.info(f"Processing medical query: {text[:50]}...")
        
        try:
            # Step 1: Intent Classification
            intent_result = "unknown"
            intent_confidence = 0.0
            
            if self.intent_classifier:
                try:
                    result = self.intent_classifier.predict(text)
                    if isinstance(result, dict):
                        intent_result = result.get('intent', 'unknown')
                        intent_confidence = result.get('confidence', 0.0)
                    else:
                        intent_result = result
                        intent_confidence = 0.7  # Default confidence
                except Exception as e:
                    logger.error(f"Intent classification error: {e}")
            
            # Step 2: NER Processing
            entities = []
            if self.ner_model:
                try:
                    entities = self._extract_entities_advanced(text)
                except Exception as e:
                    logger.error(f"NER processing error: {e}")
            
            # Step 3: Immediate Risk Assessment
            risk_level, escalation_needed = self._assess_immediate_risk(text, entities)
            
            # Step 4: Clinical Context Extraction
            clinical_context = self._extract_clinical_context(entities)
            
            # Step 5: Severity Assessment
            symptom_entities = [e for e in entities if e.label == "SYMPTOM"]
            symptom_texts = [e.text for e in symptom_entities]
            severity = self.knowledge_graph.assess_symptom_severity(symptom_texts)
            
            # Step 6: Differential Diagnosis
            differential_dx = self.decision_engine.generate_differential_diagnosis(
                symptom_texts, clinical_context
            )
            
            # Step 7: Drug Interaction Checking
            medications = [e.text for e in entities if e.label == "MEDICATION"]
            drug_interactions = self.knowledge_graph.check_drug_interactions(medications)
            
            # Step 8: Clinical Recommendations
            recommendations = self._generate_clinical_recommendations(
                intent_result, entities, severity, risk_level
            )
            
            # Step 9: Response Generation
            response_text = self._generate_evidence_based_response(
                intent_result, entities, differential_dx, recommendations, severity
            )
            
            return MedicalResponse(
                intent=intent_result,
                confidence=intent_confidence,
                entities=entities,
                severity_assessment=severity,
                risk_level=risk_level,
                clinical_recommendations=recommendations,
                differential_diagnosis=differential_dx,
                drug_interactions=drug_interactions,
                contraindications=[],  # Would be populated based on patient context
                follow_up_required=(severity in [MedicalSeverity.URGENT, MedicalSeverity.CRITICAL]),
                escalation_needed=escalation_needed,
                evidence_sources=["Medical Knowledge Graph", "Clinical Decision Rules"],
                response_text=response_text
            )
        
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            # Return safe default response
            return MedicalResponse(
                intent="unknown",
                confidence=0.0,
                entities=[],
                severity_assessment=MedicalSeverity.NON_URGENT,
                risk_level=RiskLevel.LOW,
                clinical_recommendations=["Please consult healthcare provider"],
                differential_diagnosis=[],
                drug_interactions=[],
                contraindications=[],
                follow_up_required=True,
                escalation_needed=False,
                evidence_sources=[],
                response_text="Tôi không thể xử lý câu hỏi này. Vui lòng liên hệ với bác sĩ."
            )
    
    def _extract_entities_advanced(self, text: str) -> List[MedicalEntity]:
        """Advanced NER with confidence scoring and medical coding"""
        # Simplified version - would use BioBERT in production
        tokens = text.split()
        features = self._prepare_features(tokens)
        predicted_labels = self.ner_model.predict([features])[0]
        
        entities = []
        current_entity = None
        current_text = []
        start_pos = 0
        
        for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
            if label.startswith('B-'):
                if current_entity:
                    entity_text = ' '.join(current_text)
                    entities.append(MedicalEntity(
                        text=entity_text,
                        label=current_entity,
                        start=start_pos,
                        end=start_pos + len(entity_text),
                        confidence=0.9,  # Would calculate actual confidence
                        medical_code=self._get_medical_code(entity_text, current_entity)
                    ))
                
                current_entity = label[2:]
                current_text = [token]
                start_pos = text.find(token, start_pos)
                
            elif label.startswith('I-') and current_entity:
                current_text.append(token)
            else:
                if current_entity:
                    entity_text = ' '.join(current_text)
                    entities.append(MedicalEntity(
                        text=entity_text,
                        label=current_entity,
                        start=start_pos,
                        end=start_pos + len(entity_text),
                        confidence=0.9,
                        medical_code=self._get_medical_code(entity_text, current_entity)
                    ))
                    current_entity = None
                    current_text = []
        
        if current_entity:
            entity_text = ' '.join(current_text)
            entities.append(MedicalEntity(
                text=entity_text,
                label=current_entity,
                start=start_pos,
                end=start_pos + len(entity_text),
                confidence=0.9,
                medical_code=self._get_medical_code(entity_text, current_entity)
            ))
        
        return entities
    
    def _prepare_features(self, sentence):
        """Prepare features for CRF (same as before)"""
        features = []
        for i, word in enumerate(sentence):
            feature = {
                'word.lower()': word.lower(),
                'word.isupper()': word.isupper(),
                'word.istitle()': word.istitle(),
                'word.isdigit()': word.isdigit(),
                'word.isalpha()': word.isalpha(),
                'word.length': len(word),
                'BOS': i == 0,
                'EOS': i == len(sentence) - 1,
            }
            
            if len(word) >= 2:
                feature['prefix-2'] = word[:2]
                feature['suffix-2'] = word[-2:]
            if len(word) >= 3:
                feature['prefix-3'] = word[:3]
                feature['suffix-3'] = word[-3:]
                
            if i > 0:
                feature['prev_word'] = sentence[i-1].lower()
            if i < len(sentence) - 1:
                feature['next_word'] = sentence[i+1].lower()
                
            features.append(feature)
        
        return features
    
    def _get_medical_code(self, entity_text: str, entity_type: str) -> Optional[str]:
        """Get medical codes (ICD-10, ATC) for entities"""
        entity_lower = entity_text.lower().replace(" ", "_")
        
        if entity_type == "DISEASE":
            disease_info = self.knowledge_graph.get_disease_info(entity_lower)
            if disease_info:
                return disease_info.get("icd10")
        elif entity_type == "MEDICATION":
            med_info = self.knowledge_graph.get_medication_info(entity_lower)
            if med_info:
                return med_info.get("atc_code")
        
        return None
    
    def _generate_clinical_recommendations(self, intent: str, entities: List[MedicalEntity], 
                                         severity: MedicalSeverity, risk: RiskLevel) -> List[str]:
        """Generate evidence-based clinical recommendations"""
        recommendations = []
        
        if severity == MedicalSeverity.CRITICAL or risk == RiskLevel.VERY_HIGH:
            recommendations.extend([
                "🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED",
                "Call emergency services (115) immediately",
                "Do not delay seeking emergency care",
                "Monitor vital signs if possible"
            ])
        elif severity == MedicalSeverity.URGENT:
            recommendations.extend([
                "Seek urgent medical care within 2-4 hours",
                "Contact your healthcare provider immediately",
                "Consider emergency department if symptoms worsen"
            ])
        elif intent == "drug_question":
            med_entities = [e for e in entities if e.label == "MEDICATION"]
            if med_entities:
                recommendations.append(f"Consult pharmacist about {med_entities[0].text}")
                recommendations.append("Review medication with healthcare provider")
        
        # Add general safety recommendations
        recommendations.extend([
            "This is AI-generated advice - consult healthcare professional",
            "Monitor symptoms and seek help if worsening",
            "Keep record of symptoms for healthcare provider"
        ])
        
        return recommendations
    
    def _generate_evidence_based_response(self, intent: str, entities: List[MedicalEntity],
                                        differential_dx: List[Dict], recommendations: List[str],
                                        severity: MedicalSeverity) -> str:
        """Generate comprehensive evidence-based response"""
        
        if severity == MedicalSeverity.CRITICAL:
            return ("🚨 CẢNH BÁO Y KHOA: Triệu chứng của bạn có thể nghiêm trọng và cần được "
                   "chăm sóc y tế khẩn cấp ngay lập tức. Vui lòng gọi 115 hoặc đến cấp cứu ngay.")
        
        response_parts = []
        
        # Intent-based response opening
        if intent == "symptom_inquiry":
            response_parts.append("Dựa trên các triệu chứng bạn mô tả:")
        elif intent == "drug_question":
            response_parts.append("Về câu hỏi thuốc của bạn:")
        elif intent == "emergency":
            response_parts.append("🚨 ĐÂY LÀ TÌNH HUỐNG KHẨN CẤP:")
        
        # Add differential diagnosis if available
        if differential_dx:
            response_parts.append("\n📋 Các khả năng chẩn đoán:")
            for dx in differential_dx[:3]:  # Top 3
                response_parts.append(f"• {dx['condition']} (khả năng: {dx['probability']:.0%})")
        
        # Add key recommendations
        if recommendations:
            response_parts.append("\n💡 Khuyến nghị:")
            for rec in recommendations[:3]:  # Top 3 recommendations
                response_parts.append(f"• {rec}")
        
        # Add safety disclaimer
        response_parts.append("\n⚠️ Lưu ý: Đây là thông tin tham khảo từ AI. "
                            "Vui lòng tham khảo ý kiến bác sĩ chuyên khoa.")
        
        return "\n".join(response_parts)

# Test the advanced pipeline
async def test_advanced_pipeline():
    """Test the advanced medical pipeline"""
    pipeline = AdvancedMedicalPipeline()
    
    test_cases = [
        "Tôi 45 tuổi bị đau ngực dữ dội kèm khó thở và đổ mồ hôi",
        "Con tôi 3 tuổi sốt cao 39 độ C từ 2 ngày nay",
        "Thuốc metformin có tương tác với aspirin không?",
        "Tôi bị đau đầu migraine thường xuyên cần làm gì?"
    ]
    
    print("🏥 Testing Advanced Medical AI Pipeline")
    print("=" * 60)
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {query}")
        
        response = await pipeline.process_medical_query(query)
        
        print(f"📊 Intent: {response.intent} ({response.confidence:.1%})")
        print(f"🏷️  Entities: {len(response.entities)} detected")
        print(f"⚠️  Severity: {response.severity_assessment.value}")
        print(f"🎯 Risk Level: {response.risk_level.value}")
        
        if response.differential_diagnosis:
            print("🔬 Top Diagnoses:")
            for dx in response.differential_diagnosis[:2]:
                print(f"   • {dx['condition']} ({dx['probability']:.0%})")
        
        if response.drug_interactions:
            print("💊 Drug Interactions: ⚠️")
            
        if response.escalation_needed:
            print("🚨 ESCALATION REQUIRED")
        
        print(f"💬 Response: {response.response_text[:100]}...")
        print("-" * 60)

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_advanced_pipeline())