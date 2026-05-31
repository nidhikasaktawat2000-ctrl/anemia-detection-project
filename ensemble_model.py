"""Ensemble Model for Real-World Anemia Detection
===============================================
Combines:
1. Classical ML (XGBoost on hand-crafted features)
2. Deep Learning (EfficientNet-B0)
3. Demographic-aware thresholding
4. Consensus voting with confidence calibration
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
from typing import Dict, Tuple, Optional
import json
from pathlib import Path
from PIL import Image


class EnsembleAnemiaDetector:
    """Production-grade ensemble for real-world robustness."""
    
    def __init__(self, classical_model_path: str, dl_model_path: str):
        """Load both classical and DL models."""
        self.device = torch.device("cpu")
        
        # Load classical model
        self.classical_artifact = joblib.load(classical_model_path)
        self.classical_model = self.classical_artifact["model"]
        self.scaler = self.classical_artifact["scaler"]
        self.imputer = self.classical_artifact["imputer"]
        self.classes = self.classical_artifact.get("classes", ["anemia", "non_anemia"])
        
        # Load DL model
        self.dl_model = self._load_dl_model(dl_model_path)
        self.dl_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_dl_model(self, model_path: str):
        """Load EfficientNet-B0 model."""
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1)
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def predict_with_uncertainty(self, image: np.ndarray, 
                                age_category: str, gender: str,
                                region_type: str = "nail") -> Dict:
        """Predict anemia with uncertainty quantification."""
        
        # Get individual predictions
        classical_prob, classical_conf = self._predict_classical(image)
        dl_prob, dl_conf = self._predict_dl(image)
        
        # Ensemble combination with weighting
        ensemble_prob = self._combine_predictions(
            classical_prob, dl_prob,
            age_category, gender, region_type
        )
        
        # Get dynamic threshold based on demographics
        threshold = self._get_demographic_threshold(age_category, gender, region_type)
        
        # Make decision
        is_anemic = ensemble_prob >= threshold
        confidence = max(ensemble_prob, 1 - ensemble_prob)
        uncertainty = self._compute_uncertainty(
            classical_prob, dl_prob, ensemble_prob
        )
        
        # Consistency check
        models_agree = (classical_prob >= 0.5) == (dl_prob >= 0.5)
        
        return {
            "final_prediction": "Anemic" if is_anemic else "Normal",
            "anemia_probability": round(float(ensemble_prob), 3),
            "confidence": round(float(confidence), 3),
            "uncertainty": round(float(uncertainty), 3),
            "models_agree": models_agree,
            "recommended_confidence_level": self._interpret_confidence(confidence, uncertainty),
            "ensemble_scores": {
                "classical_model": round(float(classical_prob), 3),
                "deep_learning_model": round(float(dl_prob), 3),
                "ensemble_consensus": round(float(ensemble_prob), 3),
            },
            "individual_confidence": {
                "classical": round(float(classical_conf), 3),
                "deep_learning": round(float(dl_conf), 3),
            },
            "decision_threshold": round(float(threshold), 3),
            "recommended_action": self._recommend_action(
                is_anemic, confidence, uncertainty, age_category
            )
        }
    
    def _predict_classical(self, image: np.ndarray) -> Tuple[float, float]:
        """Predict using classical ML model."""
        from features import image_to_features
        
        # Extract features
        feat = image_to_features(image, (128, 128))
        feat_scaled = self.scaler.transform(
            self.imputer.transform(feat.reshape(1, -1))
        )
        
        # Predict
        proba = self.classical_model.predict_proba(feat_scaled)[0]
        anemia_idx = self.classes.index("anemia")
        anemia_prob = float(proba[anemia_idx])
        
        confidence = max(anemia_prob, 1 - anemia_prob)
        return anemia_prob, confidence
    
    def _predict_dl(self, image: np.ndarray) -> Tuple[float, float]:
        """Predict using deep learning model."""
        # Convert numpy array to tensor and apply transforms
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image_pil = image
        
        img_tensor = self.dl_transforms(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.dl_model(img_tensor)
            anemia_prob = torch.sigmoid(output).item()
        
        confidence = max(anemia_prob, 1 - anemia_prob)
        return anemia_prob, confidence
    
    def _combine_predictions(self, classical_prob: float, dl_prob: float,
                            age_category: str, gender: str, 
                            region_type: str) -> float:
        """Combine predictions using adaptive weighting."""
        
        w_classical = 0.35
        w_dl = 0.65
        
        # Adjust based on agreement
        agreement = 1.0 - abs(classical_prob - dl_prob)
        
        if agreement < 0.3:
            # Models disagree - weight the more confident one
            classical_conf = max(classical_prob, 1 - classical_prob)
            dl_conf = max(dl_prob, 1 - dl_prob)
            
            if classical_conf > dl_conf:
                w_classical = 0.50
                w_dl = 0.50
        
        # Adjust for region type
        if region_type in ["nail", "palm"]:
            w_dl = min(0.70, w_dl + 0.05)
            w_classical = 1.0 - w_dl
        
        ensemble_prob = w_classical * classical_prob + w_dl * dl_prob
        return ensemble_prob
    
    def _get_demographic_threshold(self, age_category: str, gender: str,
                                  region_type: str) -> float:
        """Get demographic-customized decision threshold."""
        
        base_threshold = 0.50
        
        # Age-based adjustment
        age_adjustments = {
            "Newborn (0-1 yr)": -0.05,
            "Infant (1-5 yrs)": -0.03,
            "Child (6-12 yrs)": -0.02,
            "Adolescent (13-18 yrs)": 0.00,
            "Young Adult (19-30 yrs)": 0.00,
            "Adult (31-45 yrs)": 0.02,
            "Middle-Aged (46-60 yrs)": 0.03,
        }
        
        age_adjustment = age_adjustments.get(age_category, 0.0)
        threshold = base_threshold + age_adjustment
        
        # Body part adjustments
        if region_type == "nail":
            threshold += 0.08
        elif region_type == "eye":
            threshold += 0.00
        elif region_type == "palm":
            threshold += 0.05
        
        # Clinical safety: lower threshold for sensitivity
        sensitivity_margin = 0.05
        threshold = max(0.35, threshold - sensitivity_margin)
        
        return np.clip(float(threshold), 0.35, 0.75)
    
    def _compute_uncertainty(self, classical_prob: float, dl_prob: float,
                            ensemble_prob: float) -> float:
        """Quantify prediction uncertainty."""
        
        # Model disagreement
        model_disagreement = abs(classical_prob - dl_prob)
        disagreement_uncertainty = model_disagreement * 0.5
        
        # Proximity to decision boundary
        boundary_proximity = min(
            abs(ensemble_prob - 0.5) / 0.5,
            abs(ensemble_prob - 0.35) / 0.35,
            abs(ensemble_prob - 0.75) / 0.25
        )
        boundary_uncertainty = (1.0 - boundary_proximity) * 0.3
        
        # Individual model confidence
        classical_conf = max(classical_prob, 1 - classical_prob)
        dl_conf = max(dl_prob, 1 - dl_prob)
        avg_conf = (classical_conf + dl_conf) / 2.0
        confidence_uncertainty = (1.0 - avg_conf) * 0.2
        
        total_uncertainty = (disagreement_uncertainty + 
                            boundary_uncertainty + 
                            confidence_uncertainty)
        
        return np.clip(float(total_uncertainty), 0.0, 1.0)
    
    def _interpret_confidence(self, confidence: float, uncertainty: float) -> str:
        """Interpret confidence level."""
        
        if uncertainty > 0.4:
            return "LOW - Inconclusive (recommend re-imaging)"
        
        if confidence < 0.60:
            return "MODERATE - Use with caution"
        
        if confidence < 0.80:
            return "HIGH - Reliable"
        
        return "VERY HIGH - High confidence"
    
    def _recommend_action(self, is_anemic: bool, confidence: float,
                         uncertainty: float, age_category: str) -> str:
        """Clinical recommendation."""
        
        if is_anemic:
            if confidence > 0.85 and uncertainty < 0.2:
                return "🔴 ANEMIA DETECTED - Immediate medical consultation"
            elif confidence > 0.70:
                return "🟡 ANEMIA LIKELY - Schedule consultation"
            else:
                return "🟠 POSSIBLE ANEMIA - Recommend re-imaging"
        else:
            if confidence > 0.85:
                return "✅ NO ANEMIA DETECTED - Monitor regularly"
            else:
                return "⚠️  INCONCLUSIVE - Repeat testing recommended"
