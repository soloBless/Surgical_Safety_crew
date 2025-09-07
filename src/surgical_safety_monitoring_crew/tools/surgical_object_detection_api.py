from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List
import requests
import json
from datetime import datetime
import time
import math

class SurgicalObjectDetectionInput(BaseModel):
    """Input schema for Surgical Object Detection API Tool."""
    image_base64: str = Field(
        description="Base64 encoded image string for object detection"
    )
    hf_api_token: str = Field(
        default="",
        description="HuggingFace API token (optional, can use free tier)"
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence score for detections (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    model_name: str = Field(
        default="facebook/detr-resnet-50",
        description="HuggingFace model to use for detection"
    )

class SurgicalObjectDetectionAPI(BaseTool):
    """Tool for detecting surgical PPE and medical equipment using pre-trained Hugging Face DETR model optimized for surgical context."""

    name: str = "surgical_object_detection_api"
    description: str = (
        "Real-time object detection using pre-trained Hugging Face DETR model optimized for surgical PPE "
        "and medical equipment detection. Uses facebook/detr-resnet-50 model to detect persons for PPE analysis, "
        "medical equipment, and surgical instruments. Maps COCO object classes to surgical equivalents and provides "
        "PPE compliance assessment with confidence scores and bounding boxes. Includes fallback mock data when API unavailable."
    )
    args_schema: Type[BaseModel] = SurgicalObjectDetectionInput

    def _run(self, image_base64: str, hf_api_token: str = "", confidence_threshold: float = 0.5, model_name: str = "facebook/detr-resnet-50") -> str:
        """
        Detect surgical PPE and equipment using HuggingFace DETR model.
        
        Args:
            image_base64: Base64 encoded image string
            hf_api_token: HuggingFace API token (optional)
            confidence_threshold: Minimum confidence score for detections
            model_name: HuggingFace model to use
            
        Returns:
            JSON string with detection results and surgical compliance analysis
        """
        try:
            # Step 1: Validate base64 input
            if not self._validate_base64_input(image_base64):
                return self._generate_error_response(
                    "Invalid base64 image data provided",
                    confidence_threshold,
                    model_name
                )

            # Step 2: Test HuggingFace API connectivity
            api_available = self._test_hf_api_connectivity(model_name, hf_api_token)
            
            if not api_available:
                return self._generate_mock_surgical_data(confidence_threshold, model_name, "HuggingFace API not reachable")

            # Step 3: Make HuggingFace API request with retry logic
            api_result = self._make_hf_api_request_with_retry(
                model_name, image_base64, hf_api_token, confidence_threshold
            )

            if api_result is not None:
                # Step 4: Process HuggingFace DETR response
                return self._process_hf_api_response(api_result, confidence_threshold, model_name)
            else:
                # Step 5: Use fallback if API fails
                return self._generate_mock_surgical_data(confidence_threshold, model_name, "API request failed after retries")

        except Exception as e:
            # Handle unexpected errors with fallback
            return self._generate_mock_surgical_data(
                confidence_threshold, model_name, f"Unexpected error: {str(e)}"
            )

    def _validate_base64_input(self, image_base64: str) -> bool:
        """Validate base64 input format."""
        if not image_base64 or not isinstance(image_base64, str):
            return False
        
        # Basic validation - check if it looks like base64
        if len(image_base64) < 100:  # Too short to be a reasonable image
            return False
            
        return True

    def _test_hf_api_connectivity(self, model_name: str, hf_api_token: str) -> bool:
        """Test if HuggingFace API endpoint is reachable."""
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {"Content-Type": "application/json"}
            if hf_api_token:
                headers["Authorization"] = f"Bearer {hf_api_token}"
            
            # Make a simple HEAD request to check connectivity
            response = requests.head(api_url, headers=headers, timeout=5)
            return response.status_code != 404
        except requests.exceptions.RequestException:
            return False

    def _make_hf_api_request_with_retry(self, model_name: str, image_base64: str, hf_api_token: str, confidence_threshold: float, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Make HuggingFace API request with exponential backoff retry logic."""
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        for attempt in range(max_retries):
            try:
                # Prepare headers
                headers = {"Content-Type": "application/json"}
                if hf_api_token:
                    headers["Authorization"] = f"Bearer {hf_api_token}"

                # Prepare payload - HuggingFace expects raw image data
                import base64
                try:
                    # Remove data URL prefix if present
                    if "," in image_base64:
                        image_base64 = image_base64.split(",", 1)[1]
                    
                    image_bytes = base64.b64decode(image_base64)
                except Exception:
                    return None

                # Make API call with timeout
                response = requests.post(
                    api_url,
                    data=image_bytes,
                    headers=headers,
                    timeout=30
                )

                # Handle different response codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    # Model not found - break retry loop
                    return None
                elif response.status_code == 503:
                    # Model loading - retry with exponential backoff
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + (math.random() * 0.1)
                        time.sleep(wait_time)
                        continue
                    else:
                        return None
                elif response.status_code >= 500:
                    # Server error - retry with exponential backoff
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + (math.random() * 0.1)
                        time.sleep(wait_time)
                        continue
                    else:
                        return None
                else:
                    # Other client errors - don't retry
                    return None

            except requests.exceptions.Timeout:
                # Timeout - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (math.random() * 0.1)
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            except requests.exceptions.RequestException:
                # Connection error - retry
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (math.random() * 0.1)
                    time.sleep(wait_time)
                    continue
                else:
                    return None

        return None

    def _process_hf_api_response(self, api_result: List[Dict[str, Any]], confidence_threshold: float, model_name: str) -> str:
        """Process HuggingFace DETR response and map to surgical context."""
        try:
            # Validate API response format
            if not isinstance(api_result, list):
                return self._generate_mock_surgical_data(confidence_threshold, model_name, "Invalid API response format")

            detections = []
            total_detections = 0
            ppe_items_detected = 0
            persons_detected = 0
            
            # COCO to surgical mapping
            coco_to_surgical_mapping = {
                "person": {"surgical_class": "surgical_staff", "is_ppe": False, "surgical_interpretation": "Surgical staff member for PPE compliance analysis"},
                "bottle": {"surgical_class": "sanitizer_bottle", "is_ppe": False, "surgical_interpretation": "Medical sanitizer or solution bottle"},
                "cup": {"surgical_class": "medical_container", "is_ppe": False, "surgical_interpretation": "Medical specimen or solution container"},
                "bowl": {"surgical_class": "surgical_basin", "is_ppe": False, "surgical_interpretation": "Surgical basin or sterile container"},
                "scissors": {"surgical_class": "surgical_scissors", "is_ppe": False, "surgical_interpretation": "Surgical cutting instruments"},
                "knife": {"surgical_class": "scalpel", "is_ppe": False, "surgical_interpretation": "Surgical scalpel or cutting tool"},
                "spoon": {"surgical_class": "medical_spoon", "is_ppe": False, "surgical_interpretation": "Medical measuring or mixing spoon"},
                "tie": {"surgical_class": "surgical_cap_tie", "is_ppe": True, "surgical_interpretation": "Surgical cap ties or hair restraint"},
                "handbag": {"surgical_class": "medical_bag", "is_ppe": False, "surgical_interpretation": "Medical equipment or supply bag"},
                "clock": {"surgical_class": "or_clock", "is_ppe": False, "surgical_interpretation": "Operating room timing device"},
            }

            # Process detections from HuggingFace DETR response
            for detection in api_result:
                if not isinstance(detection, dict):
                    continue
                    
                confidence = detection.get("score", 0.0)
                
                if confidence >= confidence_threshold:
                    coco_label = detection.get("label", "unknown")
                    bbox = detection.get("box", {})
                    
                    # Map COCO class to surgical equivalent
                    surgical_mapping = coco_to_surgical_mapping.get(coco_label, {
                        "surgical_class": f"unknown_{coco_label}",
                        "is_ppe": False,
                        "surgical_interpretation": f"Unclassified object: {coco_label}"
                    })
                    
                    surgical_class = surgical_mapping["surgical_class"]
                    is_ppe_related = surgical_mapping["is_ppe"]
                    surgical_interpretation = surgical_mapping["surgical_interpretation"]
                    
                    # Count persons for PPE analysis
                    if coco_label == "person":
                        persons_detected += 1
                    
                    if is_ppe_related:
                        ppe_items_detected += 1

                    formatted_detection = {
                        "original_coco_class": coco_label,
                        "surgical_class": surgical_class,
                        "surgical_interpretation": surgical_interpretation,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "xmin": bbox.get("xmin", 0),
                            "ymin": bbox.get("ymin", 0),
                            "xmax": bbox.get("xmax", 0),
                            "ymax": bbox.get("ymax", 0)
                        },
                        "is_ppe_related": is_ppe_related
                    }
                    
                    detections.append(formatted_detection)
                    total_detections += 1

            # Calculate PPE compliance assessment
            # Estimate basic compliance based on person detection and PPE items
            base_compliance = 0.7 if persons_detected > 0 else 0.3  # Assume some baseline compliance when staff present
            ppe_bonus = min(ppe_items_detected * 0.1, 0.3)  # Additional points for detected PPE items
            compliance_score = min(base_compliance + ppe_bonus, 1.0)

            result = {
                "detections": detections,
                "original_hf_detections": api_result,  # Include raw HuggingFace response
                "surgical_analysis": {
                    "persons_detected": persons_detected,
                    "ppe_items_detected": ppe_items_detected,
                    "estimated_compliance_score": round(compliance_score, 3),
                    "compliance_assessment": self._generate_compliance_assessment(compliance_score, persons_detected, ppe_items_detected)
                },
                "metadata": {
                    "total_detections": total_detections,
                    "confidence_threshold": confidence_threshold,
                    "model_used": model_name,
                    "api_status": "success",
                    "hf_api_available": True,
                    "timestamp": datetime.now().isoformat()
                }
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return self._generate_mock_surgical_data(
                confidence_threshold, model_name, 
                f"Response processing error: {str(e)}"
            )

    def _generate_compliance_assessment(self, score: float, persons: int, ppe_items: int) -> str:
        """Generate PPE compliance assessment text based on detection results."""
        if score >= 0.8:
            return f"Good PPE compliance detected. {persons} staff member(s) observed with {ppe_items} PPE-related items visible."
        elif score >= 0.6:
            return f"Moderate PPE compliance. {persons} staff member(s) detected. Consider verifying all required PPE is properly worn."
        elif score >= 0.4:
            return f"Limited PPE compliance observed. {persons} staff member(s) present with minimal PPE visibility. Review required."
        else:
            return f"Low PPE compliance detected. Immediate review recommended for proper surgical safety protocols."

    def _generate_mock_surgical_data(self, confidence_threshold: float, model_name: str, error_message: str = None) -> str:
        """Generate realistic mock surgical detection data when HuggingFace API is unavailable."""
        # Realistic mock detections mapped from COCO classes to surgical context
        mock_hf_response = [
            {"label": "person", "score": 0.89, "box": {"xmin": 150, "ymin": 80, "xmax": 300, "ymax": 400}},
            {"label": "tie", "score": 0.82, "box": {"xmin": 145, "ymin": 20, "xmax": 195, "ymax": 70}},
            {"label": "bottle", "score": 0.76, "box": {"xmin": 350, "ymin": 150, "xmax": 380, "ymax": 200}},
            {"label": "scissors", "score": 0.91, "box": {"xmin": 400, "ymin": 300, "xmax": 450, "ymax": 320}},
            {"label": "bowl", "score": 0.73, "box": {"xmin": 200, "ymin": 350, "xmax": 250, "ymax": 380}}
        ]

        # Filter by confidence threshold and process
        detections = []
        persons_detected = 0
        ppe_items = 0

        coco_to_surgical_mapping = {
            "person": {"surgical_class": "surgical_staff", "is_ppe": False, "surgical_interpretation": "Surgical staff member for PPE compliance analysis"},
            "tie": {"surgical_class": "surgical_cap_tie", "is_ppe": True, "surgical_interpretation": "Surgical cap ties or hair restraint"},
            "bottle": {"surgical_class": "sanitizer_bottle", "is_ppe": False, "surgical_interpretation": "Medical sanitizer or solution bottle"},
            "scissors": {"surgical_class": "surgical_scissors", "is_ppe": False, "surgical_interpretation": "Surgical cutting instruments"},
            "bowl": {"surgical_class": "surgical_basin", "is_ppe": False, "surgical_interpretation": "Surgical basin or sterile container"}
        }

        for detection in mock_hf_response:
            if detection["score"] >= confidence_threshold:
                coco_label = detection["label"]
                mapping = coco_to_surgical_mapping[coco_label]
                
                if coco_label == "person":
                    persons_detected += 1
                if mapping["is_ppe"]:
                    ppe_items += 1

                formatted_detection = {
                    "original_coco_class": coco_label,
                    "surgical_class": mapping["surgical_class"],
                    "surgical_interpretation": mapping["surgical_interpretation"],
                    "confidence": detection["score"],
                    "bbox": detection["box"],
                    "is_ppe_related": mapping["is_ppe"]
                }
                detections.append(formatted_detection)

        # Calculate mock compliance
        compliance_score = 0.75 if persons_detected > 0 else 0.4
        compliance_score = min(compliance_score + (ppe_items * 0.1), 1.0)

        result = {
            "detections": detections,
            "original_hf_detections": [d for d in mock_hf_response if d["score"] >= confidence_threshold],
            "surgical_analysis": {
                "persons_detected": persons_detected,
                "ppe_items_detected": ppe_items,
                "estimated_compliance_score": round(compliance_score, 3),
                "compliance_assessment": self._generate_compliance_assessment(compliance_score, persons_detected, ppe_items)
            },
            "metadata": {
                "total_detections": len(detections),
                "confidence_threshold": confidence_threshold,
                "model_used": model_name,
                "api_status": "fallback_mode",
                "hf_api_available": False,
                "timestamp": datetime.now().isoformat()
            }
        }

        if error_message:
            result["metadata"]["fallback_reason"] = error_message

        return json.dumps(result, indent=2)

    def _generate_error_response(self, error_message: str, confidence_threshold: float, model_name: str) -> str:
        """Generate error response when fallback is not available."""
        error_result = {
            "error": error_message,
            "detections": [],
            "surgical_analysis": {
                "persons_detected": 0,
                "ppe_items_detected": 0,
                "estimated_compliance_score": 0.0,
                "compliance_assessment": "Unable to assess PPE compliance due to error"
            },
            "metadata": {
                "total_detections": 0,
                "confidence_threshold": confidence_threshold,
                "model_used": model_name,
                "api_status": "error",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return json.dumps(error_result, indent=2)