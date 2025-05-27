import cv2
import numpy as np
import mediapipe as mp
import math
import os
import glob
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostureType(Enum):
    """Enumeration for posture types"""
    FACE_UP = "face_up"
    FACE_DOWN = "face_down"
    NO_CHILD = "no_child"
    UNKNOWN = "unknown"

@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    posture: PostureType
    confidence: float
    analysis_data: Dict[str, Any]
    landmarks_detected: bool
    processing_time: float

@dataclass
class DetectionConfig:
    """Configuration for detection parameters"""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 2
    nose_shoulder_threshold: float = 0.08
    nose_hip_threshold: float = 0.05
    shoulder_width_threshold: float = 0.12
    hip_width_threshold: float = 0.12
    ear_visibility_high: float = 0.7
    ear_visibility_low: float = 0.3

class ChildPostureDetector:
    """Enhanced child posture detection system"""
    
    def __init__(self, config: DetectionConfig = None):
        """Initialize the detector with configuration"""
        self.config = config or DetectionConfig()
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=self.config.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Color mapping for different postures
        self.colors = {
            PostureType.FACE_UP: (0, 255, 0),      # Green - Face up
            PostureType.FACE_DOWN: (0, 0, 255),    # Red - Face down
            PostureType.NO_CHILD: (128, 128, 128), # Gray - No child
            PostureType.UNKNOWN: (0, 255, 255)     # Yellow - Unknown
        }
        
        # Status text mapping
        self.status_text = {
            PostureType.FACE_UP: "Face Up",
            PostureType.FACE_DOWN: "Face Down",
            PostureType.NO_CHILD: "No Child Detected",
            PostureType.UNKNOWN: "Unknown Posture"
        }
        
        # Statistics tracking
        self.processing_stats = {
            'total_processed': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'posture_counts': {posture.value: 0 for posture in PostureType}
        }

    def calculate_angle(self, point1, point2, point3) -> float:
        """Calculate angle between three points"""
        try:
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        except Exception as e:
            logger.warning(f"Error calculating angle: {e}")
            return 0.0

    def extract_key_points(self, landmarks) -> Dict[str, Any]:
        """Extract key anatomical points from landmarks"""
        if not landmarks:
            return {}
        
        try:
            # Get important landmarks
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Get wrist landmarks for arm position analysis
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate derived measurements
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            body_center_y = (shoulder_center_y + hip_center_y) / 2
            
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            hip_width = abs(left_hip.x - right_hip.x)
            
            ear_visibility = (left_ear.visibility + right_ear.visibility) / 2
            
            return {
                'nose': nose,
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder,
                'left_hip': left_hip,
                'right_hip': right_hip,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'left_wrist': left_wrist,
                'right_wrist': right_wrist,
                'shoulder_center_y': shoulder_center_y,
                'hip_center_y': hip_center_y,
                'body_center_y': body_center_y,
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'ear_visibility': ear_visibility,
                'nose_to_shoulder_diff': nose.y - shoulder_center_y,
                'nose_to_hip_diff': nose.y - hip_center_y
            }
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return {}

    def analyze_view_type(self, key_points: Dict[str, Any]) -> str:
        """Determine the viewing angle/perspective"""
        shoulder_width = key_points.get('shoulder_width', 0)
        hip_width = key_points.get('hip_width', 0)
        
        if shoulder_width < self.config.shoulder_width_threshold and hip_width < self.config.hip_width_threshold:
            return 'profile'
        elif shoulder_width > 0.18 or hip_width > 0.18:
            return 'top_bottom'
        else:
            return 'angled'

    def calculate_body_orientation_features(self, key_points: Dict[str, Any]) -> Dict[str, float]:
        """Calculate advanced body orientation features"""
        features = {}
        
        # Calculate shoulder-hip alignment (crucial for tummy time detection)
        left_shoulder = key_points.get('left_shoulder')
        right_shoulder = key_points.get('right_shoulder')
        left_hip = key_points.get('left_hip')
        right_hip = key_points.get('right_hip')
        nose = key_points.get('nose')
        
        if all([left_shoulder, right_shoulder, left_hip, right_hip, nose]):
            # Calculate torso angle (angle between shoulder line and hip line)
            shoulder_slope = (right_shoulder.y - left_shoulder.y) / max(abs(right_shoulder.x - left_shoulder.x), 0.001)
            hip_slope = (right_hip.y - left_hip.y) / max(abs(right_hip.x - left_hip.x), 0.001)
            features['torso_alignment'] = abs(shoulder_slope - hip_slope)
            
            # Calculate body aspect ratio (width vs height)
            body_width = max(key_points['shoulder_width'], key_points['hip_width'])
            body_height = abs(key_points['shoulder_center_y'] - key_points['hip_center_y'])
            features['aspect_ratio'] = body_width / max(body_height, 0.01)
            
            # Head elevation relative to body center
            body_center_y = (key_points['shoulder_center_y'] + key_points['hip_center_y']) / 2
            features['head_elevation'] = body_center_y - nose.y
            
            # Shoulder-hip vertical distance (compressed in face-down position)
            features['torso_compression'] = body_height
            
        return features

    def detect_tummy_time_indicators(self, key_points: Dict[str, Any], features: Dict[str, float]) -> float:
        """Detect specific indicators of tummy time (face-down with head lifted)"""
        tummy_time_score = 0.0
        
        # Key indicator 1: Head significantly elevated above body center while body is horizontal
        head_elevation = features.get('head_elevation', 0)
        aspect_ratio = features.get('aspect_ratio', 0)
        
        if head_elevation > 0.05 and aspect_ratio > 2.0:  # Head up, body wide (horizontal)
            tummy_time_score += 3.0
        
        # Key indicator 2: Compressed torso height (typical of lying flat)
        torso_compression = features.get('torso_compression', 0)
        if torso_compression < 0.15:  # Very compressed torso
            tummy_time_score += 2.0
        
        # Key indicator 3: Arms positioned below shoulders (supporting posture)
        left_wrist = key_points.get('left_wrist') if 'left_wrist' in key_points else None
        right_wrist = key_points.get('right_wrist') if 'right_wrist' in key_points else None
        left_shoulder = key_points.get('left_shoulder')
        right_shoulder = key_points.get('right_shoulder')
        
        arm_support_score = 0
        if left_wrist and left_shoulder and left_wrist.y > left_shoulder.y:
            arm_support_score += 1
        if right_wrist and right_shoulder and right_wrist.y > right_shoulder.y:
            arm_support_score += 1
            
        if arm_support_score > 0:
            tummy_time_score += arm_support_score
        
        # Key indicator 4: Face visible but body horizontal (classic tummy time)
        face_visible = key_points.get('ear_visibility', 0) > 0.5
        body_horizontal = aspect_ratio > 1.8
        
        if face_visible and body_horizontal and head_elevation > 0.03:
            tummy_time_score += 2.5
        
        return tummy_time_score

    def calculate_posture_scores(self, key_points: Dict[str, Any]) -> Tuple[float, float]:
        """Enhanced posture scoring with tummy time detection"""
        face_up_score = 0.0
        face_down_score = 0.0
        
        nose_to_shoulder_diff = key_points.get('nose_to_shoulder_diff', 0)
        nose_to_hip_diff = key_points.get('nose_to_hip_diff', 0)
        shoulder_width = key_points.get('shoulder_width', 0)
        hip_width = key_points.get('hip_width', 0)
        ear_visibility = key_points.get('ear_visibility', 0)
        
        # Calculate advanced body orientation features
        features = self.calculate_body_orientation_features(key_points)
        
        # Detect tummy time indicators
        tummy_time_score = self.detect_tummy_time_indicators(key_points, features)
        
        # If strong tummy time indicators are present, heavily favor face-down
        if tummy_time_score >= 3.0:
            face_down_score += tummy_time_score
            # Add detailed reasoning
            key_points['tummy_time_detected'] = True
            key_points['tummy_time_score'] = tummy_time_score
        
        # Traditional scoring with modifications
        # Score based on nose position relative to shoulders (less weight if tummy time detected)
        weight = 1.0 if tummy_time_score < 2.0 else 0.5
        if nose_to_shoulder_diff < -0.03:
            face_up_score += 2.0 * weight
        elif nose_to_shoulder_diff > 0.03:
            face_down_score += 2.0 * weight
        
        # Body width analysis with tummy time context
        avg_width = (shoulder_width + hip_width) / 2
        aspect_ratio = features.get('aspect_ratio', avg_width / 0.1)
        
        if aspect_ratio > 2.5:  # Very wide body = likely horizontal position
            if features.get('head_elevation', 0) > 0.02:  # Head elevated
                face_down_score += 2.0  # Likely tummy time
            else:
                face_up_score += 1.5  # Likely lying on back
        elif avg_width > 0.15:
            face_up_score += 1.0
        else:
            face_down_score += 0.5
        
        # Ear visibility analysis with context
        if ear_visibility > self.config.ear_visibility_high:
            if features.get('head_elevation', 0) > 0.03:
                # Face visible but head elevated = likely tummy time
                face_down_score += 1.5
            else:
                face_up_score += 1.0
        elif ear_visibility < self.config.ear_visibility_low:
            face_down_score += 1.0
        
        # Nose-hip relationship
        if nose_to_hip_diff < -0.02:
            face_up_score += 1.0
        elif nose_to_hip_diff > 0.02:
            face_down_score += 1.0
        
        # Torso compression factor
        if features.get('torso_compression', 1.0) < 0.12:
            face_down_score += 1.5  # Very flat = likely lying down
        
        return face_up_score, face_down_score

    def detect_posture(self, landmarks, image_shape) -> Tuple[PostureType, Dict[str, Any]]:
        """Advanced posture detection with comprehensive analysis"""
        if not landmarks:
            return PostureType.NO_CHILD, {'reason': 'no_landmarks'}
        
        # Extract key anatomical points
        key_points = self.extract_key_points(landmarks)
        if not key_points:
            return PostureType.NO_CHILD, {'reason': 'failed_key_point_extraction'}
        
        # Check if enough key points are visible
        key_landmarks = [key_points['nose'], key_points['left_shoulder'], 
                        key_points['right_shoulder'], key_points['left_hip'], key_points['right_hip']]
        visible_points = [point for point in key_landmarks if point.visibility > 0.5]
        
        if len(visible_points) < 3:
            return PostureType.NO_CHILD, {'reason': 'insufficient_visible_points', 'visible_count': len(visible_points)}
        
        # Determine view type
        view_type = self.analyze_view_type(key_points)
        
        # Initialize analysis data
        analysis_data = {
            'view_type': view_type,
            'visible_points': len(visible_points),
            'key_measurements': {
                'nose_to_shoulder_diff': key_points['nose_to_shoulder_diff'],
                'nose_to_hip_diff': key_points['nose_to_hip_diff'],
                'shoulder_width': key_points['shoulder_width'],
                'hip_width': key_points['hip_width'],
                'ear_visibility': key_points['ear_visibility']
            }
        }
        
        # Profile view analysis
        if view_type == 'profile':
            nose_to_shoulder_diff = key_points['nose_to_shoulder_diff']
            nose_to_hip_diff = key_points['nose_to_hip_diff']
            
            if nose_to_shoulder_diff < -self.config.nose_shoulder_threshold:
                return PostureType.FACE_UP, analysis_data
            elif nose_to_hip_diff > self.config.nose_hip_threshold:
                return PostureType.FACE_DOWN, analysis_data
        
        # Top/bottom view analysis
        elif view_type == 'top_bottom':
            ear_visibility = key_points['ear_visibility']
            nose_to_shoulder_diff = key_points['nose_to_shoulder_diff']
            
            if ear_visibility > self.config.ear_visibility_high and nose_to_shoulder_diff < 0:
                return PostureType.FACE_UP, analysis_data
            elif ear_visibility < self.config.ear_visibility_low and nose_to_shoulder_diff > 0:
                return PostureType.FACE_DOWN, analysis_data
        
        # Angled view analysis using scoring system
        face_up_score, face_down_score = self.calculate_posture_scores(key_points)
        
        analysis_data['scores'] = {
            'face_up_score': face_up_score,
            'face_down_score': face_down_score
        }
        
        score_difference = abs(face_up_score - face_down_score)
        if score_difference >= 1.0:  # Minimum confidence threshold
            if face_up_score > face_down_score:
                return PostureType.FACE_UP, analysis_data
            else:
                return PostureType.FACE_DOWN, analysis_data
        
        return PostureType.UNKNOWN, analysis_data

    def process_image(self, image_path: str) -> Optional[AnalysisResult]:
        """Process a single image and return analysis results"""
        start_time = datetime.now()
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot read image: {image_path}")
                return None
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_image)
            
            # Detect posture
            posture, analysis_data = self.detect_posture(results.pose_landmarks, image.shape)
            
            # Calculate confidence
            confidence = self._calculate_confidence(analysis_data)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._update_stats(posture, results.pose_landmarks is not None)
            
            return AnalysisResult(
                posture=posture,
                confidence=confidence,
                analysis_data=analysis_data,
                landmarks_detected=results.pose_landmarks is not None,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def _calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis data"""
        base_confidence = 0.5
        
        if 'scores' in analysis_data:
            scores = analysis_data['scores']
            max_score = max(scores['face_up_score'], scores['face_down_score'])
            total_score = scores['face_up_score'] + scores['face_down_score']
            
            if total_score > 0:
                confidence = base_confidence + (max_score / total_score) * 0.5
                return min(confidence, 1.0)
        
        return base_confidence

    def _update_stats(self, posture: PostureType, landmarks_detected: bool):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        if landmarks_detected:
            self.processing_stats['successful_detections'] += 1
        else:
            self.processing_stats['failed_detections'] += 1
        
        self.processing_stats['posture_counts'][posture.value] += 1

    def create_annotated_image(self, image_path: str, result: AnalysisResult) -> Optional[np.ndarray]:
        """Create annotated image with pose landmarks and status overlay"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert for MediaPipe processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_image)
            
            # Draw pose landmarks if detected
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # Add status overlay
            annotated_image = self._add_status_overlay(image, result, os.path.basename(image_path))
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            return None

    def _add_status_overlay(self, image: np.ndarray, result: AnalysisResult, filename: str) -> np.ndarray:
        """Add comprehensive status overlay to image"""
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Get color and status text
        color = self.colors[result.posture]
        status_text = self.status_text[result.posture]
        
        # Create semi-transparent overlay
        overlay = image.copy()
        
        # Main status box
        box_height = 120
        cv2.rectangle(overlay, (10, 10), (min(500, w-10), box_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (min(500, w-10), box_height), color, 3)
        
        # Blend overlay
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Add main status text
        cv2.putText(image, status_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Add confidence
        confidence_text = f"Confidence: {result.confidence:.2f}"
        cv2.putText(image, confidence_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add processing time
        time_text = f"Time: {result.processing_time:.3f}s"
        cv2.putText(image, time_text, (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add filename
        cv2.putText(image, f"File: {filename}", (20, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add detailed analysis info with tummy time detection
        if 'view_type' in result.analysis_data:
            view_text = f"View: {result.analysis_data['view_type']}"
            cv2.putText(image, view_text, (20, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Special indicator for tummy time detection
        if result.analysis_data.get('key_measurements', {}).get('tummy_time_detected', False):
            tummy_score = result.analysis_data.get('key_measurements', {}).get('tummy_time_score', 0)
            tummy_text = f"TUMMY TIME detected (score: {tummy_score:.1f})"
            cv2.putText(image, tummy_text, (20, h-55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow highlight
        
        if 'scores' in result.analysis_data:
            scores = result.analysis_data['scores']
            score_text = f"Scores - Up: {scores['face_up_score']:.1f}, Down: {scores['face_down_score']:.1f}"
            cv2.putText(image, score_text, (20, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image

    def save_analysis_report(self, results: List[Tuple[str, AnalysisResult]], output_dir: str):
        """Save detailed analysis report as JSON"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'processing_stats': self.processing_stats,
            'results': []
        }
        
        for image_path, result in results:
            if result:
                result_data = {
                    'image_path': image_path,
                    'filename': os.path.basename(image_path),
                    'posture': result.posture.value,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    'landmarks_detected': result.landmarks_detected,
                    'analysis_data': result.analysis_data
                }
                report_data['results'].append(result_data)
        
        # Save report
        report_path = os.path.join(output_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report saved to: {report_path}")
        return report_path

    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()

    def reset_statistics(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'posture_counts': {posture.value: 0 for posture in PostureType}
        }

# Utility functions for batch processing
def process_single_image(image_path: str, detector: ChildPostureDetector, save_result: bool = True, output_dir: str = "results") -> Optional[Tuple[np.ndarray, AnalysisResult]]:
    """Process a single image with enhanced error handling and logging"""
    logger.info(f"Processing: {image_path}")
    
    # Process image
    result = detector.process_image(image_path)
    if not result:
        logger.error(f"Failed to process image: {image_path}")
        return None
    
    # Create annotated image
    annotated_image = detector.create_annotated_image(image_path, result)
    if annotated_image is None:
        logger.error(f"Failed to create annotated image: {image_path}")
        return None
    
    # Log results
    logger.info(f"Result: {result.posture.value} (confidence: {result.confidence:.2f})")
    
    # Save result if requested
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Saved result to: {output_path}")
    
    return annotated_image, result

def process_multiple_images(image_folder: str, detector: ChildPostureDetector, output_dir: str = "results") -> List[Tuple[str, AnalysisResult]]:
    """Process multiple images in a folder with comprehensive reporting"""
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    if not image_paths:
        logger.warning(f"No images found in folder: {image_folder}")
        return []
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process all images
    results = []
    successful_count = 0
    
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing image {i}/{len(image_paths)}")
        
        try:
            processed_result = process_single_image(image_path, detector, save_result=True, output_dir=output_dir)
            if processed_result:
                results.append((image_path, processed_result[1]))
                successful_count += 1
            else:
                results.append((image_path, None))
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append((image_path, None))
    
    # Generate comprehensive report
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE - SUMMARY REPORT")
    logger.info("="*60)
    logger.info(f"Total images processed: {len(image_paths)}")
    logger.info(f"Successful detections: {successful_count}")
    logger.info(f"Failed detections: {len(image_paths) - successful_count}")
    
    # Create output directory and save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = detector.save_analysis_report(results, output_dir)
    
    # Print posture statistics
    stats = detector.get_statistics()
    logger.info("\nPosture Distribution:")
    for posture_type, count in stats['posture_counts'].items():
        if count > 0:
            percentage = (count / stats['total_processed']) * 100
            logger.info(f"- {posture_type.replace('_', ' ').title()}: {count} images ({percentage:.1f}%)")
    
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    return results

def main():
    """Enhanced main function with improved user interface"""
    print("="*60)
    print("ADVANCED CHILD POSTURE DETECTION SYSTEM")
    print("="*60)
    print("Features:")
    print("- Enhanced pose detection with confidence scoring")
    print("- Multiple viewing angle analysis")
    print("- Comprehensive reporting and statistics")
    print("- Batch processing capabilities")
    print("="*60)
    
    # Initialize detector with default configuration
    config = DetectionConfig()
    detector = ChildPostureDetector(config)
    
    while True:
        print("\nSelect processing mode:")
        print("1. Process single image")
        print("2. Process multiple images (batch)")
        print("3. View processing statistics")
        print("4. Configure detection parameters")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                try:
                    result = process_single_image(image_path, detector)
                    if result:
                        print(f"\nDetection completed successfully!")
                        print(f"Posture: {result[1].posture.value}")
                        print(f"Confidence: {result[1].confidence:.2f}")
                        print(f"Processing time: {result[1].processing_time:.3f}s")
                        
                        # Display image
                        cv2.imshow('Detection Result', result[0])
                        print("\nPress any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print("Failed to process image!")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Image file not found!")
        
        elif choice == '2':
            folder_path = input("Enter folder path containing images: ").strip()
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                output_dir = input("Enter output directory (default: 'results'): ").strip() or "results"
                
                try:
                    results = process_multiple_images(folder_path, detector, output_dir)
                    print(f"\nBatch processing completed!")
                    print(f"Results saved in: {output_dir}")
                except Exception as e:
                    print(f"Error during batch processing: {e}")
            else:
                print("Folder not found!")
        
        elif choice == '3':
            stats = detector.get_statistics()
            print("\nProcessing Statistics:")
            print(f"Total images processed: {stats['total_processed']}")
            print(f"Successful detections: {stats['successful_detections']}")
            print(f"Failed detections: {stats['failed_detections']}")
            
            if stats['total_processed'] > 0:
                print("\nPosture Distribution:")
                for posture_type, count in stats['posture_counts'].items():
                    if count > 0:
                        percentage = (count / stats['total_processed']) * 100
                        print(f"- {posture_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
            reset_choice = input("\nReset statistics? (y/n): ").strip().lower()
            if reset_choice == 'y':
                detector.reset_statistics()
                print("Statistics reset!")
        
        elif choice == '4':
            print("\nCurrent Configuration:")
            print(f"Min detection confidence: {config.min_detection_confidence}")
            print(f"Min tracking confidence: {config.min_tracking_confidence}")
            print(f"Model complexity: {config.model_complexity}")
            print(f"Nose-shoulder threshold: {config.nose_shoulder_threshold}")
            print(f"Shoulder width threshold: {config.shoulder_width_threshold}")
            
            print("\nConfiguration modification not implemented in this demo.")
            print("Edit the DetectionConfig class to customize parameters.")
        
        elif choice == '5':
            print("\nThank you for using the Child Posture Detection System!")
            break
        
        else:
            print("Invalid choice! Please select 1-5.")

if __name__ == "__main__":
    main()