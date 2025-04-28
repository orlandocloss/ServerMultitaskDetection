# pip install ultralytics torchvision pillow numpy scikit-learn tabulate tqdm requests


import os
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet50
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
import time
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
import requests
import logging
import sys
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_multitask(species_list, test_set, yolo_weights, hierarchical_weights, output_dir="."):
    """
    Run the two-stage classifier on a test set.
    
    Args:
        species_list (list): List of species names used for training
        test_set (str): Path to the test directory
        yolo_weights (str): Path to the YOLO model file
        hierarchical_weights (str): Path to the hierarchical classifier model file
        output_dir (str): Directory to save output CSV files (default: current directory)
    
    Returns:
        Results from the classifier
    """
    classifier = TwoStage(yolo_weights, hierarchical_weights, species_list, output_dir)
    results, output_file = classifier.run(test_set)
    print("Testing complete with metrics calculated at all taxonomic levels")
    return results, output_file

def cuda_cleanup():
    """Clear CUDA cache and reset device"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
def setup_gpu():
    """Set up GPU with better error handling and reporting"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available on this system")
        return torch.device("cpu")
    
    try:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA device(s)")
        
        for i in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_properties.name} with {gpu_properties.total_memory / 1e9:.2f} GB memory")
        
        device = torch.device("cuda:0")
        test_tensor = torch.ones(1, device=device)
        test_result = test_tensor * 2
        del test_tensor, test_result
        
        logger.info("CUDA initialization successful")
        return device
    except Exception as e:
        logger.error(f"CUDA initialization error: {str(e)}")
        logger.warning("Falling back to CPU")
        return torch.device("cpu")

class HierarchicalInsectClassifier(nn.Module):
    def __init__(self, num_classes_per_level):
        """
        Args:
            num_classes_per_level (list): Number of classes for each taxonomic level
        """
        super(HierarchicalInsectClassifier, self).__init__()
        
        self.backbone = resnet50(pretrained=True)
        backbone_output_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        
        self.branches = nn.ModuleList()
        for num_classes in num_classes_per_level:
            branch = nn.Sequential(
                nn.Linear(backbone_output_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            self.branches.append(branch)
        
        self.num_levels = len(num_classes_per_level)
        
        self.register_buffer('class_means', torch.zeros(sum(num_classes_per_level)))
        self.register_buffer('class_stds', torch.ones(sum(num_classes_per_level)))
        self.class_counts = [0] * sum(num_classes_per_level)
        self.output_history = defaultdict(list)
        
    def forward(self, x):
        R0 = self.backbone(x)
        
        outputs = []
        for branch in self.branches:
            outputs.append(branch(R0))
            
        return outputs

def get_taxonomy(species_list):
    """
    Retrieves taxonomic information for a list of species from GBIF API.
    Creates a hierarchical taxonomy dictionary with family, genus, and species relationships.
    """
    taxonomy = {1: [], 2: {}, 3: {}}
    species_to_genus = {}
    genus_to_family = {}
    
    logger.info(f"Building taxonomy from GBIF for {len(species_list)} species")
    
    print("\nTaxonomy Results:")
    print("-" * 80)
    print(f"{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
    print("-" * 80)
    
    for species_name in species_list:
        url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
        try:
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') == 'ACCEPTED' or data.get('status') == 'SYNONYM':
                family = data.get('family')
                genus = data.get('genus')
                
                if family and genus:
                    status = "OK"
                    
                    print(f"{species_name:<30} {family:<20} {genus:<20} {status}")
                    
                    species_to_genus[species_name] = genus
                    genus_to_family[genus] = family
                    
                    if family not in taxonomy[1]:
                        taxonomy[1].append(family)
                    
                    taxonomy[2][genus] = family
                    taxonomy[3][species_name] = genus
                else:
                    error_msg = f"Species '{species_name}' found in GBIF but family and genus not found, could be spelling error in species, check GBIF"
                    logger.error(error_msg)
                    print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                    print(f"Error: {error_msg}")
                    sys.exit(1)  # Stop the script
            else:
                error_msg = f"Species '{species_name}' not found in GBIF, could be spelling error, check GBIF"
                logger.error(error_msg)
                print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                print(f"Error: {error_msg}")
                sys.exit(1)  # Stop the script
                
        except Exception as e:
            error_msg = f"Error retrieving data for species '{species_name}': {str(e)}"
            logger.error(error_msg)
            print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
            print(f"Error: {error_msg}")
            sys.exit(1)  # Stop the script
    
    taxonomy[1] = sorted(list(set(taxonomy[1])))
    print("-" * 80)
    
    num_families = len(taxonomy[1])
    num_genera = len(taxonomy[2])
    num_species = len(taxonomy[3])
    
    print("\nFamily indices:")
    for i, family in enumerate(taxonomy[1]):
        print(f"  {i}: {family}")
    
    print("\nGenus indices:")
    for i, genus in enumerate(taxonomy[2].keys()):
        print(f"  {i}: {genus}")
    
    print("\nSpecies indices:")
    for i, species in enumerate(species_list):
        print(f"  {i}: {species}")
    
    logger.info(f"Taxonomy built: {num_families} families, {num_genera} genera, {num_species} species")
    return taxonomy, species_to_genus, genus_to_family

def create_mappings(taxonomy):
    """Create index mappings from taxonomy"""
    level_to_idx = {}
    idx_to_level = {}

    for level, labels in taxonomy.items():
        if isinstance(labels, list):
            level_to_idx[level] = {label: idx for idx, label in enumerate(labels)}
            idx_to_level[level] = {idx: label for idx, label in enumerate(labels)}
        else:  # Dictionary
            level_to_idx[level] = {label: idx for idx, label in enumerate(labels.keys())}
            idx_to_level[level] = {idx: label for idx, label in enumerate(labels.keys())}
    
    return level_to_idx, idx_to_level

class TwoStage:
    def __init__(self, yolo_model_path, hierarchical_model_path, species_names, output_dir="."):
        """
        Initialize the two-stage insect detection and classification model.
        
        Args:
            yolo_model_path (str): Path to the YOLO model weights for insect detection
            hierarchical_model_path (str): Path to the hierarchical classifier model weights
            species_names (list): List of species names to classify
            output_dir (str): Directory to save output files (default: current directory)
        """
        cuda_cleanup()
        
        self.device = setup_gpu()
        logger.info(f"Using device: {self.device}")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        logger.info(f"Results will be saved to: {self.output_dir}")
            
        print(f"Using device: {self.device}")

        self.yolo_model = YOLO(yolo_model_path)
        
        self.species_names = species_names
        
        logger.info(f"Loading model from {hierarchical_model_path}")
        try:
            checkpoint = torch.load(hierarchical_model_path, map_location='cpu')
            logger.info("Model loaded to CPU successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            
            if "taxonomy" in checkpoint:
                print("Using taxonomy from saved model")
                taxonomy = checkpoint["taxonomy"]
                if "species_list" in checkpoint:
                    saved_species = checkpoint["species_list"]
                    print(f"Saved model was trained on: {', '.join(saved_species)}")
    
                taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
            else:
                taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
        else:
            state_dict = checkpoint
            taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
        
        level_to_idx, idx_to_level = create_mappings(taxonomy)
        
        self.level_to_idx = level_to_idx
        self.idx_to_level = idx_to_level
        
        if hasattr(taxonomy, "items"):
            num_classes_per_level = [len(classes) if isinstance(classes, list) else len(classes.keys()) 
                                    for level, classes in taxonomy.items()]
        # else:
        #     num_classes_per_level = [4, 5, 9]  # Example values, adjust as needed
        
        print(f"Using model with class counts: {num_classes_per_level}")
        
        self.classification_model = HierarchicalInsectClassifier(num_classes_per_level)
        
        try:
            self.classification_model.load_state_dict(state_dict)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Attempting to load with strict=False...")
            self.classification_model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded with strict=False")
        
        try:
            self.classification_model.to(self.device)
            print(f"Model successfully transferred to {self.device}")
        except RuntimeError as e:
            logger.error(f"Error transferring model to {self.device}: {e}")
            print(f"Error transferring model to {self.device}, falling back to CPU")
            self.device = torch.device("cpu")
            # No need to move to CPU since it's already there
            
        self.classification_model.eval()

        self.classification_transform = transforms.Compose([
            transforms.Resize((768, 768)),  # Fixed size for all validation images
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model successfully loaded")
        print(f"Using species: {', '.join(species_names)}")
        
        self.species_to_genus = species_to_genus
        self.genus_to_family = genus_to_family

    def get_frames(self, test_dir):
        """
        Process images in the test directory to detect and classify insects.
        
        Args:
            test_dir (str): Path to directory containing test images
            
        Returns:
            tuple: A tuple containing:
                - results_dict (dict): Dictionary with detection and classification results
                - output_file (str): Path to the saved JSON results file
                
        Raises:
            FileNotFoundError: If the test directory doesn't exist
        """
        image_dir = test_dir
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Get all files in the directory
        all_files = os.listdir(image_dir)
        # Filter for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        image_names = [f for f in all_files if os.path.splitext(f.lower())[1] in image_extensions]
        
        if not image_names:
            logger.warning(f"No image files found in directory: {image_dir}")
            print(f"No image files found in directory: {image_dir}")
            # Return empty results if no images found
            # Use directory name for timestamp instead of generating a new one
            dir_name = os.path.basename(os.path.normpath(test_dir))
            output_file = os.path.join(self.output_dir, f"detection_results_{dir_name}.json")
            with open(output_file, 'w') as f:
                json.dump({}, f, indent=2)
            return {}, output_file
        
        results_dict = {}

        for image_name in tqdm(image_names, desc="Processing Images", unit="image"):
            image_path = os.path.join(image_dir, image_name)
            frame = cv2.imread(image_path)
            with torch.no_grad():
                results = self.yolo_model(frame, conf=0.55, verbose=False)

            detections = results[0].boxes
            frame_results = []

            if detections:
                for box in detections:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = xyxy[:4]
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    insect_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    insect_crop_rgb = cv2.cvtColor(insect_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(insect_crop_rgb)
                    input_tensor = self.classification_transform(pil_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.classification_model(input_tensor)
                    
                    family_output = outputs[0]    # First output is family (level 1)
                    genus_output = outputs[1]   # Second output is genus (level 2)
                    species_output = outputs[2]  # Third output is species (level 3)
                    
                    genus_idx = genus_output.argmax(dim=1).item()
                    family_idx = family_output.argmax(dim=1).item()
                    species_idx = species_output.argmax(dim=1).item()
                    
                    genus_confidences = torch.nn.functional.softmax(genus_output, dim=1)
                    family_confidences = torch.nn.functional.softmax(family_output, dim=1)
                    species_confidences = torch.nn.functional.softmax(species_output, dim=1)
                    
                    genus_confidence = float(genus_confidences[0][genus_idx].item())
                    family_confidence = float(family_confidences[0][family_idx].item())
                    species_confidence = float(species_confidences[0][species_idx].item())
                    
                    genus_name = self.idx_to_level[2][genus_idx] if genus_idx in self.idx_to_level[2] else f"unknown-genus-{genus_idx}"
                    family_name = self.idx_to_level[1][family_idx] if family_idx in self.idx_to_level[1] else f"unknown-family-{family_idx}"
                    species_name = self.species_names[species_idx] if species_idx < len(self.species_names) else f"unknown-species-{species_idx}"
                    
                    img_height, img_width, _ = frame.shape
                    x_center_norm = float(x_center / img_width)
                    y_center_norm = float(y_center / img_height)
                    width_norm = float(width / img_width)
                    height_norm = float(height / img_height)

                    # Use directory name for timestamp
                    dir_name = os.path.basename(os.path.normpath(test_dir))
                    
                    detection_result = {
                        "image_data": image_name,
                        "family": family_name,
                        "genus": genus_name,
                        "species": species_name,
                        "family_confidence": family_confidence,
                        "genus_confidence": genus_confidence,
                        "species_confidence": species_confidence,
                        "timestamp": dir_name,
                        "bbox": [x_center_norm, y_center_norm, width_norm, height_norm]
                    }
                    
                    frame_results.append(detection_result)
            
            results_dict[image_name] = frame_results

        dir_name = os.path.basename(os.path.normpath(test_dir))
        output_file = os.path.join(self.output_dir, f"detection_results_{dir_name}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        return results_dict, output_file

    def run(self, test_dir):
        """
        Run the complete detection and classification pipeline on a test directory.
        
        Args:
            test_dir (str): Path to directory containing test images
            
        Returns:
            tuple: A tuple containing:
                - results_dict (dict): Dictionary with detection and classification results
                - output_file (str): Path to the saved JSON results file
        """
        results_dict, output_file = self.get_frames(test_dir)
        print(f"\nProcessed {len(results_dict)} frames")
        print(f"Results saved to {output_file}")
        
        return results_dict, output_file