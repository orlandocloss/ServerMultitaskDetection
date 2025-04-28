# Sensing Garden Video Processing

This repository contains code to automatically download and process videos from the Sensing Garden API, extract frames, run object detection and species classification, and upload results back to the Sensing Garden database.

## Setup Instructions
1. Setup server
2. Clone the Repository
```
git clone https://github.com/orlandocloss/sensing-garden-processing.git
cd sensing-garden-processing
```

3. Create Python Environment
Create and activate a virtual environment, then install the required dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
4. Configure Environment Variables
Create a .env file in the root directory with the following content:
```
# API Key for authentication (required)
SENSING_GARDEN_API_KEY=your_api_key_here

# Base URL for the API
API_BASE_URL=https://api.sensinggarden.com/v1/

# AWS Credentials for S3 video access
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
```

6. Download Model Weights
Download the required model weights files:

*YOLO weights for general object detection*
*Hierarchical classifier weights for species identification*

Place them in the project directory or specify their paths when running the script.

## Running the Processing Script

The main script (client_processing.py) can be run with various command-line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| --output-path | ./output | Directory where processed data, extracted frames, and results will be stored |
| --yolo-weights | ./small-generic.pt | Path to the YOLO weights file for initial insect detection |
| --hierarchical-weights | ./best_bjerge.pt | Path to the hierarchical classifier weights file for species identification |
| --device-id | b8f2ed92a70e5df3 | Device ID for the Sensing Garden API to fetch videos from and upload results to |
| --model-id | best_bjerge.pt | Model ID that will be associated with the classifications in the Sensing Garden database |
| --time-interval | None | Time interval in seconds between extracted frames (None extracts all frames) |
| --species-file | None | Path to a text file containing species names (one per line) to detect. If not provided, a default list is used |
| --interval-hours | 1.0 | Run interval in hours - how frequently the system checks for new videos |

Example Usage:
```
python client_processing.py --output-path ./my_results --yolo-weights ./models/insect_detector.pt --hierarchical-weights ./models/species_classifier.pt --time-interval 1.0 --species-file ./species_list.txt --interval-hours 3
```
