# GPX Running Analytics Dashboard

Live Demo: https://runthing-celerates.streamlit.app/

An end-to-end Python pipeline for parsing GPX files, engineering advanced run features, visualizing routes and pace profiles, training Machine Learning regression/classification models, and interacting with an AI Pace Coach powered by the Google GenAI SDK.

This dashboard is built with Streamlit, Plotly, and scikit-learn. It allows runners to analyze their physical activities, train predictive models on their pacing, classify run types, inspect fatigue patterns, and get personalized advice from an AI running coach.

## Features

- Interactive Route Mapping: Visualization of run paths overlaid on dark cartographic maps, with segment-by-segment pace gradients and start/finish indicators.
- Pace and Elevation Profiles: Analysis of speed, rolling pace, elevation change, and segment-by-segment pace variability.
- Fleet-Wide Distributions: Box, violin, histogram, and scatter plots showing distributions of duration, distance, pace, and cumulative training volume.
- Pace Prediction Pipeline: Regression models (Linear Regression, Ridge Regression, Random Forest, Gradient Boosting) to predict average pace based on distance, elevation, run type, and fatigue metrics.
- Run Type Classification Pipeline: Classifiers (Random Forest, Logistic Regression, Gradient Boosting, Support Vector Classifier, Decision Tree) to categorize runs as easy, tempo, interval, or long.
- Scenario Simulator: Sliders to configure hypothetical runs and predict their expected pace, run type classification probabilities, and simulated pace curves over distance.
- Fatigue Trend Analysis: Visual tracking of the fatigue index (pace delta between the second and first halves of a run) and rolling training loads.
- Interactive AI Running Coach: A chat assistant powered by Gemini 2.5 Flash, providing training recommendations and contextual analytics based on the active run history.

## Repository Structure

- app.py: Main Streamlit dashboard script that implements the multi-tab interface, user configurations, and visual plotting.
- data_loader.py: Module for loading local folders or file uploads, parsing GPX track points (latitude, longitude, elevation, timestamp) using gpxpy, and generating synthetic GPX runs.
- feature_engineering.py: Code that computes point-level metrics (segment distances, speed, elapsed time, rolling pace, pace variability) and run-level summaries (fatigue index, elevation gain per kilometer).
- models.py: Machine learning pipelines for training, evaluating, saving, and loading predictive and classification models.
- gpx_exporter.py: Utility to convert internal pandas DataFrames back into standards-compliant GPX 1.1 XML bytes, with support for zipping multiple files.
- data_generator.py: Standalone utility to generate synthetic runs in CSV format.
- requirements.txt: List of dependencies required to run the dashboard.
- .gitignore: Configuration for git to ignore temporary folders, saved model artifacts, and environment variables.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A Gemini API Key (optional, required only for the AI running coach tab)

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/iamgalenc/RunThing-Celerates.git
   cd RunThing-Celerates
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. Set up your Gemini API Key as an environment variable (optional):
   ```bash
   # On Windows (Command Prompt):
   set GEMINI_API_KEY=your_api_key_here
   # On Windows (PowerShell):
   $env:GEMINI_API_KEY="your_api_key_here"
   # On macOS/Linux:
   export GEMINI_API_KEY="your_api_key_here"
   ```
   Alternatively, you can enter the API key directly in the dashboard UI under the Coach tab, or define it in a local `.env` file in the project directory.

2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to the local URL (usually `http://localhost:8501`).

### Deployment to Streamlit Cloud

To deploy this application to Streamlit Community Cloud:
1. Push your code to a public GitHub repository.
2. Log into the Streamlit Share dashboard.
3. Click "New app", select your repository, branch, and set the main file path to `app.py`.
4. In the Advanced Settings, add your API key under the Secrets field:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```
5. Click "Deploy". The application will be live at `https://your-app-name.streamlit.app`.

## Data Loading Options

When you launch the app, you can choose from three data sources in the sidebar:
- Synthetic Demo Data: Instantly generates 50 synthetic running files covering easy, tempo, interval, and long runs.
- Upload GPX Files: Upload one or more of your own `.gpx` files from Garmin, Strava, Apple Watch, or other fitness devices.
- Local Folder: Specify the absolute path to a folder on your computer containing `.gpx` files.

## Machine Learning Details

The dashboard trains models dynamically based on the current dataset loaded into memory.

### Regression (Pace Prediction)
- Target: Average Pace (minutes per kilometer)
- Inputs: Distance (km), Elevation Gain (m), Elevation Gain per km, Run Type, Pace Variability, Fatigue Index.
- Available Models: Random Forest, Linear Regression, Ridge, Gradient Boosting.
- Saved Path: Saved as `saved_models/pace_regression_[model_name].joblib`.

### Classification (Run Type Classification)
- Target: Run Type (Easy, Tempo, Interval, Long)
- Inputs: Distance (km), Elevation Gain (m), Average Pace, Pace Standard Deviation, Pace Variability, Fatigue Index, Duration, Elevation Gain per km.
- Available Models: Random Forest, Logistic Regression, Gradient Boosting, Support Vector Machine (SVM), Decision Tree.
- Saved Path: Saved as `saved_models/run_classifier_[model_name].joblib`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
