
## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file (if you don't have one yet) with the following content:
    ```txt
    pandas
    numpy
    scikit-learn
    joblib
    gradio
    matplotlib
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Instructions

Run the scripts in the following order from your terminal within the project directory:

1.  **Generate Data:**
    ```bash
    python 007_DataGenerator.py
    ```
    This will create `k12_tutoring_dataset.csv`.

2.  **Train Score Prediction Model:**
    ```bash
    python 008_ScorePredictor.py
    ```
    This will create `score_predictor_model.pkl` and optionally a feature importance plot.

3.  **Train Promotion Prediction Model:**
    ```bash
    python 011_TrainPromotionModel.py
    ```
    This will create `promotion_predictor_model.pkl` and optionally a feature importance plot.

4.  **Launch the Gradio Web App:**
    ```bash
    python 010_GradioRecommenderApp_with_Promotion.py
    ```
    This will start the Gradio interface and provide a local URL (e.g., `http://127.0.0.1:7860`).

5.  **Interact:** Open the provided URL in your web browser. Enter the student details in the interface and click "Submit" to see the AI predictions and recommendations.

## Models Used

*   **Promotion Prediction Model (`promotion_predictor_model.pkl`):**
    *   **Type:** `RandomForestClassifier` from Scikit-learn.
    *   **Purpose:** Predicts a binary outcome (Yes/No) for student promotion.
    *   **Key Features Used (Example):** `Age`, `IQ`, `Time_Per_Day`, `Level_Student`, `Earning_Class`. (Refer to `011_TrainPromotionModel.py` for exact features).

*   **Score Prediction Model (`score_predictor_model.pkl`):**
    *   **Type:** `RandomForestRegressor` from Scikit-learn.
    *   **Purpose:** Predicts the numerical assessment score for the student's *next* module.
    *   **Key Features Used (Example):** `Age`, `IQ`, `Time_Per_Day`, `Level_Student`, `Course_Name`, `Material_Level` (of the *next* module). (Refer to `008_ScorePredictor.py` for exact features).

## Future Enhancements

*   Integrate with real student data (requires careful handling of privacy and ethics).
*   Implement more sophisticated models (e.g., Gradient Boosting, Neural Networks, Deep Knowledge Tracing).
*   Develop a more detailed and dynamic curriculum map.
*   Add user accounts and track student progress over time.
*   Connect recommendations to actual learning resources.
*   Implement original Goal 4: Dynamic adaptation/generation of learning material based on student needs.
*   Perform hyperparameter tuning for better model performance.
*   Expand Exploratory Data Analysis (EDA).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. ( **Note:** You should add a file named `LICENSE` containing the text of the MIT license, or choose another appropriate license).

---

*This README provides a guide to understanding, setting up, and running the AI-Powered K-12 Course Recommender & Promotion Predictor project.*
