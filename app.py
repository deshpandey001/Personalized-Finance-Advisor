from flask import Flask, request, jsonify, render_template # ADDED render_template
import pandas as pd
import os
import numpy as np # Ensure numpy is imported for calculations

# Import your core logic from finance.py
from finance import (
    train_portfolio_model,
    initialize_gemini,
    get_llm_explanation,
    apply_compliance_rules,
    get_insurance_recommendation,
    generate_synthetic_data
)

app = Flask(__name__,template_folder="templates")

# --- Load and Train Models on App Startup ---
# This ensures the model is ready to go as soon as the server starts.
print("Initializing models and LLM client...")
df = generate_synthetic_data()
portfolio_model, features = train_portfolio_model(df)
gemini_model = initialize_gemini()
print("Initialization complete.")

# Hardcoded rules for compliance
compliance_rules = {
    "equity_cap": {
        "condition_value": 5, # Risk score below this value triggers the cap
        "max_stocks": 70      # Maximum stock allocation allowed
    }
}

# --- ROOT ROUTE: Serves the HTML file (Fixes the 404/Not Found issue) ---
@app.route('/')
def index():
    """Renders the main HTML template when the user visits the root URL."""
    return render_template('index.html')

# --- API ROUTE: Handles predictions ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get user data from the frontend
        user_data = request.json
        
        user_profile = pd.DataFrame({
            'age': [user_data['age']],
            'income': [user_data['income']],
            'savings': [user_data['savings']],
            'risk_score': [user_data['risk_score']]
        })
        
        # 2. Run the financial prediction logic
        prediction = portfolio_model.predict(user_profile)
        predicted_allocation = {
            'stocks': round(prediction[0][0], 2),
            'bonds': round(prediction[0][1], 2),
            'cash': round(prediction[0][2], 2)
        }

        # 3. Get LLM Explanation (with feature importances)
        # Using list comprehension to safely access feature importances
        feature_importances = [estimator.feature_importances_ for estimator in portfolio_model.estimators_]
        
        if feature_importances:
            avg_importance = np.mean(feature_importances, axis=0)
            top_features_indices = np.argsort(avg_importance)[::-1][:2]
            top_features_names = [features[i] for i in top_features_indices]
        else:
            top_features_names = ["Model Data Unavailable"]

        explanation = "LLM explanation unavailable (API key or model issue)."
        if gemini_model:
            explanation = get_llm_explanation(
                gemini_model, 
                user_profile, 
                prediction, 
                top_features_names
            )

        # 4. Apply compliance rules
        adjusted_prediction, compliance_explanation = apply_compliance_rules(prediction[0], user_profile, compliance_rules)
        adjusted_allocation = {
            'stocks': round(adjusted_prediction[0], 2),
            'bonds': round(adjusted_prediction[1], 2),
            'cash': round(adjusted_prediction[2], 2)
        }

        # 5. Get insurance recommendations (using actual user data)
        insurance_recs = get_insurance_recommendation({
            'age': user_data['age'], 
            'income': user_data['income'], 
            'married_flag': 1, # Mocked flag for simplicity
            'kids_flag': 1     # Mocked flag for simplicity
        })
        
        # 6. Return all results in a single JSON response
        response = {
            'predicted_allocation': predicted_allocation,
            'llm_explanation': explanation,
            'adjusted_allocation': adjusted_allocation,
            'compliance_explanation': compliance_explanation,
            'insurance_recommendations': insurance_recs
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Log the error for debugging and return a 500 status to the client
        app.logger.error(f"Prediction failed due to: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during prediction."}), 500

if __name__ == '__main__':
    app.run(debug=True)
