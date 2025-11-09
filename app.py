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
    generate_synthetic_data,
    extract_life_events
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

@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the main HTML template and handles form submission."""
    if request.method == 'POST':
        try:
            # 1. Get user data from the form
            user_data = {
                'age': int(request.form['age']),
                'income': int(request.form['income']),
                'savings': int(request.form['savings']),
                'risk_score': float(request.form['risk_score']),
                'goals': request.form.get('goals', '') # Optional field
            }
            
            # --- START: NEW CODE TO EXTRACT LIFE EVENTS ---
            life_events = {} # Default to an empty dictionary
            if gemini_model and user_data['goals']: # Check if model exists and goals text was provided
                try:
                    life_events = extract_life_events(gemini_model, user_data['goals'])
                    app.logger.info(f"Extracted life events: {life_events}") # Good for debugging
                except Exception as e:
                    app.logger.error(f"Could not extract life events: {e}")
            # --- END: NEW CODE ---
            
            user_profile = pd.DataFrame({
                'age': [user_data['age']],
                'income': [user_data['income']],
                'savings': [user_data['savings']],
                'risk_score': [user_data['risk_score']]
            })
            
            # 2. Run the financial prediction logic
            prediction = portfolio_model.predict(user_profile)
            
            # 3. Get LLM Explanation
            feature_importances = [estimator.feature_importances_ for estimator in portfolio_model.estimators_]
            avg_importance = np.mean(feature_importances, axis=0)
            top_features_indices = np.argsort(avg_importance)[::-1][:2]
            top_features_names = [features[i] for i in top_features_indices]

            explanation = "LLM explanation unavailable."
            if gemini_model:
                explanation = get_llm_explanation(
                    gemini_model, 
                    user_profile, 
                    prediction, 
                    top_features_names,
                    life_events
                )
            # 4. Apply compliance rules
            adjusted_prediction, compliance_explanation = apply_compliance_rules(prediction[0], user_profile, compliance_rules)

            # 5. Get insurance recommendations
            insurance_recs = get_insurance_recommendation({
                'age': user_data['age'], 
                'income': user_data['income'], 
                'married_flag': 1, # Mocked
                'kids_flag': 1     # Mocked
            })

            # 6. Prepare data for the results template
            # NOTE: These are placeholder values. You should calculate these in finance.py
            projections_data = {
                'conservative_growth': [user_data['savings'] * (1.03**i) for i in range(31)],
                'expected_growth': [user_data['savings'] * (1.06**i) for i in range(31)],
                'optimistic_growth': [user_data['savings'] * (1.09**i) for i in range(31)],
                'ten_year': user_data['savings'] * (1.06**10),
                'twenty_year': user_data['savings'] * (1.06**20),
                'thirty_year': user_data['savings'] * (1.06**30),
            }
            
            # Per-asset expected returns (%) and risk (volatility %) - simple assumptions
            cash_return = 0.5
            cash_risk = 0.1
            bonds_return = 3.1
            bonds_risk = 4.2
            stocks_return = 9.2
            stocks_risk = 15.8

            # Adjusted allocation (percentages) from model
            alloc = adjusted_prediction
            stocks_pct = float(alloc[0])
            bonds_pct = float(alloc[1])
            cash_pct = float(alloc[2])

            # Compute portfolio expected return and portfolio risk as weighted averages
            # (This is a simplification — for production use covariance matrix or Monte Carlo)
            expected_return = (stocks_pct * stocks_return + bonds_pct * bonds_return + cash_pct * cash_return) / 100.0
            portfolio_risk = (stocks_pct * stocks_risk + bonds_pct * bonds_risk + cash_pct * cash_risk) / 100.0

            analysis_data = {
                'expected_return': round(expected_return, 2),
                'portfolio_risk': round(portfolio_risk, 2),
                'cash_risk': cash_risk, 'cash_return': cash_return,
                'bonds_risk': bonds_risk, 'bonds_return': bonds_return,
                'stocks_risk': stocks_risk, 'stocks_return': stocks_return,
            }

            results = {
                'allocation': {
                    'stocks': round(adjusted_prediction[0], 2),
                    'bonds': round(adjusted_prediction[1], 2),
                    'cash': round(adjusted_prediction[2], 2)
                },
                'analysis': analysis_data,
                'gemini_explanation': explanation,
                'projections': projections_data,
                'insurance_recommendations': insurance_recs,
                'compliance_explanation': compliance_explanation,
                'life_events': life_events
            }
            
            return render_template('results.html', **results)

        except Exception as e:
            app.logger.error(f"Prediction failed due to: {e}", exc_info=True)
            return render_template('index.html', error="An error occurred during analysis.")

    # For GET requests, just render the input form
    return render_template('index.html')

# --- API ROUTE: Handles predictions ---
# This can be removed or kept for other purposes if needed
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
