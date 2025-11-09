import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
import os
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Ensure you have set your Gemini API key as an environment variable
# If not, uncomment and set your key here:
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# --- 1. Data Generation and Model Training ---

def generate_synthetic_data(num_samples=1000):
    """Generates a synthetic dataset for investment portfolio prediction."""
    np.random.seed(42)
    age = np.random.randint(25, 65, num_samples)
    income = np.random.randint(50000, 200000, num_samples)
    savings = np.random.randint(10000, 500000, num_samples)
    risk_score = np.random.uniform(1, 10, num_samples)

    # Rule-based allocation logic
    stocks = (risk_score * 5 + (65 - age) / 2)
    bonds = (100 - stocks) * (age / 65)
    cash = 100 - stocks - bonds

    allocations = np.vstack([stocks, bonds, cash]).T
    allocations[allocations < 0] = 0
    allocations = (allocations / allocations.sum(axis=1)[:, np.newaxis]) * 100
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'savings': savings,
        'risk_score': risk_score,
        'stocks_pct': allocations[:, 0],
        'bonds_pct': allocations[:, 1],
        'cash_pct': allocations[:, 2]
    })
    
    return df

def train_portfolio_model(df):
    """Trains a RandomForestRegressor model for portfolio allocation."""
    features = ['age', 'income', 'savings', 'risk_score']
    targets = ['stocks_pct', 'bonds_pct', 'cash_pct']
    X = df[features]
    y = df[targets]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    multi_target_rf = MultiOutputRegressor(rf)
    multi_target_rf.fit(X_train, y_train)

    y_pred = multi_target_rf.predict(X_test)
    print(f"Model trained. Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    
    return multi_target_rf, features

# --- 2. LLM Integration for Explainability and Feature Extraction ---

def initialize_gemini():
    """Initializes and returns the Gemini Pro model."""
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return None

def get_llm_explanation(model, user_profile, prediction, top_features, life_events=None):
    """Generates a human-like explanation for a portfolio recommendation, including life event advice."""

    # Start building the prompt with the core financial data
    prompt_parts = [
        "You are 'NiveshMitra', a friendly and professional AI financial advisor. Your goal is to provide clear, simple, and actionable advice.",
        "An investor with the following profile:",
        f"- Age: {user_profile['age'].iloc[0]}",
        f"- Annual Income: ${user_profile['income'].iloc[0]:,}",
        f"- Total Savings: ${user_profile['savings'].iloc[0]:,}",
        f"- Risk Score (1-10): {user_profile['risk_score'].iloc[0]}",
        "",
        "Based on our analysis, we recommend the following portfolio allocation:",
        f"- Stocks: {prediction[0][0]:.1f}%",
        f"- Bonds: {prediction[0][1]:.1f}%",
        f"- Cash: {prediction[0][2]:.1f}%",
        "",
        f"The primary factors from your profile that influenced this allocation were your: {', '.join(top_features)}.",
    ]

    # Dynamically add a section for life events if they exist
    if life_events:
        event_details = []
        for event, years in life_events.items():
            if years is not None:
                event_name = event.replace('_years', '').replace('_', ' ').title()
                event_details.append(f"- {event_name} (in approx. {years} years)")
        
        if event_details:
            prompt_parts.append("\nWe've also noted your upcoming life goals:")
            prompt_parts.extend(event_details)

    # Add the final instructions for the AI
    prompt_parts.extend([
        "\n--- YOUR TASK ---",
        "1.  **Explain the Portfolio:** In 1-2 sentences, briefly explain WHY this portfolio allocation is suitable for the user's profile (e.g., 'Given your age and risk score, this mix balances growth with stability.').",
        "2.  **Provide Life Event Advice:** Based on the user's upcoming life goals, provide 1-2 actionable, bullet-pointed financial tips for each event. For example, for 'Buying a House', suggest creating a dedicated high-yield savings account for a down payment. For 'Retirement', suggest increasing contributions.",
        "3.  **Tone:** Keep the language encouraging, simple, and easy to understand. Avoid complex jargon.",
        "Begin your response with a friendly greeting like 'Hello! Here is your personalized financial plan:'"
    ])
    
    final_prompt = "\n".join(prompt_parts)

    try:
        response = model.generate_content(
            final_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return response.text
    except Exception as e:
        return f"Error generating explanation: {e}"

def extract_life_events(model, text_input):
    """Uses Gemini to extract life events from unstructured text."""
    extraction_prompt = f"""
    From the following text, extract any life events and their timelines into a JSON object.
    Use the keys: 'marriage_years', 'retirement_years', 'house_years', 'kids_years'.
    If an event is not mentioned, its value should be null.

    Example:
    Input: 'I'm saving up for a house in 5 years.'
    Output: {{ "marriage_years": null, "retirement_years": null, "house_years": 5, "kids_years": null }}

    Input text: "{text_input}"
    Output:
    """
    try:
        response = model.generate_content(
            extraction_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error extracting life events: {e}")
        return {}

# --- 3. Compliance and Joint Prediction ---

def apply_compliance_rules(allocation, profile, rules):
    """Adjusts portfolio allocation based on hardcoded compliance rules."""
    adjusted_allocation = np.copy(allocation)
    explanation = ""

    # Rule 1: Equity cap for low-risk investors
    if profile['risk_score'][0] < rules['equity_cap']['condition_value']:
        if adjusted_allocation[0] > rules['equity_cap']['max_stocks']:
            excess = adjusted_allocation[0] - rules['equity_cap']['max_stocks']
            adjusted_allocation[0] = rules['equity_cap']['max_stocks']
            adjusted_allocation[2] += excess  # Shift excess to cash
            explanation = f"Your stock allocation was reduced from {allocation[0]:.2f}% to {adjusted_allocation[0]:.2f}% because of regulatory rules for moderate risk investors. The excess was moved to cash."
    
    return adjusted_allocation, explanation

# --- In your finance.py file, locate the get_insurance_recommendation function ---

def get_insurance_recommendation(user_data): 
    """Provides a rule-based insurance recommendation, expecting direct int values (fixed access)."""
    recommendations = []
    
    # Rule 1: Life insurance for married individuals with kids (mocked flags)
    if user_data.get('married_flag', 0) == 1 and user_data.get('kids_flag', 0) == 1:
        recommendations.append("Life insurance (e.g., coverage of 10x income)")
    
    # Rule 2: Health/critical illness cover for aging, high-income individuals
    # CRITICAL CHANGE: Removed [0] indexing
    if user_data['age'] > 50 and user_data['income'] > 150000: 
        recommendations.append("Health/Critical Illness Cover")
    elif user_data['age'] < 30 and user_data['income'] > 80000:
        recommendations.append("Consider reviewing disability income insurance as you grow your career.")
    
    return recommendations

# --- Main Execution Flow ---

if __name__ == "__main__":
    # --- Part 1: Profile-based Investment Prediction ---
    print("--- 1. Generating Data and Training Model ---")
    df = generate_synthetic_data()
    portfolio_model, features = train_portfolio_model(df)
    
    # Get a sample user profile
    sample_user_profile = pd.DataFrame({
        'age': [35],
        'income': [120000],
        'savings': [150000],
        'risk_score': [7.5]
    })

    # Predict portfolio allocation
    prediction = portfolio_model.predict(sample_user_profile)
    
    # Get feature importances for explanation
    importance_stocks = portfolio_model.estimators_[0].feature_importances_
    avg_importance = (importance_stocks + portfolio_model.estimators_[1].feature_importances_ + portfolio_model.estimators_[2].feature_importances_) / 3
    top_features_indices = np.argsort(avg_importance)[::-1][:2]
    top_features_names = [features[i] for i in top_features_indices]

    print("\n--- Model Prediction and Explainability ---")
    print(f"Sample User Profile:\n{sample_user_profile.to_string(index=False)}")
    print(f"Predicted Allocation: Stocks={prediction[0][0]:.2f}%, Bonds={prediction[0][1]:.2f}%, Cash={prediction[0][2]:.2f}%")
    print(f"Top influencing factors: {', '.join(top_features_names)}")

    # Initialize Gemini
    gemini_model = initialize_gemini()
    if gemini_model:
        explanation = get_llm_explanation(gemini_model, sample_user_profile, prediction, top_features_names)
        print("\n--- LLM-Generated Explanation ---")
        print(explanation)
    
    print("\n" + "="*50)

    # --- Part 2: Life-Event-Aware Portfolio Prediction ---
    print("\n--- 2. Life-Event-Aware Portfolio Prediction ---")
    user_text = "I'm a young professional getting married next year and I'm also planning for my retirement in 20 years."
    print(f"User input text: '{user_text}'")

    if gemini_model:
        events = extract_life_events(gemini_model, user_text)
        print("LLM Extracted Events:", events)
        
        # Add a flag to the user profile based on the extracted event
        if events.get('marriage_years'):
            sample_user_profile['marriage_flag'] = [1] # Keep list for DataFrame consistency
        if events.get('retirement_years'):
            sample_user_profile['retirement_flag'] = [1] # Keep list for DataFrame consistency
        
        # A full implementation would involve retraining the model with these features
        # For this example, we just show the flags being added
        print("Updated User Profile with Life Event Flags:", sample_user_profile.to_dict())

    print("\n" + "="*50)

    # --- Part 3 & 4: Insurance Prediction and Compliance ---
    print("\n--- 3. Joint Insurance and Portfolio Prediction with Compliance ---")
    
    # Get insurance recommendations
    # FIXED: Pass simple integers for the function expecting direct access
    insurance_recs = get_insurance_recommendation({
        'age': 55, 
        'income': 200000, 
        'married_flag': 1, 
        'kids_flag': 1
    })
    print("Insurance Recommendations:", insurance_recs)

    # Demonstrate compliance check
    compliance_rules = {
        "equity_cap": {
            "condition_value": 5,
            "max_stocks": 70
        }
    }

    # Simulate a low-risk user to trigger the rule
    low_risk_user = pd.DataFrame({
        'age': [40],
        'income': [80000],
        'savings': [100000],
        'risk_score': [4.0]
    })
    
    original_prediction = portfolio_model.predict(low_risk_user)
    adjusted_prediction, compliance_explanation = apply_compliance_rules(original_prediction[0], low_risk_user, compliance_rules)

    print("\n--- Real-time Market Policies Compliance ---")
    print(f"Original Prediction (Low Risk User): {np.round(original_prediction[0], 2)}")
    print(f"Adjusted Prediction (after compliance): {np.round(adjusted_prediction, 2)}")
    
    if compliance_explanation and gemini_model:
        llm_prompt_compliance = f"Original allocation: {np.round(original_prediction[0], 2)}. Adjusted allocation: {np.round(adjusted_prediction, 2)}. The reason is: '{compliance_explanation}'. Please rephrase this as friendly, clear financial advice."
        print("\n--- LLM-Generated Compliance Explanation ---")
        response = gemini_model.generate_content(llm_prompt_compliance, safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        })
        print(response.text)