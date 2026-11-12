
import streamlit as st

st.set_page_config(
    page_title="FEAST | Food Estimation and Analysis using Smart Technology",
    page_icon="🍲",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }

    /* All text */
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp li,
    .stApp strong {
        color: #FFFFFF !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
    }

    /* Header */
    [data-testid="stHeader"] {
        background-color: #000000;
    }

    /* Input fields */
    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="select"] {
        color: #FFFFFF !important;
        background-color: #222222 !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    .stButton > button,
    .stButton > button p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍲 FEAST AI")
st.subheader("Food Estimation and Analysis using Smart Technology")

st.write(
    """
    An ANN-Based Food Demand Forecasting,
    Classification and Surplus Estimation System.
    """
)

st.info(
    "FEAST uses Artificial Neural Networks (ANN) "
    "to predict food demand, estimate potential surplus, "
    "and support food waste reduction."
)

st.markdown("---")

st.markdown(
    """
    ### Project Features

    - ANN Regression – Predict expected food orders.
    - ANN Classification – Classify demand as Low, Medium, or High.
    - Surplus Estimation – Estimate potential surplus using prepared quantity.
    - Food Rescue Recommendations – Suggest ways to manage potential surplus.
    """
)

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Local Project Paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(PROJECT_PATH)
MODEL_PATH = os.path.join(PROJECT_PATH, "models")
DATASET_PATH = os.path.join(ROOT_PATH, "food_dataset")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ANN Regression Architecture
class ANNRegressor(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


# ANN Classification Architecture
class ANNClassifier(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.model(x)

# Load Saved Artifacts
@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        os.path.join(MODEL_PATH, "preprocessor.pkl")
    )

    thresholds = joblib.load(
        os.path.join(MODEL_PATH, "demand_thresholds.pkl")
    )

    input_features = len(preprocessor.get_feature_names_out())

    if input_features != 217:
        raise ValueError(
        f"Expected 217 input features, got {input_features}"
    )

    regressor = ANNRegressor(input_features).to(device)
    classifier = ANNClassifier(input_features).to(device)

    regressor.load_state_dict(
        torch.load(
            os.path.join(MODEL_PATH, "ann_regressor.pth"),
            map_location=device,
            weights_only=True
        )
    )

    classifier.load_state_dict(
        torch.load(
            os.path.join(MODEL_PATH, "ann_classifier.pth"),
            map_location=device,
            weights_only=True
        )
    )

    regressor.eval()
    classifier.eval()

    return preprocessor, thresholds, regressor, classifier


try:
    preprocessor, thresholds, regressor, classifier = load_models()

    st.sidebar.success("All models loaded successfully!")
    st.sidebar.write("Device:", device)
    st.sidebar.write("Input Features:", len(preprocessor.get_feature_names_out()))

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Load Dataset for Input Options
@st.cache_data
def load_dataset():
    return pd.read_csv(
        os.path.join(DATASET_PATH, "train_prepared.csv")
    )

train_data = load_dataset()

# Application Input Section
st.markdown("---")
st.header("📋 Food Demand Prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Food Center Details")

    week = st.slider(
    "Week",
    min_value=int(train_data["week"].min()),
    max_value=int(train_data["week"].max()),
    value=int(train_data["week"].min()),
    help="The week number for which food demand is being predicted."
)

    center_id = st.selectbox(
        "Fulfilment Center",
        sorted(train_data["center_id"].unique().tolist()),
        help="Unique identifier of the fulfilment center responsible for food distribution."
    )

    meal_id = st.selectbox(
        "Meal ID",
        sorted(train_data["meal_id"].unique().tolist()),
        help="Unique identifier assigned to the food item or meal."
    )

    category = st.selectbox(
        "Food Category",
        sorted(train_data["category"].unique().tolist()),
        help="Category to which the selected meal belongs, such as Beverages, Rice or Sandwich."
    )

    cuisine = st.selectbox(
        "Cuisine",
        sorted(train_data["cuisine"].unique().tolist()),
        help="Cuisine type associated with the selected meal, such as Continental or Indian."
    )

with col2:
    st.subheader("Pricing and Promotion")

    checkout_price = st.number_input(
        "Checkout Price",
        min_value=0.01,
        value=100.0,
        help="Selling price of the meal offered to customers after applying discounts."
    )

    base_price = st.number_input(
        "Base Price",
        min_value=0.01,
        value=120.0,
        help="Original or standard price of the meal before applying discounts."
    )

    emailer_for_promotion = st.selectbox(
        "Email Promotion",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Indicates whether the meal is promoted through email marketing campaigns."
    )

    homepage_featured = st.selectbox(
        "Homepage Featured",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Indicates whether the meal is featured on the food ordering platform's homepage."
    )

st.subheader("Prepared Food Quantity")

prepared_quantity = st.number_input(
    "Enter Prepared Quantity (Orders)",
    min_value=0,
    value=100,
    step=1,
    help="Number of food portions planned for preparation. Used to estimate potential surplus by comparing it with predicted demand."
)

st.caption(
    "Enter the quantity of food you plan to prepare. "
    "This will be used later to estimate potential surplus."
)

st.markdown("---")
st.subheader("🏢 Additional Center Information")

col3, col4 = st.columns(2)

with col3:
    city_code = st.selectbox(
        "City Code",
        sorted(train_data["city_code"].unique().tolist()),
        help="Unique code identifying the city in which the fulfilment center operates."
    )

    region_code = st.selectbox(
        "Region Code",
        sorted(train_data["region_code"].unique().tolist()),
        help="Code identifying the geographical region associated with the fulfilment center."
    )

with col4:
    center_type = st.selectbox(
        "Center Type",
        sorted(train_data["center_type"].unique().tolist()),
        help="Classification of the fulfilment center based on its operational type."
    )

    op_area = st.number_input(
        "Operating Area",
        min_value=0.01,
        value=float(train_data["op_area"].median()),
        help="Operational area measure associated with the fulfilment center."
    )

st.success("All required input fields are available!")

st.markdown("---")

# Prediction Button
if st.button("🔍 Predict Food Demand", use_container_width=True):

    # Feature Engineering
    discount_amount = base_price - checkout_price

    discount_percentage = (
        discount_amount / base_price
    ) * 100

    promotion_active = int(
        emailer_for_promotion == 1 or homepage_featured == 1
    )

    # Create Input DataFrame
    input_data = pd.DataFrame([{
        "week": week,
        "center_id": center_id,
        "meal_id": meal_id,
        "checkout_price": checkout_price,
        "base_price": base_price,
        "emailer_for_promotion": emailer_for_promotion,
        "homepage_featured": homepage_featured,
        "category": category,
        "cuisine": cuisine,
        "city_code": city_code,
        "region_code": region_code,
        "center_type": center_type,
        "op_area": op_area,
        "discount_amount": discount_amount,
        "discount_percentage": discount_percentage,
        "promotion_active": promotion_active
    }])

    # Maintain Original Feature Order
    feature_columns = [
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
        "category",
        "cuisine",
        "city_code",
        "region_code",
        "center_type",
        "op_area",
        "discount_amount",
        "discount_percentage",
        "promotion_active"
    ]

    input_data = input_data[feature_columns]
    
    st.write("Input Data Prepared Successfully!")
    # Input Field Descriptions
    field_descriptions = {
        "week": "Week number for which food demand is being predicted.",
        "center_id": "Unique identifier of the fulfilment center.",
        "meal_id": "Unique identifier assigned to the food item or meal.",
        "checkout_price": "Selling price of the meal offered to customers.",
        "base_price": "Original or standard price of the meal.",
        "emailer_for_promotion": "Indicates whether the meal is promoted through email campaigns.",
        "homepage_featured": "Indicates whether the meal is featured on the platform homepage.",
        "category": "Food category to which the selected meal belongs.",
        "cuisine": "Cuisine type associated with the selected meal.",
        "city_code": "Identifier of the city where the fulfilment center operates.",
        "region_code": "Geographical region identifier of the fulfilment center.",
        "center_type": "Classification of the fulfilment center.",
        "op_area": "Operational area measure associated with the fulfilment center.",
        "discount_amount": "Difference between the base price and checkout price.",
        "discount_percentage": "Discount amount expressed as a percentage of the base price.",
        "promotion_active": "Indicates whether any promotional activity is active."
    }
    # Display Input Details with Readable Names
    field_names = {
        "week": "Week",
        "center_id": "Fulfilment Center ID",
        "meal_id": "Meal ID",
        "checkout_price": "Checkout Price",
        "base_price": "Base Price",
        "emailer_for_promotion": "Email Promotion",
        "homepage_featured": "Homepage Featured",
        "category": "Food Category",
        "cuisine": "Cuisine",
        "city_code": "City Code",
        "region_code": "Region Code",
        "center_type": "Center Type",
        "op_area": "Operating Area",
        "discount_amount": "Discount Amount",
        "discount_percentage": "Discount Percentage",
        "promotion_active": "Promotion Active"
    }

    vertical_data = input_data.T.reset_index()
    vertical_data.columns = ["Feature", "Selected Value"]

    vertical_data["Feature"] = vertical_data["Feature"].map(field_names)

    vertical_data["Description"] = vertical_data["Feature"].map(
        {
            field_names[key]: value
            for key, value in field_descriptions.items()
        }
    )
    # Display Readable Input Table
    table_html = vertical_data.to_html(
        index=False,
        classes="feast-table",
        escape=True
    )

    st.markdown(
        """
        <style>
        .feast-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #111111 !important;
        }

        .feast-table th {
            background-color: #333333 !important;
            color: #FFFFFF !important;
            font-weight: bold !important;
            padding: 12px !important;
            border: 1px solid #777777 !important;
            text-align: left !important;
        }

        .feast-table td {
            background-color: #111111 !important;
            color: #FFFFFF !important;
            font-weight: 500 !important;
            padding: 12px !important;
            border: 1px solid #777777 !important;
            text-align: left !important;
        }

        .feast-table tr:nth-child(even) td {
            background-color: #222222 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(table_html, unsafe_allow_html=True)


    # Transform Input Using Saved Preprocessor

    processed_input = preprocessor.transform(input_data)

    if hasattr(processed_input, "toarray"):
        processed_input = processed_input.toarray()

    processed_input = np.asarray(
        processed_input,
        dtype=np.float32
    )

    input_tensor = torch.tensor(
        processed_input,
        dtype=torch.float32,
        device=device
    )

    # Model Predictions
    with torch.no_grad():

        # Regression Prediction
        regression_output = regressor(input_tensor)

        predicted_orders = float(
            np.expm1(regression_output.cpu().numpy()[0][0])
        )

        predicted_orders = max(0, predicted_orders)

        # Classification Prediction
        classification_output = classifier(input_tensor)

        predicted_class = int(
            torch.argmax(classification_output, dim=1).item()
        )

        predicted_demand = thresholds["class_mapping"][
            predicted_class
        ]

    # Display Results
    st.markdown("---")
    st.header("📊 Food Demand Prediction Results")

    st.metric(
    "Predicted Food Orders",
    f"{predicted_orders:.0f}"
)

    st.metric(
    "Predicted Demand Level",
    predicted_demand
)

    st.success("Both ANN models generated predictions successfully!")
    

    # Surplus Estimation
    estimated_surplus = max(
        0,
        prepared_quantity - predicted_orders
    )

    st.markdown("---")
    st.header("♻️ Food Surplus Estimation")

    surplus_col1, surplus_col2 = st.columns(2)

    with surplus_col1:
        st.metric(
            "Prepared Quantity",
            f"{prepared_quantity}"
        )

    with surplus_col2:
        st.metric(
            "Estimated Surplus",
            f"{estimated_surplus:.0f}"
        )

    if estimated_surplus > 0:
        st.warning(
            f"Approximately {estimated_surplus:.0f} orders "
            "may remain based on the demand forecast."
        )
    else:
        st.success(
            "No potential surplus is estimated based on the forecast."
        )

    # Food Rescue Recommendation Engine
    st.markdown("---")
    st.header("🌱 Smart Food Rescue Recommendations")

    if estimated_surplus <= 0:

        st.success(
            "Recommended Action: Normal Food Preparation"
        )

        st.write(
            "Your planned preparation quantity is within "
            "the predicted demand. No potential surplus "
            "has been estimated."
        )

    elif estimated_surplus <= prepared_quantity * 0.20:

        st.info(
            "Recommended Action: Redistribute Potential Surplus"
        )

        st.write(
            "Consider redistributing the estimated surplus "
            "to nearby food outlets or community organizations."
        )

    else:

        st.warning(
            "Recommended Action: Prioritize Food Rescue"
        )

        st.write(
            "A relatively large surplus is estimated. "
            "Consider arranging timely donation or redistribution "
            "through eligible food rescue organizations."
        )

    st.caption(
        "Food safety reminder: Only donate or redistribute food "
        "that has been safely prepared, stored, transported, "
        "and is suitable for consumption. Follow applicable "
        "food safety requirements."
    )
