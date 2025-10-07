# pylint: disable=no-member
from flask import Flask, request, jsonify
import os

# Local imports
from lambdaml.deploy import ModelWrapper
from lambdaml.log import setup_logger


logger = setup_logger(__name__)

def create_app():
    # Initialialize flask app and model
    app = Flask(__name__)
    model = ModelWrapper(
        trial_dir=os.environ['APP_CONFIG_PATH'],
    )
    
    # Define the app
    @app.route("/predict", methods=["POST"])
    def predict():
        bank_tables = request.get_json()
        try:
            prob = model.predict(bank_tables)
            return jsonify({"probability": prob})
        except TypeError as e:
            return jsonify({"error": str(e)}), 500

    return app
