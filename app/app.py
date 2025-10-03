# app.py
from flask import Flask, request, jsonify
from core.deploy import ModelWrapper

app = Flask(__name__)
model = ModelWrapper()

@app.route("/predict", methods=["POST"])
def predict():
    bank_tables = request.get_json()
    try:
        prob = model.predict(bank_tables)
        return jsonify({"probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Now from command line run like
# """
# cd gnn_server
# export FLASK_APP=app.py
# flask run --host=0.0.0.0 --port=5000

# and in java
# import java.net.URI;
# import java.net.http.HttpClient;
# import java.net.http.HttpRequest;
# import java.net.http.HttpResponse;

# public class GNNClient {
#     public static void main(String[] args) throws Exception {
#         String json = """
#         {
#             "x": [
#                 [1,0,0,0,0,0,0],
#                 [0,1,0,0,0,0,0],
#                 [0,0,1,0,0,0,0]
#             ],
#             "edge_index": [
#                 [0, 1, 2],
#                 [1, 2, 0]
#             ]
#         }
#         """;

#         HttpClient client = HttpClient.newHttpClient();
#         HttpRequest request = HttpRequest.newBuilder()
#             .uri(new URI("http://localhost:5000/predict"))
#             .header("Content-Type", "application/json")
#             .POST(HttpRequest.BodyPublishers.ofString(json))
#             .build();

#         HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

#         System.out.println("Prediction: " + response.body());
#     }
# }
# """