from flask import Flask, send_from_directory  
from database import init_db
from chatbot_route import chatbot_bp
from fitness_route import fitness_bp

# Initialize Flask app
app = Flask(__name__)

# Initialize database
init_db()

# Register blueprints
app.register_blueprint(chatbot_bp, url_prefix="/chatbot")
app.register_blueprint(fitness_bp, url_prefix="/fitness")

@app.route("/")
def home():
    return {"message": "FitPal API is running!"}

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
