from flask import Blueprint, request, jsonify
from chatbot_utils import classify_text, generate_response
from database import session, User, Interaction

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/user/<int:user_id>", methods=["GET"])
def get_user_profile(user_id):
    """
    Retrieve user profile details.
    """
    try:
        print(f"📢 Fetching user ID: {user_id}")  # ✅ Debugging log
        user = session.query(User).filter_by(id=user_id).first()

        if not user:
            print("❌ User not found")  # ✅ Debugging log
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "user_id": user.id,
            "name": user.name,
            "email": user.email,
            "height": user.height,
            "weight": user.weight,
            "preferences": user.preferences
        })

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")  # ✅ Debugging log
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("/users", methods=["GET"])
def get_all_users():
    """
    Retrieve all registered users from the database.
    """
    try:
        users = session.query(User).all()
        
        # Convert user objects to a list of dictionaries
        users_list = [
            {
                "user_id": user.id,
                "name": user.name,
                "email": user.email,
                "height": user.height,
                "weight": user.weight,
                "preferences": user.preferences
            }
            for user in users
        ]

        return jsonify(users_list), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("/", methods=["POST"])
def chatbot():
    """
    Handle user input, classify text, and return chatbot response.
    """
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        user_id = data.get("user_id")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # Classify text
        category = classify_text(user_input)

        # Generate response
        response = generate_response(user_input, category)

        # Store conversation history
        if user_id:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                new_interaction = Interaction(
                    user_id=user_id, input_text=user_input, response_text=response
                )
                session.add(new_interaction)
                session.commit()

        return jsonify({"message": response, "category": category})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("/user/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    """
    Update user details in the database.
    """
    try:
        data = request.get_json()
        user = session.query(User).filter_by(id=user_id).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        # Update user fields if provided
        user.name = data.get("name", user.name)
        user.email = data.get("email", user.email)
        user.height = data.get("height", user.height)
        user.weight = data.get("weight", user.weight)
        user.preferences = data.get("preferences", user.preferences)

        session.commit()

        return jsonify({"message": "User updated successfully!"}), 200

    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    
@chatbot_bp.route("/history/<int:user_id>", methods=["GET"])
def get_chat_history(user_id):
    """
    Retrieve chat history for a specific user.
    """
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        interactions = session.query(Interaction).filter_by(user_id=user_id).order_by(Interaction.created_at).all()

        chat_history = [
            {
                "text": interaction.input_text,
                "sender": "user",
                "created_at": interaction.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for interaction in interactions
        ] + [
            {
                "text": interaction.response_text,
                "sender": "bot",
                "created_at": interaction.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for interaction in interactions
        ]

        return jsonify(chat_history)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("/register", methods=["POST"])
def register_user():
    try:
        data = request.get_json()
        print(f"📥 Received Data: {data}")  # ✅ Debugging input data

        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        height = data.get("height")
        weight = data.get("weight")
        preferences = data.get("preferences", "No preferences")

        if not name or not email:
            print("❌ Missing name or email!")  # ✅ Debug missing data
            return jsonify({"error": "Name and email are required!"}), 400

        if not isinstance(height, int) or not isinstance(weight, int):
            print("❌ Invalid height or weight!")  # ✅ Debug invalid numbers
            return jsonify({"error": "Height and weight must be numbers!"}), 400

        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            print("❌ User already exists!")  # ✅ Debug existing user
            return jsonify({"error": "User with this email already exists!"}), 400

        new_user = User(name=name, email=email, height=height, weight=weight, preferences=preferences)
        session.add(new_user)
        session.commit()

        return jsonify({
            "message": "User registered successfully!",
            "user_id": new_user.id,
            "name": new_user.name,
            "email": new_user.email,
            "height": new_user.height,
            "weight": new_user.weight,
            "preferences": new_user.preferences
        }), 201

    except Exception as e:
        print(f"❌ Exception: {e}")  # ✅ Debug unexpected errors
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("user/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    try:
        # Fetch user from the database
        user = session.query(User).filter_by(id=user_id).first()
        
        if not user:
            return jsonify({"error": "User not found"}), 404

        # ✅ Delete all interactions linked to this user
        session.query(Interaction).filter_by(user_id=user_id).delete()

        # ✅ Now delete the user
        session.delete(user)
        session.commit()

        return jsonify({"message": "User deleted successfully!"}), 200

    except Exception as e:
        print(f"❌ Error deleting user: {str(e)}")
        session.rollback()  # Rollback in case of failure
        return jsonify({"error": str(e)}), 500

