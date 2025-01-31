from flask import Flask

# Create Flask app
app = Flask(__name__)

# Import views after app creation to avoid circular imports
from app import views
