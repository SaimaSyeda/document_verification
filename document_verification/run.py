# In run.py
from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)