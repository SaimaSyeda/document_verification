import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-development'
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload