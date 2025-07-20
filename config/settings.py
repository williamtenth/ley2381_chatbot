import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Settings:
    # Rutas del proyecto
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Configuración de Claude
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
    MAX_TOKENS = 4000
    
    # Configuración del vector store
    VECTOR_DB_PATH = PROCESSED_DATA_DIR / "vector_db"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Configuración del bot
    MAX_MESSAGE_LENGTH = 4096  # Límite de Telegram
    CACHE_SIZE = 100
    
    # Configuración MCP Server
    MCP_HOST = "localhost"
    MCP_PORT = 8000
    
    # Archivo de la ley
    LAW_PDF_PATH = DATA_DIR / "ley_2381_2024.pdf"
    
    def __init__(self):
        # Crear directorios si no existen
        self.DATA_DIR.mkdir(exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        
        # Validar API keys
        if not self.ANTHROPIC_API_KEY:
            print("⚠️  ANTHROPIC_API_KEY no encontrada en variables de entorno")
        if not self.TELEGRAM_BOT_TOKEN:
            print("⚠️  TELEGRAM_BOT_TOKEN no encontrado en variables de entorno")

# Instancia global
settings = Settings()
