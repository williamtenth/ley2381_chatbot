import sys
import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    filters, 
    ContextTypes
)
from telegram.constants import ChatAction, ParseMode
from loguru import logger

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from src.openai_agent import OpenAIAgent

class TelegramBot:
    def __init__(self):
        if not settings.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN no configurado. Revisa tu archivo .env")
            
        self.application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
        self.agent = OpenAIAgent()
        self.user_sessions = {}  # Cache de sesiones de usuario
        
        # Configurar handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Configura los manejadores de comandos y mensajes"""
        
        # Comandos
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("info", self.info_command))
        self.application.add_handler(CommandHandler("ejemplos", self.examples_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        
        # Botones inline
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        
        # Mensajes de texto (consultas)
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_query)
        )
        
        logger.info("Handlers configurados correctamente")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        user = update.effective_user
        
        # Crear teclado inline
        keyboard = [
            [
                InlineKeyboardButton("📖 Ver ejemplos", callback_data="examples"),
                InlineKeyboardButton("ℹ️ Información", callback_data="info")
            ],
            [
                InlineKeyboardButton("🔍 Buscar artículo", callback_data="search_article"),
                InlineKeyboardButton("💡 Ayuda", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_message = f"""
🤖 ¡Hola {user.first_name}! Soy tu asistente especializado en la **Ley 2381 de 2024** sobre el Sistema de Protección Social Integral.

💬 **¿Cómo puedo ayudarte?**
Puedes hacerme preguntas sobre:
• Requisitos para pensiones
• Definiciones de la ley
• Procedimientos específicos
• Artículos concretos

🎯 **Ejemplo:** 
_"¿Cuáles son los requisitos para la pensión de vejez?"_

👇 Usa los botones de abajo o simplemente escribe tu pregunta.
        """
        
        await update.message.reply_text(
            welcome_message, 
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Registrar usuario
        self.user_sessions[user.id] = {
            "first_interaction": datetime.now(),
            "query_count": 0,
            "username": user.username or user.first_name
        }
        
        logger.info(f"Usuario {user.id} ({user.first_name}) inició conversación")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        help_message = """
🆘 **AYUDA - Cómo usar el bot**

**Comandos disponibles:**
• `/start` - Iniciar o reiniciar el bot
• `/help` - Mostrar esta ayuda
• `/info` - Información sobre la Ley 2381
• `/ejemplos` - Ver ejemplos de consultas
• `/stats` - Estadísticas de uso

**Tipos de consultas que puedes hacer:**

🔍 **Búsquedas generales:**
_"¿Qué es el sistema de protección social?"_

📋 **Requisitos específicos:**
_"¿Cuáles son los requisitos para pensión de invalidez?"_

📄 **Artículos específicos:**
_"artículo 15"_ o _"muéstrame el artículo 23"_

🧮 **Cálculos y aportes:**
_"¿Cómo se calculan los aportes?"_

💡 **Tip:** Sé específico en tus preguntas para obtener mejores respuestas.
        """
        
        await update.message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN)
    
    async def info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /info"""
        info_message = """
📋 **LEY 2381 DE 2024**

**Sistema de Protección Social Integral para la Vejez, Invalidez y Muerte**

🎯 **Objetivo:** Garantizar el amparo contra las contingencias derivadas de la vejez, la invalidez y la muerte de origen común.

📊 **Datos de la ley:**
• Total de artículos: 132
• Extensión: 35 páginas
• Año de promulgación: 2024

🤖 **Sobre este bot:**
• Respuestas basadas en IA (OpenAI GPT)
• Búsqueda semántica inteligente
• Referencias exactas a artículos
• Análisis de intención de consultas

⚖️ **Aviso legal:** Este bot proporciona información general. Para asesoría legal específica, consulta con un profesional.
        """
        
        await update.message.reply_text(info_message, parse_mode=ParseMode.MARKDOWN)
    
    async def examples_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /ejemplos"""
        examples_message = """
💡 **EJEMPLOS DE CONSULTAS**

**📋 Consultas generales:**
• _"¿Qué es el sistema de protección social?"_
• _"¿Cuál es el objeto de la ley?"_
• _"¿Qué contingencias cubre la ley?"_

**👥 Requisitos y condiciones:**
• _"¿Cuáles son los requisitos para la pensión de vejez?"_
• _"¿Qué documentos necesito para pensión de invalidez?"_
• _"¿Cuándo puedo acceder a mi pensión?"_

**💰 Aportes y cálculos:**
• _"¿Cómo se calculan los aportes?"_
• _"¿Cuál es el monto mínimo de aportes?"_
• _"¿Cómo se determina el valor de la pensión?"_

**📄 Artículos específicos:**
• _"artículo 1"_
• _"muéstrame el artículo 25"_
• _"explícame el artículo 50"_

**🏢 Instituciones y entidades:**
• _"¿Qué entidades administran el sistema?"_
• _"¿Cuáles son las funciones de cada entidad?"_

¡Prueba con cualquiera de estos ejemplos!
        """
        
        await update.message.reply_text(examples_message, parse_mode=ParseMode.MARKDOWN)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /stats"""
        user_id = update.effective_user.id
        user_session = self.user_sessions.get(user_id, {})
        
        # Estadísticas del vector store
        vector_stats = self.agent.vector_store.get_statistics()
        
        stats_message = f"""
📊 **ESTADÍSTICAS**

**Tu uso:**
• Consultas realizadas: {user_session.get('query_count', 0)}
• Primera interacción: {user_session.get('first_interaction', 'N/A')}

**Base de conocimiento:**
• Total documentos indexados: {vector_stats.get('total_documents', 'N/A')}
• Artículos disponibles: {vector_stats.get('articles_count', 'N/A')}
• Secciones adicionales: {vector_stats.get('sections_count', 'N/A')}

**Sistema:**
• Motor de IA: OpenAI GPT-3.5
• Búsqueda: Vector semántico
• Última actualización: Ley 2381 de 2024
        """
        
        await update.message.reply_text(stats_message, parse_mode=ParseMode.MARKDOWN)
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja los botones inline"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "examples":
            await self.examples_command(update, context)
        elif data == "info":
            await self.info_command(update, context)
        elif data == "help":
            await self.help_command(update, context)
        elif data == "search_article":
            await query.edit_message_text(
                "🔍 **Búsqueda de artículos**\n\n"
                "Para buscar un artículo específico, simplemente escribe:\n"
                "• _\"artículo 15\"_\n"
                "• _\"art 23\"_\n"
                "• _\"muéstrame el artículo 7\"_\n\n"
                "O haz cualquier pregunta sobre la ley.",
                parse_mode=ParseMode.MARKDOWN
            )
    
    async def handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja las consultas de los usuarios"""
        user = update.effective_user
        query_text = update.message.text
        
        # Actualizar estadísticas de usuario
        if user.id in self.user_sessions:
            self.user_sessions[user.id]["query_count"] += 1
        
        logger.info(f"Consulta de {user.first_name} ({user.id}): {query_text}")
        
        # Mostrar que el bot está escribiendo
        await update.message.reply_chat_action(ChatAction.TYPING)
        
        try:
            # Procesar consulta con el agente
            result = self.agent.process_query(query_text)
            
            # Preparar respuesta
            response_text = result.response
            
            # Agregar información de fuentes si hay
            if result.sources:
                source_list = []
                for source in result.sources[:3]:  # Mostrar máximo 3 fuentes
                    source_list.append(f"• {source['reference']}")
                
                if source_list:
                    response_text += f"\n\n📚 **Referencias consultadas:**\n" + "\n".join(source_list)
            
            # Agregar tiempo de procesamiento
            response_text += f"\n\n⏱️ _Procesado en {result.processing_time:.1f}s_"
            
            # Verificar límite de caracteres de Telegram
            if len(response_text) > 4096:
                # Dividir mensaje largo
                parts = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                for part in parts:
                    await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)
            
            logger.info(f"Respuesta enviada a {user.first_name}. Fuentes: {len(result.sources)}")
            
        except Exception as e:
            error_message = f"""
❌ **Error procesando tu consulta**

Lo siento, ocurrió un error inesperado. Por favor:
1. Verifica que tu pregunta esté bien formulada
2. Intenta reformular tu consulta
3. Si el problema persiste, contacta al administrador

_Error: {str(e)}_
            """
            
            await update.message.reply_text(error_message, parse_mode=ParseMode.MARKDOWN)
            logger.error(f"Error procesando consulta de {user.first_name}: {e}")
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja errores globales"""
        logger.error(f"Error en bot: {context.error}")
        
        if update and update.message:
            await update.message.reply_text(
                "❌ Ocurrió un error inesperado. Por favor intenta de nuevo.",
                parse_mode=ParseMode.MARKDOWN
            )
    
    def run(self):
        """Ejecuta el bot"""
        try:
            # Configurar manejador de errores
            self.application.add_error_handler(self.error_handler)
            
            logger.info("🤖 Iniciando bot de Telegram...")
            logger.info(f"Bot configurado para Ley 2381 de 2024")
            
            # Ejecutar bot
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
        except Exception as e:
            logger.error(f"Error ejecutando bot: {e}")
            raise

def main():
    """Función principal"""
    try:
        # Verificar configuración
        if not settings.TELEGRAM_BOT_TOKEN:
            print("❌ Error: TELEGRAM_BOT_TOKEN no configurado")
            print("   1. Obtén tu token de @BotFather en Telegram")
            print("   2. Agrégalo al archivo .env como: TELEGRAM_BOT_TOKEN=tu_token_aqui")
            return
            
        if not settings.OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY no configurado")
            print("   1. Obtén tu API key en: https://platform.openai.com/api-keys")
            print("   2. Agrégala al archivo .env como: OPENAI_API_KEY=tu_key_aqui")
            return
        
        # Crear y ejecutar bot
        bot = TelegramBot()
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
