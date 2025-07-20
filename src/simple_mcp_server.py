import sys
import asyncio
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import uvicorn

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import settings
    print("✅ Configuración cargada desde config.settings")
except ImportError:
    # Fallback para deployment
    import os
    from dotenv import load_dotenv
    
    # Intentar cargar .env si existe
    load_dotenv()
    
    class Settings:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        MCP_HOST = "0.0.0.0"
        MCP_PORT = int(os.getenv("PORT", 8000))
    
    settings = Settings()
    print("✅ Configuración cargada desde variables de entorno")

# Modelos Pydantic para el MCP
class MCPRequest(BaseModel):
    method: str = Field(..., description="Método a ejecutar")
    params: Dict[str, Any] = Field(default={}, description="Parámetros del método")
    id: Optional[str] = Field(default=None, description="ID de la solicitud")

class MCPResponse(BaseModel):
    result: Any = Field(..., description="Resultado de la operación")
    error: Optional[str] = Field(default=None, description="Error si lo hay")
    id: Optional[str] = Field(default=None, description="ID de la solicitud")
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchRequest(BaseModel):
    query: str = Field(..., description="Consulta a realizar")
    max_results: int = Field(default=5, description="Número máximo de resultados")

class SimpleAgent:
    """Agente simplificado sin vector store pesado"""
    
    def __init__(self):
        print(f"🔍 Verificando OPENAI_API_KEY...")
        print(f"   - Variable de entorno: {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f"   - Settings: {bool(settings.OPENAI_API_KEY)}")
        
        if not settings.OPENAI_API_KEY:
            error_msg = """
❌ OPENAI_API_KEY no configurada
            
Soluciones:
1. En Railway > Variables, agrega: OPENAI_API_KEY=sk-proj-tu_key
2. Verifica que la variable esté guardada correctamente
3. Redeploy el proyecto
            """
            print(error_msg)
            raise ValueError("OPENAI_API_KEY no configurada")
        
        self.api_key = settings.OPENAI_API_KEY
        print(f"✅ API Key configurada correctamente (sk-...{self.api_key[-8:]})")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Datos de la ley (simplificado para demo)
        self.law_data = self.load_law_summary()
        print("✅ Datos de la ley cargados")
    
    def load_law_summary(self) -> Dict[str, str]:
        """Carga un resumen de la ley para demos"""
        return {
            "1": "OBJETO. El Sistema de Protección Social Integral para la Vejez, Invalidez y Muerte de origen común, tiene por objeto garantizar el amparo contra las contingencias derivadas de la vejez, la invalidez y la muerte de origen común.",
            "2": "DEFINICIONES. Para efectos de la presente ley se establecen las siguientes definiciones...",
            "15": "REQUISITOS PARA LA PENSIÓN DE VEJEZ. Para tener derecho a la pensión de vejez, el afiliado debe cumplir con los siguientes requisitos...",
            "25": "CÁLCULO DE LA PENSIÓN. El monto de la pensión se calculará con base en el promedio de los salarios de cotización..."
        }
    
    def call_openai_api(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Llama a la API de OpenAI"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Error de API: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Error OpenAI: {e}")
    
    def search_law(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Búsqueda simplificada en la ley"""
        results = []
        
        # Búsqueda simple por palabras clave
        query_lower = query.lower()
        
        for article_num, content in self.law_data.items():
            content_lower = content.lower()
            
            # Puntuación simple basada en coincidencias
            score = 0
            for word in query_lower.split():
                if word in content_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    "article_number": article_num,
                    "content": content,
                    "score": score,
                    "type": "article"
                })
        
        # Ordenar por puntuación y limitar resultados
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
    
    def get_article(self, article_number: str) -> Optional[Dict[str, Any]]:
        """Obtiene un artículo específico"""
        content = self.law_data.get(article_number)
        if content:
            return {
                "article_number": article_number,
                "content": content,
                "type": "article"
            }
        return None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Procesa una consulta completa"""
        start_time = datetime.now()
        
        try:
            # Buscar contenido relevante
            relevant_content = self.search_law(query, max_results=3)
            
            if not relevant_content:
                response = "No encontré información relevante sobre tu consulta en la Ley 2381 de 2024."
                sources = []
            else:
                # Generar respuesta con OpenAI
                context = "\n\n".join([
                    f"ARTÍCULO {item['article_number']}: {item['content']}"
                    for item in relevant_content
                ])
                
                messages = [
                    {
                        "role": "system",
                        "content": "Eres un asistente especializado en la Ley 2381 de 2024. Responde basándote únicamente en la información proporcionada y incluye las referencias de los artículos."
                    },
                    {
                        "role": "user",
                        "content": f"CONTEXTO DE LA LEY:\n{context}\n\nCONSULTA: {query}\n\nResponde con base en la información proporcionada."
                    }
                ]
                
                response = self.call_openai_api(messages, max_tokens=800)
                sources = [f"Artículo {item['article_number']}" for item in relevant_content]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "response": response,
                "sources": sources,
                "processing_time": processing_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            return {
                "query": query,
                "response": f"Error procesando consulta: {str(e)}",
                "sources": [],
                "processing_time": 0,
                "timestamp": start_time.isoformat()
            }

class SimpleMCPServer:
    def __init__(self):
        self.app = FastAPI(
            title="Ley 2381 MCP Server (Ligero)",
            description="Servidor MCP simplificado para la Ley 2381 de 2024",
            version="1.0.0"
        )
        
        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Inicializar agente simplificado
        try:
            self.agent = SimpleAgent()
            print("✅ Agente simplificado inicializado correctamente")
        except Exception as e:
            print(f"❌ Error inicializando agente: {e}")
            raise
        
        # Estadísticas del servidor
        self.stats = {
            "requests_count": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "start_time": datetime.now()
        }
        
        # Configurar rutas
        self.setup_routes()
    
    def setup_routes(self):
        """Configura las rutas del servidor MCP"""
        
        @self.app.get("/")
        async def root():
            return {
                "name": "Ley 2381 MCP Server (Ligero)",
                "version": "1.0.0",
                "description": "Servidor MCP simplificado para la Ley 2381 de 2024",
                "status": "online",
                "features": [
                    "Búsqueda básica en artículos",
                    "Consultas con IA",
                    "API REST",
                    "Protocolo MCP"
                ],
                "stats": self.stats
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "agent": "ready",
                "api_key_configured": bool(settings.OPENAI_API_KEY)
            }
        
        @self.app.get("/tools")
        async def list_tools():
            return [
                {
                    "name": "search_law",
                    "description": "Busca contenido en la Ley 2381 de 2024",
                    "parameters": {"query": "string", "max_results": "integer"}
                },
                {
                    "name": "get_article", 
                    "description": "Obtiene un artículo específico",
                    "parameters": {"article_number": "string"}
                },
                {
                    "name": "process_query",
                    "description": "Procesa consulta completa con IA",
                    "parameters": {"query": "string"}
                }
            ]
        
        @self.app.post("/tools/search")
        async def search_law(request: SearchRequest):
            try:
                self.stats["requests_count"] += 1
                results = self.agent.search_law(request.query, request.max_results)
                self.stats["successful_requests"] += 1
                
                return {
                    "query": request.query,
                    "results": results,
                    "count": len(results)
                }
            except Exception as e:
                self.stats["failed_requests"] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tools/article/{article_number}")
        async def get_article(article_number: str):
            try:
                self.stats["requests_count"] += 1
                article = self.agent.get_article(article_number)
                
                if article:
                    self.stats["successful_requests"] += 1
                    return article
                else:
                    self.stats["failed_requests"] += 1
                    raise HTTPException(status_code=404, detail=f"Artículo {article_number} no encontrado")
                    
            except HTTPException:
                raise
            except Exception as e:
                self.stats["failed_requests"] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tools/query")
        async def process_query(request: SearchRequest):
            try:
                self.stats["requests_count"] += 1
                result = self.agent.process_query(request.query)
                self.stats["successful_requests"] += 1
                return result
            except Exception as e:
                self.stats["failed_requests"] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_stats():
            uptime = datetime.now() - self.stats["start_time"]
            return {
                **self.stats,
                "uptime_seconds": uptime.total_seconds(),
                "success_rate": (
                    self.stats["successful_requests"] / max(self.stats["requests_count"], 1)
                ) * 100
            }
    
    def run(self, host: str = "0.0.0.0", port: int = None):
        """Ejecuta el servidor MCP"""
        if port is None:
            port = int(os.getenv("PORT", 8000))
        
        print(f"🚀 Iniciando servidor MCP ligero en http://{host}:{port}")
        print(f"📋 Documentación disponible en http://{host}:{port}/docs")
        print(f"🔑 OpenAI API Key configurada: {bool(settings.OPENAI_API_KEY)}")
        print(f"🌐 Health check: http://{host}:{port}/health")
        
        uvicorn.run(
            self.app, 
            host=host, 
            port=port, 
            log_level="info",
            access_log=True
        )

def main():
    """Función principal"""
    try:
        server = SimpleMCPServer()
        server.run()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
