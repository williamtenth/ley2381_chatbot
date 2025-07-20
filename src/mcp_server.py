import sys
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import uvicorn

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from src.openai_agent import OpenAIAgent

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

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str = Field(..., description="Consulta a realizar")
    max_results: int = Field(default=5, description="Número máximo de resultados")

class ArticleRequest(BaseModel):
    article_number: str = Field(..., description="Número del artículo a buscar")

class MCPServer:
    def __init__(self):
        self.app = FastAPI(
            title="Ley 2381 MCP Server",
            description="Model Context Protocol Server para la Ley 2381 de 2024",
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
        
        # Inicializar agente
        try:
            self.agent = OpenAIAgent()
            logger.info("Agente OpenAI inicializado en servidor MCP")
        except Exception as e:
            logger.error(f"Error inicializando agente: {e}")
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
                "name": "Ley 2381 MCP Server",
                "version": "1.0.0",
                "description": "Model Context Protocol Server para la Ley 2381 de 2024",
                "tools": await self.list_tools(),
                "stats": self.stats
            }
        
        @self.app.post("/mcp", response_model=MCPResponse)
        async def mcp_endpoint(request: MCPRequest, background_tasks: BackgroundTasks):
            """Endpoint principal del protocolo MCP"""
            self.stats["requests_count"] += 1
            
            try:
                result = await self.handle_mcp_request(request)
                self.stats["successful_requests"] += 1
                
                # Log en background
                background_tasks.add_task(
                    self.log_request, 
                    request.method, 
                    request.params, 
                    success=True
                )
                
                return MCPResponse(
                    result=result,
                    id=request.id
                )
                
            except Exception as e:
                self.stats["failed_requests"] += 1
                error_msg = str(e)
                
                # Log en background
                background_tasks.add_task(
                    self.log_request, 
                    request.method, 
                    request.params, 
                    success=False, 
                    error=error_msg
                )
                
                return MCPResponse(
                    result=None,
                    error=error_msg,
                    id=request.id
                )
        
        @self.app.get("/tools")
        async def list_tools():
            """Lista las herramientas disponibles"""
            return await self.list_tools()
        
        @self.app.post("/tools/search")
        async def search_law(request: SearchRequest):
            """Herramienta de búsqueda semántica"""
            try:
                results = self.agent.vector_store.search(
                    request.query, 
                    n_results=request.max_results
                )
                return {
                    "query": request.query,
                    "results": results,
                    "count": len(results)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tools/article")
        async def get_article(request: ArticleRequest):
            """Herramienta para obtener un artículo específico"""
            try:
                article = self.agent.vector_store.get_article_by_number(
                    request.article_number
                )
                if article:
                    return {
                        "article_number": request.article_number,
                        "content": article
                    }
                else:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Artículo {request.article_number} no encontrado"
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tools/query")
        async def process_query(query_request: SearchRequest):
            """Herramienta para procesar consultas completas"""
            try:
                result = self.agent.process_query(query_request.query)
                return {
                    "query": result.query,
                    "response": result.response,
                    "sources": result.sources,
                    "processing_time": result.processing_time,
                    "timestamp": result.timestamp.isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_stats():
            """Estadísticas del servidor"""
            uptime = datetime.now() - self.stats["start_time"]
            return {
                **self.stats,
                "uptime_seconds": uptime.total_seconds(),
                "success_rate": (
                    self.stats["successful_requests"] / max(self.stats["requests_count"], 1)
                ) * 100
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check del servidor"""
            try:
                # Verificar que el vector store esté disponible
                vector_stats = self.agent.vector_store.get_statistics()
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "vector_store": {
                        "documents": vector_stats.get("total_documents", 0),
                        "available": vector_stats.get("total_documents", 0) > 0
                    },
                    "agent": "ready"
                }
            except Exception as e:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Service unhealthy: {str(e)}"
                )
    
    async def handle_mcp_request(self, request: MCPRequest) -> Any:
        """Maneja las solicitudes MCP según el método"""
        method = request.method
        params = request.params
        
        if method == "tools/list":
            return await self.list_tools()
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_arguments = params.get("arguments", {})
            
            if tool_name == "search_law":
                query = tool_arguments.get("query", "")
                max_results = tool_arguments.get("max_results", 5)
                results = self.agent.vector_store.search(query, n_results=max_results)
                return {
                    "tool": "search_law",
                    "query": query,
                    "results": results
                }
            
            elif tool_name == "get_article":
                article_number = tool_arguments.get("article_number")
                article = self.agent.vector_store.get_article_by_number(article_number)
                return {
                    "tool": "get_article",
                    "article_number": article_number,
                    "content": article
                }
            
            elif tool_name == "process_query":
                query = tool_arguments.get("query", "")
                result = self.agent.process_query(query)
                return {
                    "tool": "process_query",
                    "query": result.query,
                    "response": result.response,
                    "sources": result.sources,
                    "processing_time": result.processing_time
                }
            
            else:
                raise ValueError(f"Herramienta desconocida: {tool_name}")
        
        elif method == "resources/list":
            return await self.list_resources()
        
        elif method == "resources/read":
            resource_uri = params.get("uri")
            return await self.read_resource(resource_uri)
        
        else:
            raise ValueError(f"Método MCP desconocido: {method}")
    
    async def list_tools(self) -> List[ToolDefinition]:
        """Lista las herramientas disponibles según el protocolo MCP"""
        return [
            ToolDefinition(
                name="search_law",
                description="Busca contenido en la Ley 2381 de 2024 usando búsqueda semántica",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Consulta a buscar en la ley"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Número máximo de resultados",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            ToolDefinition(
                name="get_article",
                description="Obtiene un artículo específico de la Ley 2381 de 2024",
                parameters={
                    "type": "object",
                    "properties": {
                        "article_number": {
                            "type": "string",
                            "description": "Número del artículo a obtener"
                        }
                    },
                    "required": ["article_number"]
                }
            ),
            ToolDefinition(
                name="process_query",
                description="Procesa una consulta completa usando IA y retorna respuesta estructurada",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Consulta sobre la ley a procesar"
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """Lista los recursos disponibles"""
        vector_stats = self.agent.vector_store.get_statistics()
        
        return [
            {
                "uri": "law://2381/2024/full",
                "name": "Ley 2381 de 2024 Completa",
                "description": "Texto completo de la Ley 2381 de 2024",
                "mimeType": "application/json"
            },
            {
                "uri": "law://2381/2024/stats",
                "name": "Estadísticas de la Base de Conocimiento",
                "description": f"Estadísticas: {vector_stats.get('total_documents', 0)} documentos",
                "mimeType": "application/json"
            }
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Lee un recurso específico"""
        if uri == "law://2381/2024/stats":
            return self.agent.vector_store.get_statistics()
        elif uri == "law://2381/2024/full":
            return {
                "description": "Ley 2381 de 2024 - Sistema de Protección Social",
                "total_articles": self.agent.vector_store.get_statistics().get("articles_count", 0),
                "access_methods": ["search", "article_number", "semantic_query"]
            }
        else:
            raise ValueError(f"Recurso no encontrado: {uri}")
    
    async def log_request(self, method: str, params: Dict, success: bool, error: str = None):
        """Log de solicitudes en background"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "params": params,
            "success": success,
            "error": error
        }
        
        if success:
            logger.info(f"MCP Request: {method} - Success")
        else:
            logger.error(f"MCP Request: {method} - Error: {error}")
    
    def run(self, host: str = "0.0.0.0", port: int = None):
        """Ejecuta el servidor MCP"""
        # Usar puerto de la variable de entorno para deployment
        if port is None:
            port = int(os.getenv("PORT", settings.MCP_PORT))
        
        logger.info(f"🚀 Iniciando servidor MCP en http://{host}:{port}")
        logger.info(f"📋 Documentación disponible en http://{host}:{port}/docs")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

def main():
    """Función principal"""
    try:
        # Verificar configuración
        if not settings.OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY no configurado")
            return
        
        # Crear y ejecutar servidor
        server = MCPServer()
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Servidor MCP detenido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal en servidor MCP: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
