import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from dataclasses import dataclass
import openai
from loguru import logger

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from src.vector_store import LawVectorStore

@dataclass
class QueryResult:
    """Resultado de una consulta al agente"""
    response: str
    sources: List[Dict[str, Any]]
    query: str
    timestamp: datetime
    processing_time: float

class OpenAIAgent:
    def __init__(self):
        # Verificar API key antes de crear el cliente
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY no configurada. Revisa tu archivo .env")
        
        try:
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("Cliente de OpenAI inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando cliente OpenAI: {e}")
            raise
            
        self.vector_store = LawVectorStore()
        self.conversation_cache = {}
        
        # Verificar que el vector store esté poblado
        if self.vector_store.collection.count() == 0:
            logger.warning("Vector store vacío. Ejecuta: python -m src.vector_store")
    
    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analiza la intención de la consulta del usuario"""
        intent_prompt = f"""
        Analiza la siguiente consulta sobre la Ley 2381 de 2024 (Sistema de Protección Social) y determina:

        1. TIPO DE CONSULTA:
        - "definition": Busca definiciones o conceptos
        - "procedure": Pregunta sobre procedimientos o trámites
        - "requirement": Busca requisitos o condiciones
        - "calculation": Involucra cálculos de pensiones, aportes, etc.
        - "general": Consulta general o exploratoria
        - "specific_article": Busca un artículo específico

        2. PALABRAS CLAVE: Identifica los términos más importantes

        3. ESPECIFICIDAD: 
        - "high": Pregunta muy específica
        - "medium": Pregunta moderadamente específica  
        - "low": Pregunta general o amplia

        Consulta: "{query}"

        Responde SOLO en formato JSON:
        {{
            "type": "tipo_de_consulta",
            "keywords": ["palabra1", "palabra2"],
            "specificity": "nivel",
            "suggested_search_terms": ["término1", "término2"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": intent_prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            # Extraer JSON de la respuesta
            response_text = response.choices[0].message.content.strip()
            
            # Buscar JSON en la respuesta
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                intent_analysis = json.loads(json_str)
            else:
                # Fallback si no se puede parsear
                intent_analysis = {
                    "type": "general",
                    "keywords": [query],
                    "specificity": "medium",
                    "suggested_search_terms": [query]
                }
                
            logger.info(f"Análisis de intención: {intent_analysis['type']} - {intent_analysis['specificity']}")
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Error en análisis de intención: {e}")
            # Fallback básico
            return {
                "type": "general",
                "keywords": [query],
                "specificity": "medium",
                "suggested_search_terms": [query]
            }
    
    def search_relevant_content(self, query: str, intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca contenido relevante basado en la consulta y análisis de intención"""
        
        # Determinar número de resultados según especificidad
        n_results = {
            "high": 3,
            "medium": 5,
            "low": 7
        }.get(intent_analysis["specificity"], 5)
        
        # Buscar con la consulta original
        results = self.vector_store.search(query, n_results=n_results)
        
        # Si hay términos de búsqueda sugeridos, hacer búsquedas adicionales
        if intent_analysis.get("suggested_search_terms"):
            for term in intent_analysis["suggested_search_terms"][:2]:  # Máximo 2 términos adicionales
                if term.lower() != query.lower():
                    additional_results = self.vector_store.search(term, n_results=3)
                    results.extend(additional_results)
        
        # Eliminar duplicados manteniendo los de mayor similitud
        seen_ids = {}
        unique_results = []
        
        for result in results:
            result_id = result['metadata'].get('article_number', result['metadata'].get('section_number', 'unknown'))
            
            if result_id not in seen_ids or result['similarity_score'] > seen_ids[result_id]['similarity_score']:
                seen_ids[result_id] = result
        
        unique_results = list(seen_ids.values())
        
        # Ordenar por relevancia
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Limitar resultados finales
        return unique_results[:n_results]
    
    def generate_response(self, query: str, relevant_content: List[Dict[str, Any]], intent_analysis: Dict[str, Any]) -> str:
        """Genera respuesta usando OpenAI GPT con el contenido relevante"""
        
        # Preparar contexto
        context_parts = []
        sources_info = []
        
        for i, content in enumerate(relevant_content):
            metadata = content['metadata']
            if metadata['type'] == 'article':
                source_ref = f"Artículo {metadata['article_number']}"
                context_parts.append(f"ARTÍCULO {metadata['article_number']}:\n{content['content']}")
            else:
                source_ref = f"{metadata['type'].title()} {metadata.get('section_number', 'N/A')}"
                context_parts.append(f"{source_ref.upper()}:\n{content['content']}")
            
            sources_info.append({
                "reference": source_ref,
                "similarity": content['similarity_score'],
                "type": metadata['type']
            })
        
        context = "\n\n".join(context_parts)
        
        # Crear prompt personalizado según el tipo de consulta
        system_prompt = f"""
        Eres un asistente especializado en la Ley 2381 de 2024 sobre el Sistema de Protección Social Integral para la Vejez, Invalidez y Muerte en Colombia.

        INSTRUCCIONES:
        1. Responde ÚNICAMENTE basándote en la información proporcionada de la ley
        2. Utiliza un lenguaje claro y accesible para cualquier persona
        3. Si la información no está en el contexto proporcionado, indícalo claramente
        4. Estructura tu respuesta de manera organizada
        5. Al final de tu respuesta, incluye las referencias exactas de los artículos utilizados

        TIPO DE CONSULTA DETECTADO: {intent_analysis['type']}
        ESPECIFICIDAD: {intent_analysis['specificity']}

        FORMATO DE RESPUESTA:
        [Respuesta en lenguaje natural y claro]

        **Referencias:**
        - [Lista de artículos citados]
        """
        
        user_prompt = f"""
        CONTEXTO DE LA LEY 2381 DE 2024:
        {context}

        CONSULTA DEL USUARIO:
        {query}

        Por favor, responde la consulta basándote únicamente en la información proporcionada.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"Respuesta generada exitosamente para consulta sobre: {intent_analysis['type']}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}"
    
    def generate_summary(self, content: List[Dict[str, Any]], topic: str) -> str:
        """Genera un resumen de múltiples artículos sobre un tema específico"""
        
        if not content:
            return "No se encontró información suficiente para generar un resumen."
        
        # Preparar contenido para resumen
        articles_text = []
        for item in content:
            metadata = item['metadata']
            if metadata['type'] == 'article':
                articles_text.append(f"Artículo {metadata['article_number']}: {item['content']}")
        
        if not articles_text:
            return "No se encontraron artículos relevantes para generar el resumen."
        
        combined_content = "\n\n".join(articles_text)
        
        summary_prompt = f"""
        Genera un resumen ejecutivo claro y organizado sobre "{topic}" basándote en los siguientes artículos de la Ley 2381 de 2024:

        {combined_content}

        El resumen debe:
        1. Ser conciso pero completo
        2. Usar lenguaje accesible
        3. Estar bien estructurado
        4. Incluir los puntos más importantes
        5. Mencionar los artículos de referencia al final

        FORMATO:
        ## Resumen: {topic}

        [Contenido del resumen organizado en párrafos]

        **Artículos consultados:** [Lista de artículos]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return f"Error al generar resumen: {str(e)}"
    
    def process_query(self, query: str, generate_summary_if_multiple: bool = True) -> QueryResult:
        """Procesa una consulta completa del usuario"""
        start_time = datetime.now()
        
        try:
            # 1. Analizar intención
            intent_analysis = self.analyze_intent(query)
            
            # 2. Buscar contenido relevante
            relevant_content = self.search_relevant_content(query, intent_analysis)
            
            if not relevant_content:
                response = "No encontré información relevante sobre tu consulta en la Ley 2381 de 2024. ¿Podrías reformular tu pregunta o ser más específico?"
                sources = []
            else:
                # 3. Generar respuesta
                if len(relevant_content) > 3 and generate_summary_if_multiple and intent_analysis['specificity'] == 'low':
                    # Para consultas generales con muchos resultados, generar resumen
                    response = self.generate_summary(relevant_content, query)
                else:
                    # Respuesta normal
                    response = self.generate_response(query, relevant_content, intent_analysis)
                
                # Preparar información de fuentes
                sources = []
                for content in relevant_content:
                    metadata = content['metadata']
                    if metadata['type'] == 'article':
                        sources.append({
                            "reference": f"Artículo {metadata['article_number']}",
                            "similarity_score": content['similarity_score'],
                            "type": "article"
                        })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                response=response,
                sources=sources,
                query=query,
                timestamp=start_time,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                response=f"Ocurrió un error al procesar tu consulta: {str(e)}",
                sources=[],
                query=query,
                timestamp=start_time,
                processing_time=processing_time
            )

def main():
    """Función principal para testing"""
    if not settings.OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY no configurada")
        print("   1. Obtén tu API key en: https://platform.openai.com/api-keys")
        print("   2. Agrégala al archivo .env como: OPENAI_API_KEY=tu_key_aqui")
        return
    
    agent = OpenAIAgent()
    
    # Consultas de prueba
    test_queries = [
        "¿Qué es el Sistema de Protección Social?",
        "¿Cuáles son los requisitos para la pensión de vejez?",
        "¿Cómo se calculan los aportes?",
        "artículo 15"
    ]
    
    print("🤖 Probando el agente de OpenAI...")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔍 Consulta: {query}")
        print("-" * 30)
        
        result = agent.process_query(query)
        
        print(f"⏱️  Tiempo de procesamiento: {result.processing_time:.2f}s")
        print(f"📄 Fuentes consultadas: {len(result.sources)}")
        print(f"\n💬 Respuesta:\n{result.response}")
        
        if result.sources:
            print(f"\n📚 Referencias:")
            for source in result.sources:
                print(f"   - {source['reference']} (relevancia: {source['similarity_score']:.3f})")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
