import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from loguru import logger
import uuid

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

class LawVectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = None
        self.collection = None
        self.setup_chroma()
        
    def setup_chroma(self):
        """Configura ChromaDB"""
        try:
            # Configurar ChromaDB para persistencia
            self.client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Crear o obtener colección
            self.collection = self.client.get_or_create_collection(
                name="ley_2381_2024",
                metadata={"description": "Ley 2381 de 2024 - Sistema de Protección Social"}
            )
            
            logger.info("ChromaDB configurado exitosamente")
            
        except Exception as e:
            logger.error(f"Error configurando ChromaDB: {e}")
            raise
    
    def load_processed_data(self) -> Dict[str, Any]:
        """Carga los datos procesados del PDF"""
        processed_file = settings.PROCESSED_DATA_DIR / "processed_law.json"
        
        if not processed_file.exists():
            raise FileNotFoundError(
                f"Archivo procesado no encontrado: {processed_file}. "
                "Ejecuta primero: python -m src.pdf_processor"
            )
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Datos cargados: {data['metadata']['total_articles']} artículos")
        return data
    
    def prepare_documents(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepara los documentos para indexación"""
        documents = []
        
        # Procesar artículos
        for article in data['articles']:
            doc = {
                'id': f"article_{article['article_number']}",
                'content': article['content'],
                'type': 'article',
                'article_number': article['article_number'],
                'metadata': {
                    'article_number': article['article_number'],
                    'type': 'article',
                    'length': len(article['content'])
                }
            }
            documents.append(doc)
        
        # Procesar secciones (capítulos, títulos)
        for i, section in enumerate(data['sections']):
            doc = {
                'id': f"section_{section['type']}_{section['section_number']}",
                'content': section['content'],
                'type': section['type'],
                'section_number': section['section_number'],
                'metadata': {
                    'section_number': section['section_number'],
                    'type': section['type'],
                    'length': len(section['content'])
                }
            }
            documents.append(doc)
        
        logger.info(f"Preparados {len(documents)} documentos para indexación")
        return documents
    
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]], List[Dict]]:
        """Crea embeddings para los documentos"""
        texts = [doc['content'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        logger.info("Generando embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Convertir a lista de listas (requerido por ChromaDB)
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        return ids, embeddings_list, metadatas, texts
    
    def reset_vector_store(self):
        """Elimina completamente el vector store para empezar de cero"""
        try:
            logger.info("Eliminando vector store existente...")
            self.client.delete_collection("ley_2381_2024")
            self.collection = self.client.create_collection(
                name="ley_2381_2024",
                metadata={"description": "Ley 2381 de 2024 - Sistema de Protección Social"}
            )
            logger.info("Vector store reiniciado exitosamente")
        except Exception as e:
            logger.warning(f"No se pudo eliminar la colección existente: {e}")
            # Crear nueva colección
            self.collection = self.client.get_or_create_collection(
                name="ley_2381_2024",
                metadata={"description": "Ley 2381 de 2024 - Sistema de Protección Social"}
            )
    def index_documents(self, force_reindex: bool = False):
        """Indexa todos los documentos en el vector store"""
        try:
            # Verificar si ya existe contenido
            count = self.collection.count()
            if count > 0 and not force_reindex:
                logger.info(f"Vector store ya contiene {count} documentos. Use force_reindex=True para reindexar.")
                return
            
            if force_reindex and count > 0:
                self.reset_vector_store()
            
            # Cargar y preparar datos
            data = self.load_processed_data()
            documents = self.prepare_documents(data)
            
            # Verificar IDs únicos antes de indexar
            ids_set = set()
            unique_documents = []
            
            for doc in documents:
                if doc['id'] not in ids_set:
                    ids_set.add(doc['id'])
                    unique_documents.append(doc)
                else:
                    logger.warning(f"ID duplicado encontrado y omitido: {doc['id']}")
            
            logger.info(f"Documentos únicos a indexar: {len(unique_documents)}")
            
            # Crear embeddings
            ids, embeddings, metadatas, texts = self.create_embeddings(unique_documents)
            
            # Indexar en ChromaDB
            logger.info("Indexando documentos en ChromaDB...")
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            final_count = self.collection.count()
            logger.info(f"✅ Indexación completada. Total de documentos: {final_count}")
            
        except Exception as e:
            logger.error(f"Error en indexación: {e}")
            raise
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Busca documentos relevantes usando similitud semántica"""
        try:
            if self.collection.count() == 0:
                raise ValueError("Vector store vacío. Ejecuta index_documents() primero.")
            
            # Buscar documentos similares
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Formatear resultados
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convertir distancia a similitud
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
            
            logger.info(f"Búsqueda completada. Encontrados {len(formatted_results)} resultados para: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            raise
    
    def get_article_by_number(self, article_number: str) -> Dict[str, Any]:
        """Obtiene un artículo específico por su número"""
        try:
            results = self.collection.get(
                ids=[f"article_{article_number}"],
                include=['documents', 'metadatas']
            )
            
            if not results['documents']:
                return None
            
            return {
                'content': results['documents'][0],
                'metadata': results['metadatas'][0]
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo artículo {article_number}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del vector store"""
        try:
            total_docs = self.collection.count()
            
            # Obtener muestra para análisis
            sample = self.collection.get(limit=total_docs, include=['metadatas'])
            
            articles_count = sum(1 for meta in sample['metadatas'] if meta['type'] == 'article')
            sections_count = total_docs - articles_count
            
            return {
                'total_documents': total_docs,
                'articles_count': articles_count,
                'sections_count': sections_count,
                'collection_name': self.collection.name
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}

def main():
    """Función principal para testing"""
    vector_store = LawVectorStore()
    
    try:
        print("🔄 Iniciando indexación...")
        vector_store.index_documents()
        
        print("\n📊 Estadísticas del vector store:")
        stats = vector_store.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n🔍 Probando búsqueda...")
        query = "pensión de vejez"
        results = vector_store.search(query, n_results=3)
        
        print(f"\nResultados para '{query}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
            print(f"   Tipo: {result['metadata']['type']}")
            if result['metadata']['type'] == 'article':
                print(f"   Artículo: {result['metadata']['article_number']}")
            print(f"   Contenido: {result['content'][:200]}...")
        
        print("\n✅ Vector store funcionando correctamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
