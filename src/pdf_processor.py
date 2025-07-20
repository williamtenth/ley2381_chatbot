import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pdfplumber
from loguru import logger

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

class LawPDFProcessor:
    def __init__(self):
        self.pdf_path = settings.LAW_PDF_PATH
        self.output_dir = settings.PROCESSED_DATA_DIR
        
    def extract_text_from_pdf(self) -> str:
        """Extrae todo el texto del PDF"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                        
            logger.info(f"Texto extraído exitosamente. Longitud: {len(full_text)} caracteres")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extrayendo texto del PDF: {e}")
            raise
    
    def segment_by_articles(self, text: str) -> List[Dict[str, str]]:
        """Segmenta el texto por artículos"""
        articles = []
        seen_articles = set()  # Para evitar duplicados
        
        # Patrones para detectar artículos
        article_pattern = r'ARTÍCULO\s+(\d+)\.?\s*(.*?)(?=ARTÍCULO\s+\d+|$)'
        
        matches = re.finditer(article_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            article_number = match.group(1)
            article_content = match.group(2).strip()
            
            # Verificar si ya procesamos este artículo
            if article_number in seen_articles:
                logger.warning(f"Artículo {article_number} duplicado - omitiendo")
                continue
                
            # Limpiar el contenido
            article_content = re.sub(r'\n+', ' ', article_content)
            article_content = re.sub(r'\s+', ' ', article_content)
            
            if len(article_content) > 50:  # Filtrar artículos muy cortos
                articles.append({
                    "article_number": article_number,
                    "content": article_content,
                    "type": "article"
                })
                seen_articles.add(article_number)
                
        logger.info(f"Se encontraron {len(articles)} artículos únicos")
        return articles
    
    def extract_chapters_and_titles(self, text: str) -> List[Dict[str, str]]:
        """Extrae capítulos y títulos para contexto adicional"""
        sections = []
        seen_sections = set()  # Para evitar duplicados
        
        # Patrones para capítulos y títulos
        patterns = [
            (r'CAPÍTULO\s+([IVX\d]+)\.?\s*(.*?)(?=CAPÍTULO|ARTÍCULO|$)', "chapter"),
            (r'TÍTULO\s+([IVX\d]+)\.?\s*(.*?)(?=TÍTULO|CAPÍTULO|ARTÍCULO|$)', "title"),
        ]
        
        for pattern, section_type in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                number = match.group(1)
                content = match.group(2).strip()
                
                # Crear ID único
                section_id = f"{section_type}_{number}"
                
                # Verificar duplicados
                if section_id in seen_sections:
                    logger.warning(f"Sección {section_id} duplicada - omitiendo")
                    continue
                
                # Limpiar contenido
                content = re.sub(r'\n+', ' ', content)
                content = re.sub(r'\s+', ' ', content)
                
                if len(content) > 20:
                    sections.append({
                        "section_number": number,
                        "content": content,
                        "type": section_type
                    })
                    seen_sections.add(section_id)
        
        logger.info(f"Se encontraron {len(sections)} secciones únicas")
        return sections
    
    def process_pdf(self) -> Dict[str, List[Dict]]:
        """Procesa completamente el PDF y retorna los segmentos"""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo PDF: {self.pdf_path}")
        
        logger.info("Iniciando procesamiento del PDF...")
        
        # Extraer texto
        full_text = self.extract_text_from_pdf()
        
        # Segmentar por artículos
        articles = self.segment_by_articles(full_text)
        
        # Extraer secciones adicionales
        sections = self.extract_chapters_and_titles(full_text)
        
        # Combinar todo
        processed_data = {
            "articles": articles,
            "sections": sections,
            "metadata": {
                "total_articles": len(articles),
                "total_sections": len(sections),
                "total_characters": len(full_text)
            }
        }
        
        # Guardar resultado
        output_file = self.output_dir / "processed_law.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Procesamiento completado. Datos guardados en: {output_file}")
        return processed_data

def main():
    """Función principal para testing"""
    processor = LawPDFProcessor()
    
    try:
        result = processor.process_pdf()
        print(f"✅ Procesamiento exitoso:")
        print(f"   - Artículos encontrados: {result['metadata']['total_articles']}")
        print(f"   - Secciones encontradas: {result['metadata']['total_sections']}")
        print(f"   - Caracteres totales: {result['metadata']['total_characters']}")
        
        # Mostrar algunos ejemplos
        if result['articles']:
            print(f"\n📄 Ejemplo de artículo:")
            article = result['articles'][0]
            print(f"   Artículo {article['article_number']}: {article['content'][:200]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
