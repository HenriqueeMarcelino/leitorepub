import streamlit as st
from ebooklib import epub
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import tempfile
import io
import re
import os
import base64
from PIL import Image
import json
from pathlib import Path
import asyncio
import edge_tts
import pyttsx3
from datetime import datetime

st.set_page_config(page_title="Biblioteca Digital EPUB", layout="wide", initial_sidebar_state="expanded")

# Diretório para armazenar livros
BOOKS_DIR = Path("epub_library")
BOOKS_DIR.mkdir(exist_ok=True)

# Vozes Edge-TTS testadas e funcionais para português brasileiro
PORTUGUESE_VOICES = {
    "Antônio (Masculino)": "pt-BR-AntonioNeural",
    "Francisca (Feminino)": "pt-BR-FranciscaNeural",
    "Thalita (Feminino, Multilingual)": "pt-BR-ThalitaMultilingualNeural"
}


# Lista completa de vozes para testar (algumas podem não funcionar)
ALL_PORTUGUESE_VOICES = {
    "Antônio (Masculino)": "pt-BR-AntonioNeural",
    "Brenda (Feminino)": "pt-BR-BrendaNeural", 
    "Donato (Masculino)": "pt-BR-DonatoNeural",
    "Elza (Feminino)": "pt-BR-ElzaNeural",
    "Fabio (Masculino)": "pt-BR-FabioNeural",
    "Francisca (Feminino)": "pt-BR-FranciscaNeural",
    "Giovanni (Masculino)": "pt-BR-GiovanniNeural",
    "Humberto (Masculino)": "pt-BR-HumbertoNeural",
    "Julio (Masculino)": "pt-BR-JulioNeural",
    "Leila (Feminino)": "pt-BR-LeilaNeural",
    "Leticia (Feminino)": "pt-BR-LeticiaNeural",
    "Manuela (Feminino)": "pt-BR-ManuelaNeural",
    "Nicolas (Masculino)": "pt-BR-NicolasNeural",
    "Thalita (Feminino)": "pt-BR-ThalitaNeural",
    "Valeria (Feminino)": "pt-BR-ValeriaNeural",
    "Yara (Feminino)": "pt-BR-YaraNeural"
}

async def test_voice_availability(voice_code: str) -> bool:
    """Testa se uma voz está disponível"""
    try:
        test_text = "Teste"
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tf.close()
        
        communicate = edge_tts.Communicate(text=test_text, voice=voice_code)
        await communicate.save(tf.name)
        
        # Verificar se o arquivo foi criado e tem conteúdo
        success = os.path.exists(tf.name) and os.path.getsize(tf.name) > 0
        
        try:
            os.unlink(tf.name)
        except:
            pass
            
        return success
    except:
        return False

def get_available_voices():
    """Retorna apenas as vozes que estão funcionando"""
    if 'available_voices' not in st.session_state:
        st.session_state.available_voices = PORTUGUESE_VOICES.copy()  # Começar com as vozes testadas
    return st.session_state.available_voices

@st.cache_data(show_spinner=False)
def load_epub_from_bytes(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tf:
        tf.write(file_bytes)
        temp_path = tf.name
    
    try:
        book = epub.read_epub(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    # Extrair metadados
    metadata = {
        'title': book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Título Desconhecido",
        'author': book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else "Autor Desconhecido",
        'language': book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else "pt",
        'description': book.get_metadata('DC', 'description')[0][0] if book.get_metadata('DC', 'description') else "",
        'publisher': book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else "",
        'date': book.get_metadata('DC', 'date')[0][0] if book.get_metadata('DC', 'date') else "",
    }

    # Extrair capa
    cover_image = None
    for item in book.get_items():
        if item.get_name().lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            try:
                img_data = item.get_content()
                img = Image.open(io.BytesIO(img_data))
                cover_image = img
                break
            except:
                continue

    # Extrair capítulos com imagens
    ordered_items = []
    for spine_item in book.spine:
        item_id = spine_item[0]
        it = book.get_item_with_id(item_id)
        if it is not None:
            ordered_items.append(it)

    in_spine_ids = {it.get_id() for it in ordered_items}
    html_items = [it for it in book.get_items() if isinstance(it, epub.EpubHtml)]
    ordered_items += [it for it in html_items if it.get_id() not in in_spine_ids]

    chapters = []
    images = {}
    
    # Primeiro, coletamos todas as imagens
    for item in book.get_items():
        if item.get_type() == "image":
            images[item.get_name()] = item.get_content()

    for it in ordered_items:
        if not isinstance(it, epub.EpubHtml):
            continue

        soup = BeautifulSoup(it.get_content(), "html.parser")
        
        # Processar imagens no capítulo
        chapter_images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src in images:
                chapter_images.append({
                    'src': src,
                    'alt': img.get('alt', ''),
                    'data': images[src]
                })

        # Limpar tags indesejadas mas preservar estrutura
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        # Melhor extração de texto preservando formatação
        full_text = extract_clean_text(soup)
        
        # Título do capítulo
        title = soup.title.string.strip() if soup.title and soup.title.string else it.get_name()
        
        if full_text:
            chapters.append({
                'title': title,
                'text': full_text,
                'images': chapter_images,
                'word_count': len(full_text.split())
            })

    return {
        'metadata': metadata,
        'cover': cover_image,
        'chapters': chapters
    }

def extract_clean_text(soup):
    """Extrai texto preservando melhor a formatação e pontuação"""
    # Remover elementos desnecessários
    for element in soup(["script", "style", "noscript", "meta", "link"]):
        element.decompose()
    
    # Adicionar quebras de linha antes de elementos de bloco
    for element in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'li']):
        if element.name == 'br':
            element.replace_with('\n')
        else:
            element.insert_before('\n')
            element.insert_after('\n')
    
    # Extrair texto
    text = soup.get_text()
    
    # Limpeza aprimorada
    text = clean_extracted_text(text)
    
    return text

def clean_extracted_text(text):
    """Limpa o texto extraído mantendo a formatação adequada"""
    if not text:
        return ""
    
    # Normalizar quebras de linha
    text = re.sub(r'\r\n?', '\n', text)
    
    # Remover espaços em excesso no início e fim das linhas
    lines = [line.strip() for line in text.split('\n')]
    
    # Reconstruir o texto
    cleaned_lines = []
    for line in lines:
        if line:  # Se a linha não está vazia
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1]:  # Adicionar linha vazia apenas se a anterior não for vazia
            cleaned_lines.append('')
    
    text = '\n'.join(cleaned_lines)
    
    # Corrigir espaçamento entre palavras
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Corrigir problemas comuns de formatação
    text = re.sub(r'([.!?])\s*([A-ZÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙÃÕÇ])', r'\1 \2', text)  # Espaço após pontuação
    text = re.sub(r'([a-záéíóúâêîôûàèìòùãõç])([A-ZÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙÃÕÇ])', r'\1 \2', text)  # Espaço entre minúscula e maiúscula
    text = re.sub(r'\.{2,}', '...', text)  # Normalizar reticências
    text = re.sub(r'([.!?])\n+([a-záéíóúâêîôûàèìòùãõç])', r'\1\n\n\2', text)  # Quebra após pontuação final
    
    # Remover múltiplas quebras de linha consecutivas (máximo 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remover espaços antes de quebras de linha
    text = re.sub(r' +\n', '\n', text)
    
    return text.strip()

def chunk_text(text: str, max_chars: int = 1800):
    """Divide o texto em páginas respeitando parágrafos e frases"""
    if not text:
        return [""]
    
    # Dividir em parágrafos
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return [text]
    
    pages = []
    current_page = ""
    
    for paragraph in paragraphs:
        # Se o parágrafo sozinho é maior que o limite
        if len(paragraph) > max_chars:
            # Salvar página atual se não estiver vazia
            if current_page:
                pages.append(current_page.strip())
                current_page = ""
            
            # Dividir o parágrafo em frases
            sentences = re.split(r'([.!?]+\s+)', paragraph)
            temp_para = ""
            
            for sentence in sentences:
                if len(temp_para + sentence) <= max_chars:
                    temp_para += sentence
                else:
                    if temp_para:
                        pages.append(temp_para.strip())
                    temp_para = sentence
            
            if temp_para:
                current_page = temp_para
        else:
            # Se adicionar este parágrafo ultrapassaria o limite
            if len(current_page + "\n\n" + paragraph) > max_chars:
                if current_page:
                    pages.append(current_page.strip())
                current_page = paragraph
            else:
                if current_page:
                    current_page += "\n\n" + paragraph
                else:
                    current_page = paragraph
    
    # Adicionar última página se não estiver vazia
    if current_page:
        pages.append(current_page.strip())
    
    return pages if pages else [""]

def prepare_text_for_tts(text: str) -> str:
    """Prepara o texto para TTS com melhor pronúncia"""
    if not text:
        return ""
    
    # Remover texto muito curto ou inválido
    text = text.strip()
    if len(text) < 3:
        return "Texto muito curto para síntese de voz."
    
    # Expandir abreviações comuns
    abbreviations = {
        r'\bDr\.\s': 'Doutor ',
        r'\bDra\.\s': 'Doutora ',
        r'\bSr\.\s': 'Senhor ',
        r'\bSra\.\s': 'Senhora ',
        r'\bProf\.\s': 'Professor ',
        r'\bProfa\.\s': 'Professora ',
        r'\betc\.': 'etcetera',
        r'\bpág\.\s': 'página ',
        r'\bp\.\s': 'página ',
        r'\bvol\.\s': 'volume ',
        r'\bcap\.\s': 'capítulo ',
        r'\bex\.\s': 'exemplo ',
        r'\bobs\.\s': 'observação '
    }
    
    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)
    
    # Melhorar pontuação para TTS
    text = re.sub(r'([.!?])\s*\n', r'\1 ', text)  # Substituir quebras após pontuação por pausa
    text = re.sub(r'\n+', '. ', text)  # Quebras de linha viram pausas
    text = re.sub(r'\.{3,}', '... ', text)  # Reticências
    text = re.sub(r'--+', ' - ', text)  # Travessões
    
    # Limpar caracteres problemáticos para TTS
    text = re.sub(r'[""''`´]', '"', text)  # Normalizar aspas
    text = re.sub(r'[–—]', '-', text)  # Normalizar traços
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\"\'\(\)\/\%\$\@\#]', ' ', text)  # Remover caracteres especiais problemáticos
    
    # Limpar espaços duplos
    text = re.sub(r'\s+', ' ', text)
    
    # Verificar se o texto final não está vazio
    text = text.strip()
    if not text:
        return "Texto não pôde ser processado para síntese de voz."
    
    return text

async def _edge_tts_save(text: str, voice: str, rate_pct: int, outfile: str):
    """Salva áudio usando Edge TTS com configurações otimizadas"""
    try:
        rate = f"{rate_pct:+d}%"
        communicate = edge_tts.Communicate(
            text=text, 
            voice=voice, 
            rate=rate
        )
        await communicate.save(outfile)
    except Exception as e:
        raise Exception(f"Falha na comunicação Edge TTS: {str(e)}")

def tts_edge_to_mp3(text: str, voice: str = "pt-BR-AntonioNeural", rate_pct: int = 0):
    """Converte texto em áudio usando Edge TTS"""
    if not text or not text.strip():
        raise Exception("Texto vazio fornecido para TTS")
    
    # Preparar e limitar texto para TTS
    prepared_text = prepare_text_for_tts(text)
    
    # Limitar tamanho do texto (Edge TTS tem limite)
    if len(prepared_text) > 5000:
        prepared_text = prepared_text[:4500] + "..."
        st.warning("⚠️ Texto muito longo, foi truncado para o áudio.")
    
    if not prepared_text.strip():
        raise Exception("Texto preparado está vazio")
    
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tf.close()
    
    try:
        # Verificar se já existe um loop asyncio rodando
        try:
            loop = asyncio.get_running_loop()
            # Se já existe um loop, usar thread separada
            import threading
            import concurrent.futures
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(_edge_tts_save(prepared_text, voice, rate_pct, tf.name))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                future.result(timeout=30)  # Timeout de 30 segundos
                
        except RuntimeError:
            # Não existe loop rodando, pode usar asyncio.run normalmente
            asyncio.run(_edge_tts_save(prepared_text, voice, rate_pct, tf.name))
        
        # Verificar se o arquivo foi criado e tem conteúdo
        if not os.path.exists(tf.name) or os.path.getsize(tf.name) == 0:
            raise Exception("Arquivo de áudio não foi gerado ou está vazio")
        
        with open(tf.name, "rb") as f:
            data = f.read()
        
        if len(data) == 0:
            raise Exception("Arquivo de áudio está vazio")
            
        return data, tf.name
        
    except Exception as e:
        error_msg = str(e)
        if "No audio was received" in error_msg:
            raise Exception("Falha na geração de áudio. Tente uma voz diferente ou verifique sua conexão com a internet.")
        elif "timeout" in error_msg.lower():
            raise Exception("Timeout na geração de áudio. Tente um texto menor ou verifique sua conexão.")
        else:
            raise Exception(f"Erro no Edge TTS: {error_msg}")
    finally:
        try:
            if os.path.exists(tf.name):
                os.unlink(tf.name)
        except:
            pass

def tts_pyttsx3_to_wav(text: str, rate_wpm: int = 180, voice_hint: str | None = None):
    """Converte texto em áudio usando pyttsx3"""
    prepared_text = prepare_text_for_tts(text)
    
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate_wpm)
        engine.setProperty("volume", 0.9)
        
        if voice_hint:
            voices = engine.getProperty("voices")
            for v in voices:
                if voice_hint.lower() in (v.id.lower() + " " + v.name.lower()):
                    engine.setProperty("voice", v.id)
                    break
        
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tf.close()
        
        engine.save_to_file(prepared_text, tf.name)
        engine.runAndWait()
        
        with open(tf.name, "rb") as f:
            data = f.read()
        return data, tf.name
    except Exception as e:
        raise Exception(f"Erro no pyttsx3: {str(e)}")
    finally:
        try:
            engine.stop()
            os.unlink(tf.name)
        except:
            pass

def translate_block(text: str, src: str, dest: str):
    """Traduz texto dividindo em chunks menores"""
    if not text:
        return ""
        
    translator = GoogleTranslator(source=src, target=dest)
    
    # Dividir em parágrafos primeiro
    paragraphs = text.split('\n\n')
    translated_paragraphs = []
    
    current_chunk = ""
    chunk_paragraphs = []
    
    for paragraph in paragraphs:
        if len(current_chunk + paragraph) > 4000:  # Limite menor para evitar erros
            if current_chunk:
                try:
                    translated = translator.translate(current_chunk)
                    translated_paragraphs.extend(translated.split('\n\n'))
                except:
                    translated_paragraphs.extend(chunk_paragraphs)
                current_chunk = paragraph
                chunk_paragraphs = [paragraph]
            else:
                current_chunk = paragraph
                chunk_paragraphs = [paragraph]
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            chunk_paragraphs.append(paragraph)
    
    # Traduzir último chunk
    if current_chunk:
        try:
            translated = translator.translate(current_chunk)
            translated_paragraphs.extend(translated.split('\n\n'))
        except:
            translated_paragraphs.extend(chunk_paragraphs)
    
    return '\n\n'.join(translated_paragraphs)

def save_book_to_library(book_data, filename):
    """Salva um livro na biblioteca local"""
    book_path = BOOKS_DIR / f"{filename}.json"
    
    # Converter imagens para base64 para serialização
    serializable_data = book_data.copy()
    if book_data['cover']:
        img_buffer = io.BytesIO()
        book_data['cover'].save(img_buffer, format='PNG')
        serializable_data['cover'] = base64.b64encode(img_buffer.getvalue()).decode()
    else:
        serializable_data['cover'] = None
    
    # Converter imagens dos capítulos
    for chapter in serializable_data['chapters']:
        for img in chapter['images']:
            img['data'] = base64.b64encode(img['data']).decode()
    
    with open(book_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

def load_book_from_library(filename):
    """Carrega um livro da biblioteca local"""
    book_path = BOOKS_DIR / f"{filename}.json"
    if not book_path.exists():
        return None
    
    with open(book_path, 'r', encoding='utf-8') as f:
        book_data = json.load(f)
    
    # Converter base64 de volta para imagens
    if book_data['cover']:
        cover_data = base64.b64decode(book_data['cover'])
        book_data['cover'] = Image.open(io.BytesIO(cover_data))
    
    # Converter imagens dos capítulos
    for chapter in book_data['chapters']:
        for img in chapter['images']:
            img['data'] = base64.b64decode(img['data'])
    
    return book_data

def get_library_books():
    """Retorna lista de livros na biblioteca"""
    books = []
    for book_file in BOOKS_DIR.glob("*.json"):
        try:
            book_data = load_book_from_library(book_file.stem)
            if book_data:
                books.append({
                    'filename': book_file.stem,
                    'metadata': book_data['metadata'],
                    'cover': book_data['cover'],
                    'chapters_count': len(book_data['chapters']),
                    'word_count': sum(ch['word_count'] for ch in book_data['chapters'])
                })
        except Exception as e:
            st.error(f"Erro ao carregar {book_file.name}: {e}")
    return books

# Interface principal
st.title("📚 Biblioteca Digital EPUB")

# Inicializar sessão
if 'current_book' not in st.session_state:
    st.session_state.current_book = None
if 'current_chapter' not in st.session_state:
    st.session_state.current_chapter = 0
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'reading_progress' not in st.session_state:
    st.session_state.reading_progress = {}

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Upload de novo livro
    st.subheader("📖 Adicionar Livro")
    uploaded = st.file_uploader("Selecione um arquivo .epub", type=["epub"])
    
    if uploaded is not None:
        if st.button("➕ Adicionar à Biblioteca"):
            with st.spinner("Processando EPUB..."):
                try:
                    book_data = load_epub_from_bytes(uploaded.getvalue())
                    filename = uploaded.name.replace('.epub', '')
                    save_book_to_library(book_data, filename)
                    st.success("✅ Livro adicionado à biblioteca!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao processar EPUB: {e}")
    
    st.divider()
    
    # Configurações de leitura
    st.subheader("📄 Leitura")
    max_chars = st.slider("Tamanho da página", 800, 4000, 1800, 100)
    font_size = st.selectbox("Tamanho da fonte", ["Pequena", "Média", "Grande"], index=1)
    
    st.subheader("🎧 Audiobook")
    engine_choice = st.radio("Motor TTS", ["Edge-TTS (Recomendado)", "pyttsx3"], index=0)
    
    if engine_choice == "Edge-TTS (Recomendado)":
        available_voices = get_available_voices()
        
        voice_display = st.selectbox(
            "Voz", 
            list(available_voices.keys()),
            index=0
        )
        voice = available_voices[voice_display]
        rate_pct = st.slider("Velocidade (±%)", -50, 50, 0, 5)
        
        col_test, col_scan = st.columns(2)
        
        with col_test:
            # Teste de voz
            if st.button("🔊 Testar Voz"):
                test_text = "Olá! Esta é uma demonstração da voz selecionada para leitura de livros digitais."
                with st.spinner("Gerando teste de áudio..."):
                    try:
                        audio_bytes, tmp_path = tts_edge_to_mp3(test_text, voice=voice, rate_pct=rate_pct)
                        st.audio(audio_bytes, format="audio/mp3")
                        st.success("✅ Teste concluído!")
                    except Exception as e:
                        st.error(f"❌ Erro no teste: {e}")
                        st.info("💡 Esta voz pode não estar disponível. Tente outra ou use o botão 'Verificar Vozes'.")
        
        with col_scan:
            # Verificar todas as vozes disponíveis
            if st.button("🔍 Verificar Vozes"):
                with st.spinner("Testando todas as vozes disponíveis..."):
                    working_voices = {}
                    progress_bar = st.progress(0)
                    
                    for i, (name, code) in enumerate(ALL_PORTUGUESE_VOICES.items()):
                        try:
                            # Teste rápido
                            test_text = "Teste"
                            audio_bytes, tmp_path = tts_edge_to_mp3(test_text, voice=code, rate_pct=0)
                            if len(audio_bytes) > 0:
                                working_voices[name] = code
                                st.write(f"✅ {name}")
                        except:
                            st.write(f"❌ {name}")
                        
                        progress_bar.progress((i + 1) / len(ALL_PORTUGUESE_VOICES))
                    
                    st.session_state.available_voices = working_voices
                    st.success(f"🎯 Encontradas {len(working_voices)} vozes funcionais!")
                    st.rerun()
    else:
        tts_rate = st.slider("Velocidade (palavras/min)", 120, 300, 180, 10)
        voice_hint = st.text_input("Filtro de voz (opcional)")
    
    st.subheader("🌐 Tradução")
    do_translate = st.checkbox("Traduzir texto")
    if do_translate:
        col1, col2 = st.columns(2)
        with col1:
            src_lang = st.text_input("De", value="auto")
        with col2:
            dest_lang = st.text_input("Para", value="pt")

# CSS para melhorar a aparência
font_sizes = {"Pequena": "14px", "Média": "16px", "Grande": "18px"}
st.markdown(f"""
<style>
.reading-area {{
    font-size: {font_sizes[font_size]};
    line-height: 1.6;
    text-align: justify;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 10px;
    border: 1px solid #e9ecef;
}}
.book-card {{
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.book-title {{
    color: #212529;
    font-weight: bold;
    margin-bottom: 5px;
}}
.book-author {{
    color: #6c757d;
    font-style: italic;
}}
</style>
""", unsafe_allow_html=True)

# Tela principal
if st.session_state.current_book is None:
    # Mostrar biblioteca
    st.header("📚 Sua Biblioteca")
    
    library_books = get_library_books()
    
    if not library_books:
        st.info("📖 Sua biblioteca está vazia. Adicione um arquivo EPUB usando a barra lateral.")
        st.markdown("""
        ### Como usar:
        1. 📁 Faça upload de um arquivo .epub na barra lateral
        2. ➕ Clique em "Adicionar à Biblioteca" 
        3. 📖 Selecione o livro para começar a ler
        4. 🎧 Use o recurso de audiobook (clique em "🔍 Verificar Vozes" para encontrar vozes funcionais)
        5. 🌐 Traduza o conteúdo para outros idiomas se necessário
        
        ### 🎤 Sobre as Vozes:
        - **Edge TTS**: Não precisa baixar vozes, funciona via internet
        - **Vozes testadas**: Antônio, Francisca, Humberto, Thalita
        - **Verificação automática**: Use "🔍 Verificar Vozes" para testar todas
        """)
    else:
        # Mostrar livros em grid
        cols_per_row = 3
        for i in range(0, len(library_books), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, book in enumerate(library_books[i:i+cols_per_row]):
                with cols[j]:
                    # Card do livro
                    with st.container():
                        if book['cover']:
                            st.image(book['cover'], use_column_width=True)
                        else:
                            st.image("https://via.placeholder.com/300x400/cccccc/666666?text=Sem+Capa", use_column_width=True)
                        
                        st.markdown(f"<div class='book-title'>{book['metadata']['title']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='book-author'>por {book['metadata']['author']}</div>", unsafe_allow_html=True)
                        st.caption(f"📑 {book['chapters_count']} capítulos • 📝 {book['word_count']:,} palavras")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📖 Ler", key=f"open_{book['filename']}"):
                                st.session_state.current_book = book['filename']
                                st.session_state.current_chapter = 0
                                st.session_state.current_page = 1
                                st.rerun()
                        
                        with col2:
                            if st.button(f"🗑️", key=f"delete_{book['filename']}", help="Remover livro"):
                                try:
                                    os.remove(BOOKS_DIR / f"{book['filename']}.json")
                                    st.success("Livro removido!")
                                    st.rerun()
                                except:
                                    st.error("Erro ao remover livro")

else:
    # Mostrar leitor
    book_data = load_book_from_library(st.session_state.current_book)
    if book_data is None:
        st.error("❌ Livro não encontrado!")
        if st.button("🏠 Voltar à Biblioteca"):
            st.session_state.current_book = None
            st.rerun()
        st.stop()
    
    # Header do livro
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🏠 Biblioteca"):
            st.session_state.current_book = None
            st.rerun()
    
    with col2:
        st.title(book_data['metadata']['title'])
        st.caption(f"✍️ por {book_data['metadata']['author']}")
    
    with col3:
        if book_data['cover']:
            st.image(book_data['cover'], width=100)
    
    # Navegação de capítulos
    st.subheader("📖 Navegação")
    chapter_options = [f"{i+1}. {ch['title']}" for i, ch in enumerate(book_data['chapters'])]
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        chapter_idx = st.selectbox(
            "Capítulo", 
            range(len(book_data['chapters'])),
            format_func=lambda x: chapter_options[x],
            index=st.session_state.current_chapter
        )
    
    with col2:
        total_words = sum(ch['word_count'] for ch in book_data['chapters'])
        progress = (chapter_idx + 1) / len(book_data['chapters'])
        st.metric("📊 Progresso", f"{progress:.0%}")
    
    with col3:
        words_read = sum(book_data['chapters'][i]['word_count'] for i in range(chapter_idx + 1))
        est_time = words_read // 200  # ~200 palavras por minuto
        st.metric("⏱️ Tempo lido", f"~{est_time}min")
    
    if chapter_idx != st.session_state.current_chapter:
        st.session_state.current_chapter = chapter_idx
        st.session_state.current_page = 1
    
    current_chapter = book_data['chapters'][chapter_idx]
    pages = chunk_text(current_chapter['text'], max_chars=max_chars)
    
    # Navegação de páginas
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⬅️ Anterior") and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()
    
    with col2:
        if st.button("⏪ Primeiro") and st.session_state.current_page > 1:
            st.session_state.current_page = 1
            st.rerun()
    
    with col3:
        page_idx = st.number_input(
            "Página", 
            min_value=1, 
            max_value=len(pages), 
            value=st.session_state.current_page,
            key="page_input"
        )
        if page_idx != st.session_state.current_page:
            st.session_state.current_page = page_idx
    
    with col4:
        if st.button("⏩ Último") and st.session_state.current_page < len(pages):
            st.session_state.current_page = len(pages)
            st.rerun()
    
    with col5:
        if st.button("➡️ Próxima") and st.session_state.current_page < len(pages):
            st.session_state.current_page += 1
            st.rerun()
    
    st.caption(f"📄 Página {st.session_state.current_page} de {len(pages)} • 📝 {current_chapter['word_count']:,} palavras neste capítulo")
    
    # Mostrar imagens do capítulo
    if current_chapter['images']:
        with st.expander(f"🖼️ Imagens do capítulo ({len(current_chapter['images'])})", expanded=False):
            img_cols = st.columns(min(3, len(current_chapter['images'])))
            for i, img in enumerate(current_chapter['images']):
                with img_cols[i % 3]:
                    st.image(
                        io.BytesIO(img['data']), 
                        caption=img['alt'] or f"Imagem {i+1}",
                        use_column_width=True
                    )
    
    # Conteúdo da página
    page_text = pages[st.session_state.current_page - 1]
    
    # Aplicar tradução se solicitada
    if do_translate:
        with st.spinner("🌐 Traduzindo..."):
            try:
                page_text = translate_block(page_text, src_lang, dest_lang)
                st.success("✅ Texto traduzido!")
            except Exception as e:
                st.error(f"❌ Erro na tradução: {e}")
    
    # Área de leitura com estilo melhorado
    st.subheader("📖 Leitura")
    st.markdown(f'<div class="reading-area">{page_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    
    # Controles de áudio e outros recursos
    st.subheader("🎛️ Controles")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎧 Ler Página"):
            with st.spinner("🔊 Gerando áudio..."):
                try:
                    if engine_choice == "Edge-TTS (Recomendado)":
                        # Verificar se há texto suficiente
                        if len(page_text.strip()) < 10:
                            st.warning("⚠️ Texto muito curto para síntese de voz.")
                        else:
                            audio_bytes, tmp_path = tts_edge_to_mp3(page_text, voice=voice, rate_pct=rate_pct)
                            st.audio(audio_bytes, format="audio/mp3", start_time=0)
                            st.success("🎵 Áudio gerado com sucesso!")
                    else:
                        if len(page_text.strip()) < 10:
                            st.warning("⚠️ Texto muito curto para síntese de voz.")
                        else:
                            audio_bytes, tmp_path = tts_pyttsx3_to_wav(page_text, rate_wpm=tts_rate, voice_hint=voice_hint or None)
                            st.audio(audio_bytes, format="audio/wav", start_time=0)
                            st.success("🎵 Áudio gerado!")
                except Exception as e:
                    st.error(f"❌ Erro no TTS: {e}")
                    if "internet" in str(e).lower() or "connection" in str(e).lower():
                        st.info("💡 Dica: Verifique sua conexão com a internet.")
                    elif "timeout" in str(e).lower():
                        st.info("💡 Dica: Tente um texto menor ou aguarde alguns segundos.")
    
    with col2:
        # Preparar texto para download
        download_filename = f"{book_data['metadata']['title']}_cap{chapter_idx+1}_pag{st.session_state.current_page}.txt"
        download_filename = re.sub(r'[<>:"/\\|?*]', '_', download_filename)  # Limpar caracteres inválidos
        
        st.download_button(
            "💾 Baixar Texto", 
            data=page_text.encode('utf-8'), 
            file_name=download_filename,
            mime='text/plain'
        )
    
    with col3:
        if st.button("📊 Estatísticas"):
            reading_time_minutes = current_chapter['word_count'] / 200  # ~200 palavras por minuto
            
            st.info(f"""
            **📈 Estatísticas do Capítulo:**
            - 📝 Palavras: {current_chapter['word_count']:,}
            - 📄 Páginas: {len(pages)}
            - ⏱️ Tempo estimado de leitura: ~{reading_time_minutes:.1f} min
            - 🖼️ Imagens: {len(current_chapter['images'])}
            
            **📚 Estatísticas do Livro:**
            - 📑 Capítulos: {len(book_data['chapters'])}
            - 📝 Palavras totais: {total_words:,}
            - 🌍 Idioma: {book_data['metadata']['language']}
            - 📖 Editora: {book_data['metadata']['publisher'] or 'N/A'}
            - 📅 Data: {book_data['metadata']['date'] or 'N/A'}
            """)
    
    with col4:
        if st.button("🔄 Atualizar"):
            st.rerun()
    
    # Barra de progresso visual
    progress_percentage = (st.session_state.current_page / len(pages)) * 100
    st.progress(progress_percentage / 100, text=f"Progresso do capítulo: {progress_percentage:.1f}%")
    
    # Navegação rápida entre capítulos
    st.subheader("⚡ Navegação Rápida")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if st.button("⬅️ Capítulo Anterior") and chapter_idx > 0:
            st.session_state.current_chapter = chapter_idx - 1
            st.session_state.current_page = 1
            st.rerun()
    
    with nav_col2:
        st.write(f"📖 **Capítulo {chapter_idx + 1}**: {current_chapter['title'][:50]}{'...' if len(current_chapter['title']) > 50 else ''}")
    
    with nav_col3:
        if st.button("➡️ Próximo Capítulo") and chapter_idx < len(book_data['chapters']) - 1:
            st.session_state.current_chapter = chapter_idx + 1
            st.session_state.current_page = 1
            st.rerun()
    
    # Seção de notas (opcional)
    with st.expander("📝 Suas Anotações", expanded=False):
        note_key = f"note_{st.session_state.current_book}_{chapter_idx}_{st.session_state.current_page}"
        
        if note_key not in st.session_state:
            st.session_state[note_key] = ""
        
        note_text = st.text_area(
            "Faça suas anotações sobre esta página:",
            value=st.session_state[note_key],
            height=100,
            key=f"note_input_{note_key}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Salvar Anotação"):
                st.session_state[note_key] = note_text
                st.success("✅ Anotação salva!")
        
        with col2:
            if st.button("🗑️ Limpar Anotação"):
                st.session_state[note_key] = ""
                st.success("🗑️ Anotação removida!")
                st.rerun()

# Rodapé com informações
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.8em;'>
    📚 Biblioteca Digital EPUB - Versão Melhorada<br>
    🎧 Suporte a vozes neurais realistas • 🌐 Tradução automática • 📝 Sistema de anotações
</div>
""", unsafe_allow_html=True)