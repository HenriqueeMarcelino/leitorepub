import streamlit as st
from ebooklib import epub
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import tempfile, os, re, asyncio, io

# TTS: neural (Edge) e fallback offline
import edge_tts
import pyttsx3

st.set_page_config(page_title="Leitor EPUB + Audiobook + Tradução", layout="wide")

# -----------------------------
# Utilidades
# -----------------------------
def limpar_texto(texto: str) -> str:
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r'[ \t]+', ' ', texto)
    texto = re.sub(r' +\n', '\n', texto)
    return texto.strip()

def chunk_text(text: str, max_chars: int = 1800):
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    pages, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                pages.append(cur)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    pages.append(p[i:i+max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        pages.append(cur)
    return pages

def translate_block(text: str, src: str, dest: str):
    translator = GoogleTranslator(source=src, target=dest)
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 4500:
            chunks.append(cur)
            cur = line
        else:
            cur = (cur + "\n" + line) if cur else line
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:
        out.append(translator.translate(c))
    return "\n".join(out)

# -----------------------------
# EPUB
# -----------------------------
@st.cache_data(show_spinner=False)
def load_epub_from_bytes(file_bytes: bytes):
    # Algumas versões do ebooklib preferem caminho de arquivo
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

    # Ordem do spine
    ordered_items = []
    for spine_item in book.spine:
        item_id = spine_item[0]
        it = book.get_item_with_id(item_id)
        if it is not None:
            ordered_items.append(it)

    # Adiciona EpubHtml fora do spine
    in_spine_ids = {it.get_id() for it in ordered_items}
    html_items = [it for it in book.get_items() if isinstance(it, epub.EpubHtml)]
    ordered_items += [it for it in html_items if it.get_id() not in in_spine_ids]

    chapters = []
    for it in ordered_items:
        if not isinstance(it, epub.EpubHtml):
            continue

        soup = BeautifulSoup(it.get_content(), "html.parser")

        # remove tags indesejadas
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        title = soup.title.string.strip() if soup.title and soup.title.string else it.get_name()

        if text:
            chapters.append((title, text))

    return chapters

# -----------------------------
# Edge-TTS (vozes neurais)
# -----------------------------
async def _edge_list_voices():
    return await edge_tts.list_voices()

@st.cache_data(show_spinner=False, ttl=24*3600)
def get_edge_voices(locales=("pt-BR", "pt-PT")):
    # Chama o método assíncrono uma única vez e guarda em cache
    voices = asyncio.run(_edge_list_voices())
    filtered = [v for v in voices if any(v["ShortName"].startswith(loc) for loc in locales)]
    # Monta tuplas (label, shortname)
    options = []
    for v in filtered:
        friendly = v.get("FriendlyName") or v["ShortName"]
        label = f'{friendly}  ·  {v["ShortName"]}'
        options.append((label, v["ShortName"]))
    # Ordena estável
    options.sort(key=lambda x: x[1])
    return options

async def _edge_tts_save(text: str, voice: str, rate_pct: int, outfile: str):
    rate = f"{rate_pct:+d}%"
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(outfile)

def tts_edge_to_mp3(text: str, voice: str, rate_pct: int = 0):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tf.close()
    asyncio.run(_edge_tts_save(text, voice, rate_pct, tf.name))
    with open(tf.name, "rb") as f:
        data = f.read()
    return data, tf.name

# -----------------------------
# Fallback offline (pyttsx3)
# -----------------------------
def tts_pyttsx3_to_wav(text: str, rate_wpm: int = 180, voice_hint: str | None = None):
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate_wpm)
        if voice_hint:
            for v in engine.getProperty("voices"):
                if voice_hint.lower() in (v.id.lower() + " " + v.name.lower()):
                    engine.setProperty("voice", v.id)
                    break
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tf.close()
        engine.save_to_file(text, tf.name)
        engine.runAndWait()
        with open(tf.name, "rb") as f:
            data = f.read()
        return data, tf.name
    finally:
        try:
            engine.stop()
        except Exception:
            pass

# -----------------------------
# UI
# -----------------------------
st.title("Leitor de EPUB com Audiobook e Tradução")

with st.sidebar:
    st.header("Carregar EPUB")
    uploaded = st.file_uploader("Selecione um arquivo .epub", type=["epub"])
    max_chars = st.slider("Tamanho da página (caracteres)", 800, 4000, 1800, 100)
    st.divider()

    st.header("Audiobook (TTS)")
    engine_choice = st.radio("Motor de voz", ["Edge-TTS (neural)", "pyttsx3 (offline)"], index=0)

    if engine_choice == "Edge-TTS (neural)":
        # Lista automática de vozes pt-BR e pt-PT
        voice_options = get_edge_voices(locales=("pt-BR", "pt-PT"))
        if voice_options:
            default_idx = next((i for i, (_, short) in enumerate(voice_options) if short.startswith("pt-BR-Antonio")), 0)
            sel = st.selectbox("Voz (neural)", options=list(range(len(voice_options))),
                               format_func=lambda i: voice_options[i][0], index=default_idx)
            voice_shortname = voice_options[sel][1]
        else:
            st.warning("Não foi possível listar vozes do Edge-TTS. Usando Antonio (pt-BR) por padrão.")
            voice_shortname = "pt-BR-AntonioNeural"
        rate_pct = st.slider("Velocidade (±%)", -50, 50, 0, 1)
        st.caption("Dica: no terminal, `python -m edge_tts --list-voices` lista todas as vozes disponíveis.")
    else:
        tts_rate = st.slider("Velocidade (palavras/min)", 120, 300, 180, 10)
        voice_hint = st.text_input("Filtro de voz offline (opcional)")

    st.divider()
    st.header("Tradução")
    do_translate = st.checkbox("Traduzir texto exibido", value=False)
    col_lang = st.columns(2)
    with col_lang[0]:
        src_lang = st.text_input("De (ISO)", value="auto")
    with col_lang[1]:
        dest_lang = st.text_input("Para (ISO)", value="pt")

if uploaded is not None:
    try:
        chapters = load_epub_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Falha ao ler EPUB: {e}")
        st.stop()

    if not chapters:
        st.warning("Nenhum conteúdo de texto encontrado no EPUB.")
        st.stop()

    chapter_titles = [c[0] for c in chapters]
    st.subheader("Capítulo")
    chapter_idx = st.selectbox("Selecione o capítulo", options=list(range(len(chapters))),
                               format_func=lambda i: chapter_titles[i], index=0)

    title, raw_text = chapters[chapter_idx]
    pages = chunk_text(raw_text, max_chars=max_chars)

    st.subheader(title)
    page_idx = st.number_input("Página", min_value=1, max_value=len(pages), value=1, step=1)
    page_text = limpar_texto(pages[page_idx - 1])

    if do_translate:
        with st.spinner("Traduzindo..."):
            try:
                page_text = translate_block(page_text, src_lang, dest_lang)
            except Exception as e:
                st.error(f"Tradução falhou: {e}")

    st.text_area("Texto", value=page_text, height=420)

    col = st.columns(2)
    with col[0]:
        if st.button("Ler página (Audiobook)"):
            with st.spinner("Gerando áudio..."):
                try:
                    if engine_choice == "Edge-TTS (neural)":
                        audio_bytes, tmp_path = tts_edge_to_mp3(page_text, voice_shortname, rate_pct)
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        audio_bytes, tmp_path = tts_pyttsx3_to_wav(page_text, rate_wpm=tts_rate, voice_hint=voice_hint or None)
                        st.audio(audio_bytes, format="audio/wav")
                    st.success("Áudio gerado.")
                    st.caption(f"Arquivo temporário: {tmp_path}")
                except Exception as e:
                    st.error(f"Falha ao sintetizar áudio: {e}")

    with col[1]:
        st.download_button(
            "Baixar texto desta página (.txt)",
            data=page_text,
            file_name=f"cap_{chapter_idx+1}_pag_{page_idx}.txt"
        )
else:
    st.info("Envie um arquivo .epub na barra lateral para começar.")
