"""
Passo 2: Query Transformation via HyDE (Hypothetical Document Embeddings)

Em vez de vetorizar a query coloquial do paciente diretamente, pedimos ao LLM
para alucinar um trecho de manual médico técnico que responderia à pergunta.
Esse Documento Hipotético vive no mesmo espaço vetorial dos documentos reais,
eliminando a lacuna semântica entre linguagem coloquial e jargão técnico.

Modos de LLM (auto-detectado via variável de ambiente):
  - GROQ_API_KEY definida → usa Groq API (llama-3.1-8b-instant) — melhor qualidade
  - Sem chave             → usa google/flan-t5-base local — zero configuração
"""

import os
from dotenv import load_dotenv

load_dotenv()

_HYDE_SYSTEM_PROMPT = (
    "Você é um médico especialista com amplo conhecimento clínico. "
    "Dado o relato coloquial de um paciente, escreva um trecho de manual médico técnico "
    "que descreveria esse quadro clínico com precisão. Use terminologia médica especializada: "
    "nomes clínicos dos sintomas, diagnósticos diferenciais relevantes, fisiopatologia e "
    "dados epidemiológicos pertinentes. Escreva diretamente o texto do manual, "
    "sem introduções, títulos ou explicações adicionais. Máximo 200 palavras."
)


# ── Groq (API) ────────────────────────────────────────────────────────────────

_groq_client = None


def _generate_via_groq(query: str) -> str:
    global _groq_client
    from groq import Groq

    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    model_name = "llama-3.1-8b-instant"
    print(f"[HyDE] Modo: Groq API ({model_name})")

    response = _groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Relato do paciente: {query}"},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ── flan-t5 local (fallback) ──────────────────────────────────────────────────

_local_model = None
_local_tokenizer = None


def _generate_via_local(query: str) -> str:
    global _local_model, _local_tokenizer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    if _local_model is None:
        model_name = "google/flan-t5-base"
        print(f"[HyDE] Modo: modelo local ({model_name}) — sem API key necessária")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _local_model.eval()

    prompt = (
        "Você é um médico especialista. Escreva um trecho técnico de manual médico "
        "descrevendo o quadro clínico abaixo com terminologia médica precisa.\n\n"
        f"Relato do paciente: {query}\n\n"
        "Trecho do manual médico:"
    )

    inputs = _local_tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True
    )
    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            num_beams=4,
        )
    return _local_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ── Interface pública ─────────────────────────────────────────────────────────


def generate_hypothetical_document(query: str) -> str:
    """
    Recebe query coloquial → retorna Documento Hipotético em linguagem técnica.

    HyDE funciona porque o embedding do documento hipotético técnico está
    geometricamente mais próximo dos embeddings dos documentos reais do corpus
    do que o embedding da query coloquial original.
    """
    print(f"\n{'='*60}")
    print(f"[HyDE] Query coloquial do paciente:")
    print(f"  '{query}'")
    print(f"[HyDE] Gerando Documento Hipotético via LLM...")

    if os.getenv("GROQ_API_KEY"):
        hypothetical_doc = _generate_via_groq(query)
    else:
        hypothetical_doc = _generate_via_local(query)

    print(f"\n[HyDE] Documento Hipotético gerado (âncora geométrica):")
    print(f"  {hypothetical_doc}")
    print(f"{'='*60}\n")

    return hypothetical_doc
