"""Voce di Sophon-Realtime Flash usata da sola, in streaming, su GPU AMD (ROCm).

Lo stadio voce di Sophon-Realtime Flash è un Qwen3-TTS 1.7B affinato: qui gira senza vLLM-Omni,
con faster-qwen3-tts (grafi + cache statica), cache del parlatore accorciata e torch.compile
su predittore e parlatore. Una frase alla volta: i testi lunghi altrimenti vengono troncati.

    python voce_veloce.py --modello flash-model/talker --voce voci/sara-it.pt \
        --testo "Ciao, come stai?" --uscita prova.wav

Provato con faster-qwen3-tts 0.3.2, qwen-tts 0.1.1, torch 2.13 + ROCm 7.15, Radeon 8060S.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from typing import Iterator

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np
import torch


def carica(modello: str, cache: int = 384, compila: bool = True, dtype=torch.bfloat16):
    """Carica il modello e prepara i grafi. `cache` è la lunghezza massima (in fotogrammi da 80 ms,
    testo compreso) di una frase: più è corta, più il parlatore è veloce."""
    from faster_qwen3_tts import FasterQwen3TTS

    m = FasterQwen3TTS.from_pretrained(modello, device="cuda", dtype=dtype, max_seq_len=cache)
    if compila:
        # Nomi interni di faster-qwen3-tts 0.3.2: vanno sostituiti prima della cattura dei grafi.
        m.predictor_graph.pred_model = torch.compile(m.predictor_graph.pred_model)
        m.talker_graph.model = torch.compile(m.talker_graph.model)
    m.warmup(prefill_len=100)
    m._cache_frase = cache
    return m


def voce_da_audio(m, wav: str) -> torch.Tensor:
    """Ricava il vettore della voce (x-vector) da 3-10 secondi di parlato pulito."""
    import soundfile as sf

    a, sr = sf.read(wav, dtype="float32", always_2d=True)
    a = a.mean(1)
    a = a / (np.abs(a).max() + 1e-9) * 0.3
    item = m.model.create_voice_clone_prompt(ref_audio=(a, sr), x_vector_only_mode=True)[0]
    return item.ref_spk_embedding.detach().float().cpu()


def frasi(testo: str, massimo: int = 220) -> list[str]:
    """Divide in frasi; quelle troppo lunghe vengono spezzate alle virgole."""
    out: list[str] = []
    for f in re.split(r"(?<=[.!?…])\s+", testo.strip()):
        f = f.strip()
        while len(f) > massimo:
            taglio = f.rfind(",", 0, massimo)
            taglio = taglio if taglio > 40 else f.rfind(" ", 0, massimo)
            if taglio <= 0:
                break
            out.append(f[: taglio + 1].strip())
            f = f[taglio + 1:].strip()
        if f:
            out.append(f)
    return out


def parla(m, voce: torch.Tensor, testo: str, lingua: str = "Italian", blocco: int = 4,
          pausa: float = 0.25, **gen) -> Iterator[tuple[np.ndarray, int]]:
    """Genera l'audio a pezzi (float32, 24 kHz). `blocco` = fotogrammi per pezzo: 4 ≈ 0,3 s di audio."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=voce.to("cuda", m.dtype),
                                x_vector_only_mode=True, icl_mode=False, ref_text=None)
    opz = dict(temperature=0.7, top_k=50, repetition_penalty=1.05,
               max_new_tokens=max(60, min(600, getattr(m, "_cache_frase", 384) - 120)))
    opz.update(gen)
    fs = 24000
    for f in frasi(testo):
        for pezzo, fs, _ in m.generate_voice_clone_streaming(
                text=f, language=lingua, voice_clone_prompt=[item], xvec_only=True,
                non_streaming_mode=False, chunk_size=blocco, **opz):
            yield np.asarray(pezzo, dtype=np.float32), fs
        yield np.zeros(int(fs * pausa), np.float32), fs


def _main() -> None:
    import soundfile as sf

    p = argparse.ArgumentParser(description="Voce di Sophon Flash in streaming, con misura dei tempi")
    p.add_argument("--modello", required=True, help="cartella «talker» di Sophon-Realtime Flash")
    p.add_argument("--voce", required=True, help="vettore .pt, oppure un WAV da cui ricavarlo")
    p.add_argument("--testo", required=True)
    p.add_argument("--uscita", default="voce.wav")
    p.add_argument("--lingua", default="Italian")
    p.add_argument("--blocco", type=int, default=4)
    p.add_argument("--cache", type=int, default=384)
    p.add_argument("--senza-compilazione", action="store_true")
    a = p.parse_args()

    t = time.time()
    m = carica(a.modello, cache=a.cache, compila=not a.senza_compilazione)
    print(f"modello pronto in {time.time() - t:.1f} s")
    voce = (torch.load(a.voce, map_location="cpu", weights_only=True).float()
            if a.voce.endswith(".pt") else voce_da_audio(m, a.voce))
    for _ in parla(m, voce, "Prova."):   # un giro a vuoto: la prima frase paga la messa a punto
        pass
    torch.cuda.synchronize()
    t = time.time(); primo = None; pezzi = []
    for pezzo, fs in parla(m, voce, a.testo, lingua=a.lingua, blocco=a.blocco):
        if primo is None:
            primo = time.time() - t
        pezzi.append(pezzo)
    d = time.time() - t
    audio = np.concatenate(pezzi)
    sf.write(a.uscita, audio, fs)
    print(f"primo audio dopo {primo:.2f} s | audio {len(audio) / fs:.1f} s | calcolo {d:.1f} s | "
          f"RTF {d / (len(audio) / fs):.2f} | GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    _main()
