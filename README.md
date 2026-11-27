# sophon-flash-voce-amd

La voce di **Sophon-Realtime Flash** usata da sola, in italiano e in streaming, su una GPU integrata AMD: primo suono dopo **0,2 secondi**, audio generato al **doppio del tempo reale**. Senza vLLM-Omni e senza NVIDIA.

*The speech stage of Sophon-Realtime Flash running standalone on an AMD integrated GPU (ROCm): 0.2 s to first audio, 2× real time, Italian included. No vLLM-Omni, no NVIDIA. Docs are in Italian.*

## L'idea

Sophon-Realtime si serve con vLLM-Omni. Ma lo stadio voce della variante **Flash** è un **Qwen3-TTS 1.7B affinato** (`talker/config.json`: `model_type: qwen3_tts`), con l'italiano tra le lingue: si carica con il pacchetto `qwen-tts` in PyTorch puro, e quindi anche con `faster-qwen3-tts`.

Le voci di Sophon sono vettori da 2048 valori (`voices/*.pt`): si passano come voce «solo x-vector». Altre se ne ricavano da 3–10 secondi di parlato.

## Misure

Mini PC con Ryzen AI Max (Strix Halo), Radeon 8060S, torch 2.13 + ROCm 7.15, `bfloat16`. Tre frasi in italiano, secondo giro. RTF = tempo di calcolo diviso durata dell'audio (sotto 1 = più veloce del tempo reale). Un fotogramma vale 80 ms di audio.

| Configurazione | Calcolo per fotogramma | RTF | Primo audio (blocchi da 4) |
|---|---|---|---|
| `qwen-tts`, torch 2.9.1 + ROCm 7.13 | – | 1,55 | – |
| `qwen-tts`, torch 2.13 + ROCm 7.15 | – | 0,79 | – |
| `faster-qwen3-tts`, cache 2048 | 58,7 ms | 0,77 | 0,37 s |
| cache del parlatore a 384 | 54,2 ms | 0,71 | 0,27 s |
| + `torch.compile` sul predittore | 42,6 ms | 0,56 | 0,22 s |
| + `torch.compile` sul parlatore | **37,7 ms** | **0,50** | **0,20 s** |
| come sopra, `max-autotune-no-cudagraphs` | 35,6 ms | 0,47 | 0,19 s |

Cosa abbiamo imparato:

- **La versione di torch/ROCm conta più di tutto il resto**: da sola dimezza i tempi.
- **I grafi di `faster-qwen3-tts` su ROCm si catturano, ma non accelerano**: qui il collo di bottiglia non è Python. Quello che la libreria dà davvero è lo streaming.
- **La cache statica predefinita (2048) è sprecata** per frasi singole: a ogni passo l'attenzione scorre tutta la cache. A 384 il parlatore passa da 27,6 a 22,6 ms.
- **`torch.compile` va applicato prima della cattura dei grafi**: predittore da 31 a 20 ms, parlatore da 22,6 a 17,7 ms.
- `float16` va come `bfloat16`; `float32` è lento il doppio.
- `max-autotune` guadagna il 5% e costa 77 secondi di avvio: non conviene.
- Memoria GPU: circa 5 GB.

Limite attuale: il parlatore legge quasi 3 GB di pesi a ogni fotogramma, quindi dipende dalla velocità della memoria. Per scendere ancora servirebbero pesi quantizzati (non provato).

## Uso

```bash
# 1. pesi: solo lo stadio voce e le voci
hf download adventists-ai/Sophon-Realtime-4B-Flash --include 'talker/*' 'voices/*' --local-dir flash-model

# 2. ambiente: torch per ROCm, poi
pip install qwen-tts faster-qwen3-tts soundfile "numpy<2.5"

# 3. una voce: un WAV di 3-10 secondi, oppure un vettore .pt
python voce_veloce.py --modello flash-model/talker --voce mia-voce.wav \
    --testo "Ciao, sono il tuo assistente. Come posso aiutarti?" --uscita prova.wav
```

```python
from voce_veloce import carica, voce_da_audio, parla

m = carica("flash-model/talker")                 # circa 15 secondi, una volta sola
voce = voce_da_audio(m, "mia-voce.wav")
for pezzo, fs in parla(m, voce, "Buongiorno! Oggi hai due appuntamenti."):
    ...                                           # float32 a 24 kHz, circa 0,3 s per pezzo
```

`misura.py` rifà la tabella sulla propria macchina.

## Trappole

- **Una frase alla volta.** Con testi lunghi il modello chiude prima e l'audio risulta troncato: `parla()` divide in frasi.
- **NumPy 2.5 rompe l'importazione di `qwen-tts`** (numba accetta al massimo la 2.4).
- `MIOPEN_FIND_MODE=FAST` evita la lunga ricerca iniziale di MIOpen.
- La cache a 384 basta per frasi fino a circa 20 secondi; oltre, alzare `--cache`.
- `voce_veloce.py` sostituisce due attributi interni di `faster-qwen3-tts` 0.3.2 (`predictor_graph.pred_model`, `talker_graph.model`): con altre versioni può servire adattarlo.
- La qualità è stata controllata facendo trascrivere l'audio a Whisper (testo riconosciuto parola per parola), non con un test d'ascolto formale.

## Cosa non c'è

- I pesi e le voci: si scaricano da Hugging Face, con la loro licenza.
- Il «cervello» di Sophon: qui c'è solo la voce. Si collega a un qualsiasi modello di testo.

## Licenza

MIT per il codice di questo repository.
