# Misura le varianti: cache corta del parlatore e torch.compile di predittore e parlatore.
# MODELLO=cartella/talker VOCE=voce.pt python misura.py [varianti...]
import os, sys, gc, time, traceback
os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')
import torch, numpy as np, soundfile as sf
T = os.environ.get('MODELLO', 'flash-model/talker'); VOCE = os.environ.get('VOCE', 'voci/voce.pt'); OUT = 'misure'; os.makedirs(OUT, exist_ok=True)
def P(*a): print(*a, flush=True)
from faster_qwen3_tts import FasterQwen3TTS
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
vec = torch.load(VOCE, map_location='cpu', weights_only=True).float()
FRASI = ["Oggi c'è il sole e hai due appuntamenti: alle dieci la riunione con il cliente, e alle quindici il controllo dei backup.",
         "Il server è acceso, ma il disco è quasi pieno.",
         "Stamattina sono arrivate tre richieste di assistenza: due sono già chiuse, la terza aspetta una risposta."]
GEN = dict(temperature=0.7, top_k=50, repetition_penalty=1.05, max_new_tokens=600)
sync = torch.cuda.synchronize
def ms(f, n=40):
    for _ in range(5): f()
    sync(); t = time.time()
    for _ in range(n): f()
    sync(); return (time.time() - t) / n * 1000
VAR = {'cache2048': dict(seq=2048), 'cache384': dict(seq=384), 'cache384+pred': dict(seq=384, pred=True), 'cache384+pred+parl': dict(seq=384, pred=True, parl=True)}
for nome in (sys.argv[1:] or list(VAR)):
    v = VAR[nome]; P('==== variante', nome)
    try:
        dt = torch.bfloat16
        m = FasterQwen3TTS.from_pretrained(T, device='cuda', dtype=dt, max_seq_len=v['seq'])
        if v.get('pred'): m.predictor_graph.pred_model = torch.compile(m.predictor_graph.pred_model)
        if v.get('parl'): m.talker_graph.model = torch.compile(m.talker_graph.model)
        t = time.time(); m.warmup(prefill_len=100); sync(); P('  preparazione %.1f s' % (time.time() - t))
        item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec.to('cuda', dt), x_vector_only_mode=True, icl_mode=False, ref_text=None)
        a = ms(m.predictor_graph.graph.replay); b = ms(m.talker_graph.graph.replay)
        P('  per fotogramma (80 ms di audio): predittore %.1f ms, parlatore %.1f ms, somma %.1f ms' % (a, b, a + b))
        ta = tc = 0; pezzi = []
        for giro in (1, 2):
            for f in FRASI:
                sync(); t = time.time()
                w, fs = m.generate_voice_clone(text=f, language='Italian', voice_clone_prompt=[item], xvec_only=True, non_streaming_mode=False, **GEN)
                sync(); d = time.time() - t; w0 = np.asarray(w[0], dtype=np.float32)
                if giro == 2: ta += len(w0) / fs; tc += d; pezzi += [w0, np.zeros(int(fs * 0.28), np.float32)]
        P('  frase intera: audio %.1f s  calcolo %.1f s  RTF %.2f' % (ta, tc, tc / ta))
        sf.write('%s/%s.wav' % (OUT, nome.replace('+', '_')), np.concatenate(pezzi), fs)
        t = time.time(); primo = None; tot = 0
        for chunk, fs, info in m.generate_voice_clone_streaming(text=FRASI[0], language='Italian', voice_clone_prompt=[item], xvec_only=True, non_streaming_mode=False, chunk_size=4, **GEN):
            if primo is None: primo = time.time() - t
            tot += len(chunk)
        d = time.time() - t; P('  streaming blocco 4: primo audio %.2f s, RTF %.2f' % (primo, d / (tot / fs)))
        del m, item; gc.collect(); torch.cuda.empty_cache()
    except Exception: P(traceback.format_exc()[-1500:])
P('FINE')
