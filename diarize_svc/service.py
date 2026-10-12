"""diarize 服务：wav + oracle_num -> 切分段落 JSON（阿里 campplus pipeline 常驻）。

POST /diarize  multipart: file=wav, oracle_num=可选
  -> {"segs": [[start_s, end_s, spk_id], ...], "used_oracle": N, "audio_sec": T}
人数策略：不传 oracle_num 时按 2..6 各跑一遍，取簇内平均相似度最高者（宁多勿少）。
"""
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile

from modelscope.pipelines import pipeline

# ============ 路径配置（2026-09-30：全部走环境变量，默认值=当前生产路径） ============
# 全新部署只需保证目录结构一致或显式设环境变量；不改任何代码。
_MS_MODELS = os.environ.get(
    "MS_MODELS_DIR", os.path.expanduser("~/.cache/modelscope/models"))
_ASR_ROOT = os.environ.get(
    "ASR_ROOT", os.path.expanduser("~/jvkit/asr"))

app = FastAPI(title="diarize-service")
sd = pipeline(task="speaker-diarization", model=f"{_MS_MODELS}/iic--speech_campplus_speaker-diarization_common/snapshots/master")
SV = None  # 懒加载 cam++ sv（评分用）


def _sv():
    global SV
    if SV is None:
        from funasr import AutoModel
        SV = AutoModel(model=f"{_MS_MODELS}/iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master", device="cpu", disable_update=True)
    return SV


def _cluster_score(audio: np.ndarray, sr: int, segs: list) -> float:
    """簇内紧度：每段 embedding 与其簇中心的平均余弦相似度（越高越像"分对了"）。"""
    sv = _sv()
    embs: dict[int, list] = {}
    for s0, s1, spk in segs:
        clip = audio[int(s0*sr):int(s1*sr)]
        if len(clip) < sr * 0.5:
            continue
        e = sv.generate(input=clip, fs=sr, extract_embedding=True)[0]["spk_embedding"]
        e = np.asarray(e, dtype=np.float32).ravel()
        e /= np.linalg.norm(e) + 1e-9
        embs.setdefault(spk, []).append(e)
    scores = []
    for spk, es in embs.items():
        if len(es) < 2:
            continue
        c = np.mean(es, axis=0)
        c /= np.linalg.norm(c) + 1e-9
        scores.extend(float(np.dot(e, c)) for e in es)
    return float(np.mean(scores)) if scores else -1.0


def _run(audio_path: str, audio: np.ndarray, sr: int, oracle: int | None) -> list:
    sd.model.model_config["merge_thr"] = 0.55
    kw = {"oracle_num": oracle} if oracle else {}
    try:
        segs = sd(audio_path, **kw)["text"]
    except AssertionError as e:
        # [patch] 阿里 pipeline 要求 VAD 后有效语音 >5s，过短直接 assert 崩；
        # 降级为空段（调用方按无段落处理），不再 500
        print(f"[patch] diarize skipped: {e}", flush=True)
        return []
    return [[float(a), float(b), int(s)] for a, b, s in segs]


@app.post("/diarize")
async def diarize(file: UploadFile = File(...), oracle_num: int | None = Form(default=None)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        audio, sr = sf.read(io.BytesIO(open(tmp, "rb").read()))
        if audio.ndim > 1:
            audio = audio[:, 0]
        dur = len(audio) / sr

        if oracle_num:
            segs = _run(tmp, audio, sr, oracle_num)
            used = oracle_num
        else:
            # 自动：2..6 各跑一遍取簇内紧度最高（音频 <20s 只跑 2..3，省时）
            cands = range(2, 7) if dur > 60 else range(2, 4)
            best, best_score, used = None, -2.0, None
            for n in cands:
                segs_n = _run(tmp, audio, sr, n)
                sc = _cluster_score(audio, sr, segs_n)
                print(f"  oracle={n} score={sc:.4f} spk={dict(Counter(s[2] for s in segs_n))}", flush=True)
                if sc > best_score:
                    best, best_score, used = segs_n, sc, n
            segs = best
        return {"segs": segs, "used_oracle": used, "audio_sec": round(dur, 1)}
    finally:
        Path(tmp).unlink(missing_ok=True)


# ============ /meeting_chunk：非实时一体化管线（VAD+区分人+逐子段ASR+热词） ============
_ASR = None


def _asr():
    global _ASR
    if _ASR is None:
        from funasr import AutoModel
        # SeACo Paraformer：paraformer-large 的热词增强版（generate 支持 hotwords）
        _ASR = AutoModel(
            model=f"{_MS_MODELS}/iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/snapshots/master",
            device="cpu", disable_pbar=True, disable_log=True, disable_update=True,
        )
    return _ASR


@app.post("/meeting_chunk")
async def meeting_chunk(file: UploadFile = File(...), name_threshold: float = Form(default=0.30),
                        max_end_silence_ms: int = Form(default=250), hotwords: str = Form(default=""),
                        anon_clusters: str = Form(default="[]"), anon_next: int = Form(default=1),
                        anon_threshold: float = Form(default=0.48), split_gap: float = Form(default=0.12), confirm_count: int = Form(default=2), tail_attach: bool = Form(default=True), tail_min_sim: float = Form(default=0.25), display_min_sec: float = Form(default=3.0), final: bool = Form(default=False), hw_weight: float = Form(default=1.0), punc: bool = Form(default=False), itn: bool = Form(default=False)):
    """wav -> [{start,end,speaker,text}]，内部完整走好管线（含归一化）。

    v2：ASR 逐子段执行（v1 整句 ASR + 子段复制整句文字，长句换人时整句文字
    会重复出现在两个子段上）；hotwords 空格分隔（FunASR 按空白切分）。
    """
    import tempfile
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        audio, sr = sf.read(tmp)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        # RMS 归一化（远场轻录音 VAD 漏检修复，已验证）
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-6 and abs(rms - 0.1) / 0.1 > 0.3:
            audio = np.clip(audio * (0.1 / rms), -1.0, 1.0)
        # VAD
        r = _vad().generate(input=audio.copy(), fs=sr, max_end_silence_time=max_end_silence_ms)
        vad = r[0]["value"] if isinstance(r, list) else r["value"]
        sents = [[a / 1000, b / 1000] for a, b in vad]
        if not sents:
            return {"segs": []}
        # 区分人（diarize_sent 逻辑内联：复用其滑窗子段切分）
        r2 = await diarize_sent(
            file=UploadFile(file=io.BytesIO(data), filename="chunk.wav"),
            sents=json.dumps(sents), name_threshold=name_threshold,
            enroll_dir="", sub_min=2.5, sub_win=0.6, sub_shift=0.3,
            sub_min_dur=0.5, merge_margin=0.10, change_margin=0.04,
            anon_clusters=anon_clusters, anon_next=anon_next,
            anon_threshold=anon_threshold, split_gap=split_gap,
            confirm_count=confirm_count, tail_attach=tail_attach,
            tail_min_sim=tail_min_sim, display_min_sec=display_min_sec,
            final=final,
        )
        # 逐子段 ASR：文字与子段时间严格对应（v1 的整句复制会有重复文字）
        asr = _asr()
        segs = []
        for seg in r2["segs"]:
            a, b = float(seg[0]), float(seg[1])
            clip = audio[int(a * sr):int(b * sr)]
            if len(clip) < sr * 0.1:
                txt = ""
            elif hotwords:
                rr = asr.generate(input=clip, fs=sr, hotword=hotwords, seaco_weight=hw_weight)  # hotword 单数; seaco_weight=偏置强度λ
                txt = (rr[0]["text"] if isinstance(rr, list) else rr["text"]).strip()
            else:
                rr = asr.generate(input=clip, fs=sr)
                txt = (rr[0]["text"] if isinstance(rr, list) else rr["text"]).strip()
            if txt and punc:
                try:
                    rp = _punc().generate(input=txt.replace(" ", ""))
                    txt = (rp[0]["text"] if isinstance(rp, list) else rp["text"]).strip()
                except Exception:
                    pass
            if txt and itn:
                try:
                    txt = _itn().normalize(txt)
                except Exception:
                    pass
            segs.append({"start": round(a, 3), "end": round(b, 3),
                         "speaker": seg[3], "text": txt})
        return {"segs": segs, "anon": r2.get("anon", [])}
    finally:
        Path(tmp).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {"ok": True}


# ============ /vad：fsmn 切句（基准测试取句子边界用） ============
_VAD = None


def _vad():
    global _VAD
    if _VAD is None:
        from funasr import AutoModel
        _VAD = AutoModel(
            model=f"{_MS_MODELS}/damo--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/v2.0.2",
            device="cpu", disable_pbar=True, disable_log=True, disable_update=True,
        )
    return _VAD


@app.post("/vad")
async def vad_endpoint(file: UploadFile = File(...), max_end_silence_ms: int = Form(default=250)):
    """wav -> VAD 句子边界 [[start_s, end_s], ...]。

    默认 250ms 静音切句（2026-09-18 定档）：800ms 会把开场自我介绍这类人间停顿
    仅 ~300ms 的轮流发言揉成多人混合段（DER 48.8% vs 250 的 12.3%，bb3cd75f14e8 实测），
    归属原子优先，碎段展示由本地显示层合并解决。
    """
    import tempfile
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        audio, sr = sf.read(tmp)
        if audio.ndim > 1:
            audio = audio[:, 0]
        r = _vad().generate(input=audio.astype(np.float32).copy(), fs=sr,
                            max_end_silence_time=max_end_silence_ms)
        segs = r[0]["value"] if isinstance(r, list) else r["value"]
        # fsmn vad 输出毫秒
        return {"sents": [[round(a / 1000, 3), round(b / 1000, 3)] for a, b in segs]}
    finally:
        Path(tmp).unlink(missing_ok=True)


# ============ /embed：ERes2NetV2 声纹 embedding（会末逐段标身份用） ============
_ERE = None


def _ere():
    global _ERE
    if _ERE is None:
        from funasr import AutoModel
        _ERE = AutoModel(
            model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
            init_param=f"{_MS_MODELS}/iic--speech_eres2netv2_sv_zh-cn_16k-common/snapshots/master/pretrained_eres2netv2.ckpt",
            device="cpu", disable_pbar=True, disable_log=True, disable_update=True,
        )
    return _ERE


@app.post("/embed")
async def embed(file: UploadFile = File(...), sr: int = Form(default=16000)):
    """wav 片段 -> ERes2NetV2 embedding（192 维，L2 归一化）。"""
    data = await file.read()
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    e = _ere().generate(input=audio, fs=sr, extract_embedding=True)[0]["spk_embedding"]
    e = np.asarray(e, dtype=np.float32).ravel()
    e = e / (np.linalg.norm(e) + 1e-9)
    return {"embedding": e.tolist()}


# ============ 声纹库接入 + 带标注切分 ============
import json as _json

_SPEAKER_DB = os.environ.get(
    "SPEAKER_DB", f"{_ASR_ROOT}/funasr-src/runtime/python/websocket/speaker_db.json")
_SPEAKER_ENROLL_DIR = os.environ.get(
    "SPEAKER_ENROLL_DIR", f"{_ASR_ROOT}/funasr-src/runtime/python/websocket/speaker_enrolls")

_DB_MULTI_CACHE: dict[str, dict] = {}
_DB_MULTI_MTIME: dict[str, dict] = {}


def _load_db_multi(enroll_dir: str | None = None) -> dict[str, np.ndarray]:
    """多段注册质心库：按人名前缀分组 speaker_enrolls/{name}_*.pcm，
    逐段 ERes2NetV2 embedding 后取平均质心再归一化。

    背景：库内单段向量 vs 会议簇质心相似度不稳（注册态 vs 会议态声学域差异，
    admin 单段只有 0.484，margin 0.004 被判匿名）；三段质心可到 0.681。

    增量重建（2026-10-09）：按"每人文件名+mtime 签名"对比，只有文件集变化的人
    才重提 embedding；其他人质心直接复用。此前是全库 mtime 一变动就全量重建
    （13 人 × ~5 段 CPU 提向量好几分钟），替换/删除一个人就把服务卡死——
    用户实测"算到一半声纹服务不可用"的根因。
    """
    import glob as _glob
    d = Path(enroll_dir or _SPEAKER_ENROLL_DIR)
    if not d.is_dir():
        return {}
    cache_key = str(d)
    by_name: dict[str, list[Path]] = {}
    for pth in sorted(d.iterdir()):
        if pth.name.endswith(".pcm"):
            by_name.setdefault(_name_of_stem(pth.stem), []).append(pth)

    cache = _DB_MULTI_CACHE.setdefault(cache_key, {})
    sigs = _DB_MULTI_MTIME.setdefault(cache_key, {})
    ere = None  # 懒加载：全命中时不碰模型
    refs: dict[str, np.ndarray] = {}
    for name, paths in by_name.items():
        sig = [(p.name, p.stat().st_mtime, p.stat().st_size) for p in paths]
        if sigs.get(name) == sig and name in cache:
            refs[name] = cache[name]
            continue
        # ---- 这人变了（新增/替换/删除过段）：只重算他 ----
        tot_sec = sum(p.stat().st_size for p in paths) / 32000.0
        if len(paths) < 3 and tot_sec < 8.0:
            print(f"  [db_multi] {name}: 待激活（{len(paths)}段/{tot_sec:.1f}s），不参与认人", flush=True)
            sigs[name] = sig
            continue
        if ere is None:
            ere = _ere()
        vecs = []
        for p in paths:
            try:
                pcm = np.frombuffer(p.read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
                if len(pcm) < 16000 * 1.5:  # <1.5s 跳过
                    continue
                _r = float(np.sqrt(np.mean(pcm ** 2)))
                if _r > 1e-6:
                    pcm = np.clip(pcm * (0.1 / _r), -1.0, 1.0)
                e = ere.generate(input=pcm, fs=16000, extract_embedding=True)[0]["spk_embedding"]
                e = np.asarray(e, dtype=np.float32).ravel()
                n = np.linalg.norm(e)
                if n < 1e-9:
                    continue
                vecs.append(e / n)
            except Exception as ex:
                print(f"  [db_multi] skip {p.name}: {ex}", flush=True)
        if vecs:
            c = np.mean(vecs, axis=0)
            c /= np.linalg.norm(c) + 1e-9
            refs[name] = c
            print(f"  [db_multi] {name}: {len(vecs)} 段质心（重建）", flush=True)
        sigs[name] = sig
    # 已从盘上消失的人：清签名防泄漏
    for gone in list(sigs.keys()):
        if gone not in by_name:
            sigs.pop(gone, None)

    _DB_MULTI_CACHE[cache_key] = refs
    return refs


def _load_db() -> dict[str, list[float]]:
    try:
        return _json.load(open(_SPEAKER_DB))
    except Exception:
        return {}


def _label_segs(audio: np.ndarray, sr: int, segs: list, name_threshold: float = 0.40) -> tuple[list, list]:
    """对已切分的 segs 逐段做 ERes2NetV2 声纹投票标身份，返回 (out, registered)。"""
    if not segs:
        return [], []

    ere = _ere()
    seg_embs = []
    for s0, s1, spk in segs:
        clip = audio[int(s0*sr):int(s1*sr)]
        if len(clip) < sr * 0.5:
            seg_embs.append(None)
            continue
        e = ere.generate(input=clip, fs=sr, extract_embedding=True)[0]["spk_embedding"]
        e = np.asarray(e, dtype=np.float32).ravel()
        e /= np.linalg.norm(e) + 1e-9
        seg_embs.append(e)

    db = _load_db()
    refs = {}
    for name, vec in db.items():
        v = np.asarray(vec, dtype=np.float32).ravel()
        v /= np.linalg.norm(v) + 1e-9
        refs[name] = v

    # 投票：段->最近注册名；簇->多数票
    # 阈值须与实时 FunASR 的 --speaker_match_threshold(0.40) 对齐：
    # 曾用 0.5 导致「实时答对(admin 0.502) -> 重标注判匿名」的回退
    NAME_MATCH_THRESHOLD = name_threshold
    seg_votes = []
    for e in seg_embs:
        if e is None or not refs:
            seg_votes.append((None, 0.0))
            continue
        scores = {n: float(np.dot(e, v)) for n, v in refs.items()}
        best = max(scores, key=scores.get)
        if scores[best] < NAME_MATCH_THRESHOLD:
            seg_votes.append((None, scores[best]))
            print(f"  [vote] below threshold {scores[best]:.3f} < {NAME_MATCH_THRESHOLD} -> anon", flush=True)
        else:
            seg_votes.append((best, scores[best]))

    from collections import Counter, defaultdict
    cluster_votes: dict[int, Counter] = defaultdict(Counter)
    for (s0, s1, spk), (name, _) in zip(segs, seg_votes):
        if name:
            cluster_votes[spk][name] += 1

    # 簇 -> 名字；同名校验：若两个簇多数票同名，保留票数多者，其余退匿名
    cluster_name: dict[int, str] = {}
    name_taken: dict[str, int] = {}
    for spk, c in sorted(cluster_votes.items(), key=lambda x: -sum(x[1].values())):
        if not c:
            continue
        top = c.most_common(1)[0]
        if top[0] in name_taken and name_taken[top[0]] >= sum(c.values()):
            continue  # 名字已被更强簇占用
        cluster_name[spk] = top[0]
        name_taken[top[0]] = max(name_taken.get(top[0], 0), sum(c.values()))

    # [patch] 匿名编号按簇（spk）而非按段：同簇同号，否则 2 人被拆成 N 个说话人
    anon = 0
    cluster_anon_name: dict[int, str] = {}
    out = []
    for s0, s1, spk in segs:
        name = cluster_name.get(spk)
        if not name:
            if spk not in cluster_anon_name:
                anon += 1
                cluster_anon_name[spk] = f"说话人{anon}"
            name = cluster_anon_name[spk]
        out.append([s0, s1, spk, name])
    return out, sorted(refs.keys())


def _cluster_label(audio: np.ndarray, sr: int, segs: list, refs: dict, n_target: int) -> tuple[list, list]:
    """过采样切分后，用段 embedding 层次聚类合并到 n_target 簇，簇质心 vs 注册库 argmax 命名。

    解决 oracle 人数切分粒度问题：oracle 给少了（如 2）阿里 pipeline 会把整段
    连续对话并成一个人；这里用更细的切分 + embedding 聚类归并回真实人数，
    再对簇质心做「相对判定」（argmax），避免绝对阈值对短句 embedding 脆弱。
    refs: {name: l2norm embedding}；n_target 为真实人数（切分簇数会多于它）。
    返回 (out, registered)，out=[[s0,s1,spk,name],...]。
    段 embedding 数组与 segs 严格等长对齐（碎段也占位，避免索引错位）。
    """
    if not segs:
        return [], []
    ere = _ere()

    # 逐段 embedding，与 segs 严格等长对齐；碎段(<0.5s)存 None
    embs: list[np.ndarray | None] = []
    for s0, s1, spk in segs:
        clip = audio[int(s0*sr):int(s1*sr)]
        if len(clip) < sr * 0.5:
            embs.append(None)
            continue
        e = ere.generate(input=clip, fs=sr, extract_embedding=True)[0]["spk_embedding"]
        e = np.asarray(e, dtype=np.float32).ravel()
        e /= np.linalg.norm(e) + 1e-9
        embs.append(e)

    valid_idx = [i for i, e in enumerate(embs) if e is not None]
    if not valid_idx:
        # 全为碎段：退化为原始 spk 命名
        return _label_segs(audio, sr, segs, name_threshold=0.40)

    X = np.array([embs[i] for i in valid_idx])  # 只取有效段
    n_seg = len(valid_idx)

    # 对有效段做层次聚类到 n_target 簇（若无有效段则退化）
    labels_valid: np.ndarray | None = None
    if n_seg > n_target:
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(X, method="average", metric="cosine")
        best_t, best_lab, best_err = None, None, 1e9
        for t in np.linspace(0.2, 0.9, 71):
            lab = fcluster(Z, t=t, criterion="distance")
            k = len(set(lab))
            if k < 2:
                continue
            err = abs(k - n_target)
            if err < best_err:
                best_err, best_t, best_lab = err, t, lab
        labels_valid = best_lab if best_lab is not None else np.arange(n_seg)
    else:
        labels_valid = np.arange(n_seg)

    # 簇质心 vs 注册库：相对判定（argmax），不设绝对阈值
    # 但保留「top1 与 top2 差距过小则判不确定」保护
    cluster_cent: dict[int, np.ndarray] = {}
    for c in sorted(set(labels_valid.tolist())):
        members = X[labels_valid == c]
        cent = members.mean(axis=0)
        cent /= np.linalg.norm(cent) + 1e-9
        cluster_cent[c] = cent

    # 一对一分配（匈牙利算法）：不同簇不可能是同一人，逐簇独立 argmax 会把
    # 第二个簇也拉向 admin（实测 c2 对 admin 0.587 vs 对蒋泽鹏 0.591，margin
    # 0.004 被判匿名）。分配后 c1 占走 admin，c2 只能在剩余注册名里选。
    cluster_name: dict[int, str] = {}
    anon = 0
    MATCH_FLOOR = 0.30  # 分配后相似度过低视为陌生人（远低于真实注册匹配 0.5+）
    if refs and cluster_cent:
        cs = sorted(cluster_cent)
        names = sorted(refs)
        M = np.array([[float(np.dot(cluster_cent[c], refs[n])) for n in names] for c in cs])
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-M)
        assign = {cs[r]: names[c] for r, c in zip(ri, ci)}
        for c in cs:
            n = assign.get(c)
            s = M[cs.index(c), names.index(n)] if n else -1.0
            if n is None or s < MATCH_FLOOR:
                anon += 1
                cluster_name[c] = f"说话人{anon}"
                print(f"  [cluster] c={c} 分配 {n} 相似度 {s:.3f} < {MATCH_FLOOR} -> 匿名", flush=True)
            else:
                cluster_name[c] = n
                print(f"  [cluster] c={c} -> {n} ({s:.3f})", flush=True)
    else:
        for c in sorted(cluster_cent):
            anon += 1
            cluster_name[c] = f"说话人{anon}"

    # 输出：每段映射到其簇名；碎段用最近有效段归属
    seg_label: dict[int, str] = {}
    for i in valid_idx:
        lab = labels_valid[valid_idx.index(i)]
        seg_label[i] = cluster_name.get(int(lab), f"说话人{spk+1}")

    out = []
    for i, (s0, s1, spk) in enumerate(segs):
        name = seg_label.get(i)
        if not name:
            # 碎段：找最近的有效段（按索引），否则按原始 spk
            near = min(valid_idx, key=lambda x: abs(x - i)) if valid_idx else None
            name = seg_label.get(near) if near is not None else f"说话人{spk+1}"
        out.append([s0, s1, spk, name])
    return out, sorted(refs.keys())


@app.post("/diarize_labeled")
async def diarize_labeled(file: UploadFile = File(...), oracle_num: int | None = Form(default=None), auto_tune: bool = Form(default=False), name_threshold: float = Form(default=0.50)):
    """切分 + 声纹标身份一步完成：每段 ERes2NetV2 embedding 对声纹库投票。

    返回 segs: [[start_s, end_s, spk_id, name], ...]
    spk_id 簇若多数票命中注册名则合并同名簇；未命中标"说话人N"。

    oracle_num>0 时按给定人数切分；auto_tune=true 且未给人数时，
    按 2..6（>60s）或 2..3（≤60s）各跑一遍，取簇内紧度最高者（与 /diarize 一致）。
    name_threshold 为逐段声纹匹配阈值（与实时 FunASR --speaker_match_threshold 同义）。
    注意：实测 0.40 会把大量 0.34~0.38 的无关段误吸成注册名（全 admin），
    0.50 则偏严（0.502 的真匹配被拒）——逐段绝对阈值天生脆弱，
    上层须配「匿名标签不覆盖实时注册名」的回写保护，不能单靠调这里。
    """
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        audio, sr = sf.read(tmp)
        if audio.ndim > 1:
            audio = audio[:, 0]
        dur = len(audio) / sr

        used = oracle_num
        if oracle_num:
            # [v2] 过采样切分 + embedding 聚类合并：oracle 直接给小时（如 2），
            # 阿里 pipeline 会把连续对话整段并成一个人（实测 0-29s 不切）。
            # 切分用 oracle_num+2（更细），再按段声纹聚类合并回 oracle_num 人。
            ov = min(oracle_num + 2, 8)
            fine_segs = _run(tmp, audio, sr, ov)
            print(f"[labeled v2] oracle={oracle_num} 过采样={ov} 切出 {len(fine_segs)} 段", flush=True)
            refs = _load_db_multi()  # 多段注册质心（稳定），不再用库内单段向量
            out, registered = _cluster_label(audio, sr, fine_segs, refs, n_target=oracle_num)
            return {"segs": out, "used_oracle": used, "audio_sec": round(dur, 1),
                    "registered": registered}
        elif auto_tune:
            # 自动选优：候选人数各跑一遍，取簇内紧度最高（宁少勿拆出假说话人）
            cands = range(2, 7) if dur > 60 else range(2, 4)
            best, best_score, used = None, -2.0, None
            for n in cands:
                segs_n = _run(tmp, audio, sr, n)
                sc = _cluster_score(audio, sr, segs_n)
                print(f"  [labeled] oracle={n} score={sc:.4f} spk={dict(Counter(s[2] for s in segs_n))}", flush=True)
                if sc > best_score:
                    best, best_score, used = segs_n, sc, n
            segs = best or []
        else:
            segs = _run(tmp, audio, sr, None)

        out, registered = _label_segs(audio, sr, segs, name_threshold=name_threshold)
        return {"segs": out, "used_oracle": used, "audio_sec": round(dur, 1),
                "registered": registered}
    finally:
        Path(tmp).unlink(missing_ok=True)


# ============ /diarize_sent：按调用方给定句子边界逐句声纹命名 ============
# 背景：阿里 pipeline 切分非确定（同音频 3 次跑出 13/8/8 段），且 oracle 直切
# 会把连续对话并成一大段。实时版 FunASR 2pass 的句子边界本身是准的，
# 因此会后版不再重新切分，直接拿实时句子边界逐句 ER argmax 命名——
# 确定性、可复现、无文字分配错位问题。


@app.post("/diarize_sent")
async def diarize_sent(file: UploadFile = File(...), sents: str = Form(default="[]"), name_threshold: float = Form(default=0.30), enroll_dir: str = Form(default=""), sub_min: float = Form(default=2.5), sub_win: float = Form(default=0.6), sub_shift: float = Form(default=0.3), sub_min_dur: float = Form(default=0.5), merge_margin: float = Form(default=0.10), change_margin: float = Form(default=0.04), anon_clusters: str = Form(default="[]"), anon_next: int = Form(default=1), anon_threshold: float = Form(default=0.48), split_gap: float = Form(default=0.12), confirm_count: int = Form(default=2), tail_attach: bool = Form(default=True), tail_min_sim: float = Form(default=0.25), display_min_sec: float = Form(default=3.0), final: bool = Form(default=False)):
    """wav + 句子边界 -> 每句的声纹命名（注册名 / 陌生人N）。

    逐句/滑窗 ERes2NetV2 embedding 对注册声纹库（多段质心）argmax：
    - top1 >= name_threshold 且领先 >= change_margin -> 注册名
    - 否则匿名，进入匿名聚类（2026-09-15 方案A，替代旧的"继承相邻句"）：
      1) 先对本场既有匿名簇（anon_clusters，时长加权更新质心）
      2) 未匹配的批内 gap 聚类（地板 anon_threshold + 相邻落差 >0.12 切开），
         新簇编号 陌生人{anon_next+i}
    返回 segs + anon（更新/新增的簇，调用方持久化后随下批回传）。"""
    import numpy as np
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        audio, sr = sf.read(tmp)
        if audio.ndim > 1:
            audio = audio[:, 0]

        sent_list = json.loads(sents) if sents else []
        if not sent_list:
            return {"segs": [], "registered": sorted(_load_db_multi(enroll_dir or None).keys()), "anon": []}

        ere = _ere()
        refs = _load_db_multi(enroll_dir or None)
        names = sorted(refs)
        anon_existing = json.loads(anon_clusters or "[]")

        def _embed(t0: float, t1: float):
            clip = audio[int(max(0.0, t0) * sr):int(t1 * sr)]
            if len(clip) < sr * 0.25:
                return None
            e = ere.generate(input=clip, fs=sr, extract_embedding=True)[0]["spk_embedding"]
            e = np.asarray(e, dtype=np.float32).ravel()
            e = e / (np.linalg.norm(e) + 1e-9)
            if not names:  # 空库（盲测/未注册任何声纹）：仍出 embedding 供匿名聚类
                return None, 0.0, 0.0, e
            sims = {n: float(np.dot(e, refs[n])) for n in names}
            best = max(sims, key=sims.get)
            runner = max((v for k, v in sims.items() if k != best), default=0.0)
            return best, sims[best], sims[best] - runner, e

        def _name_of(best: str, sim: float, margin: float) -> str | None:
            return best if (sim >= name_threshold and margin >= change_margin) else None

        # 子句滑窗切分：返回 [(a, b, name|None, emb|None)]，覆盖 [a,b]
        # 匿名窗不做邻居填充（保留匿名身份，交后续聚类）——防"继承相邻句"染色
        def _subsegs(a: float, b: float):
            wins = []  # [t0, t1, name|None, emb]
            t = a
            while t < b - 0.05:
                t1 = min(t + sub_win, b)
                if t1 - t < 0.3:
                    break
                r = _embed(t, t1)
                if r is None:
                    break
                best, sim, margin, e = r
                wins.append([t, t1, _name_of(best, sim, margin), e])
                t += sub_shift
            if not wins:
                return [(a, b, None, None)]
            wins[-1][1] = b  # 尾窗补齐
            # 换人抖动回退：仅当孤立命名窗的前邻居也是命名窗才回退；
            # 邻居是匿名（None）时保留命名——可能是真换人，交聚类判断（防染色的反向）
            for i in range(1, len(wins) - 1):
                if (wins[i][2] is not None and wins[i][2] != wins[i - 1][2]
                        and wins[i][2] != wins[i + 1][2] and wins[i - 1][2] is not None):
                    wins[i][2] = wins[i - 1][2]
            # 连续同标签合并（None 与 None 也合并成匿名段，embedding 累积）
            subs = []  # [a, b, label, embs(list)]
            for w in wins:
                if subs and subs[-1][2] == w[2]:
                    subs[-1][1] = w[1]
                    subs[-1][3].append(w[3])
                else:
                    subs.append([w[0], w[1], w[2], [w[3]]])
            # 碎段修剪：仅处理命名段（匿名段绝不并入命名邻居——染色的旧通道彻底关掉）
            guard = 0
            while len(subs) > 1 and guard < 100:
                guard += 1
                idx = next((i for i, x in enumerate(subs)
                            if x[2] is not None and x[1] - x[0] < sub_min_dur), None)
                if idx is None:
                    break
                if idx == 0:
                    tgt = 1
                elif idx == len(subs) - 1:
                    tgt = idx - 1
                else:
                    prev_d = subs[idx][0] - subs[idx - 1][0]
                    next_d = subs[idx + 1][1] - subs[idx + 1][0]
                    tgt = idx - 1 if prev_d >= next_d else idx + 1
                if tgt < idx:
                    subs[tgt][1] = subs[idx][1]
                    subs[tgt][3].extend(subs[idx][3])
                else:
                    subs[tgt][0] = subs[idx][0]
                    subs[tgt][3].extend(subs[idx][3])
                del subs[idx]
            # 边界 = 换人两侧窗重叠区中点；段 embedding = 窗均值（L2 归一）
            out = []
            for i, x in enumerate(subs):
                x0 = a if i == 0 else (subs[i - 1][1] + x[0]) / 2
                x1 = b if i == len(subs) - 1 else (x[1] + subs[i + 1][0]) / 2
                es = [e for e in x[3] if e is not None]
                if es:
                    emb = np.mean(es, axis=0)
                    emb = emb / (np.linalg.norm(emb) + 1e-9)
                else:
                    emb = None
                out.append((x0, x1, x[2], emb))
            return out

        # 逐句：短句/高置信整句 argmax；长句滑窗切分
        out_subs: list[list[tuple]] = []
        sent_emb: dict[int, object] = {}
        for a, b in sent_list:
            a = max(0.0, float(a))
            b = float(b)
            r = _embed(a, b)  # 空库也计算 embedding（匿名聚类的输入）
            if r is None:
                out_subs.append([(a, b, None, None)])
                continue
            merged_best, merged_sim, merged_margin, e = r
            sent_emb[len(out_subs)] = e
            if (merged_sim >= name_threshold and merged_margin >= merge_margin) \
                    or b - a < max(sub_min, 0.8):
                name = merged_best if (merged_sim >= name_threshold
                                       and merged_margin >= change_margin) else None
                out_subs.append([(a, b, name, e if name is None else None)])
            else:
                out_subs.append(_subsegs(a, b))

        # ---------- 匿名聚类 ----------
        anon_units = []  # (sent_idx, a, b, emb)
        for si, subs in enumerate(out_subs):
            for a, b, name, emb in subs:
                if name is None:
                    anon_units.append((si, a, b, emb if emb is not None else sent_emb.get(si)))
        labels: dict[tuple, str] = {}   # (sent_idx, a, b) -> 陌生人N / 人名 / None(候选未定)
        nxt = anon_next

        def _log(msg: str):
            print(f"  [anon] {msg}", flush=True)

        def _wmean(items):
            """时长加权质心（items: [(key, emb, dur)]）。"""
            tw = sum(d for _, _, d in items) or 1e-9
            m = sum(e * d for _, e, d in items) / tw
            return m / (np.linalg.norm(m) + 1e-9)

        def _robust_centroid(memb):
            """稳健定形：成员对'其余成员的质心'相似度 < 入簇地板 即踢出（最多2轮）。
            返回 (centroid, secs, kept, expelled)。防单段/错配段把簇心拖歪（雪崩）。"""
            items = list(memb)
            expelled = []
            for _ in range(2):
                if len(items) < 3:
                    break
                worst_i, worst_s = -1, 9.0
                for i, (key, e0, d0) in enumerate(items):
                    others = [x for j, x in enumerate(items) if j != i]
                    s = float(np.dot(e0, _wmean(others)))
                    if s < worst_s:
                        worst_s, worst_i = s, i
                if worst_s < anon_threshold:
                    kicked = items.pop(worst_i)
                    expelled.append(kicked)
                    _log(f"质心剔除 段 kicked_sim={worst_s:.3f} < {anon_threshold}")
                else:
                    break
            return _wmean(items), sum(d for _, _, d in items), items, expelled

        # 正式簇池（跨批持久化的成熟簇）与候选池（未成型，攒够 confirm_count 段才升级）
        pool: dict[str, dict] = {}
        cands: list[dict] = []
        for c in anon_existing:
            if not c.get("emb"):
                continue
            e = np.asarray(c["emb"], dtype=np.float32)
            if c.get("cand"):
                cands.append({"emb": e, "secs": float(c.get("secs", 0.0)),
                              "n": int(c.get("n", 1)), "members": [], "memb": [],
                              "cid": c.get("id", "")})
            else:
                pool[c["id"]] = {"emb": e, "secs": float(c.get("secs", 0.0))}
        _log(f"状态恢复: 正式簇{len(pool)} 候选{len(cands)} 下一起始号{nxt}")

        unmatched = []
        for si, a, b, emb in anon_units:
            if emb is None:
                unmatched.append((si, a, b, emb))
                continue
            dur = b - a
            # 1) 先比正式簇（成熟簇优先）
            best_id, best_sim = None, 0.0
            for cid, c in pool.items():
                s = float(np.dot(emb, c["emb"]))
                if s > best_sim:
                    best_sim, best_id = s, cid
            if best_id and best_sim >= anon_threshold:
                labels[(si, a, b)] = best_id
                cc = pool[best_id]
                w_new = best_sim * dur
                w = cc["secs"] + w_new
                ne = (cc["emb"] * cc["secs"] + emb * w_new) / w
                cc["emb"] = ne / (np.linalg.norm(ne) + 1e-9)
                cc["secs"] = w
                _log(f"入簇 {a:.2f}-{b:.2f} -> {best_id} sim={best_sim:.3f}")
                continue
            # 2) 再比候选池
            bi, bs = -1, 0.0
            for i, cd in enumerate(cands):
                s = float(np.dot(emb, cd["emb"]))
                if s > bs:
                    bs, bi = s, i
            if bi >= 0 and bs >= anon_threshold:
                cd = cands[bi]
                w_new = bs * dur
                w = cd["secs"] + w_new
                ne = (cd["emb"] * cd["secs"] + emb * w_new) / w
                cd["emb"] = ne / (np.linalg.norm(ne) + 1e-9)
                cd["secs"] = w
                cd["n"] += 1
                cd["members"].append((si, a, b))
                cd["memb"].append(((si, a, b), emb, dur))
                if len(cd["memb"]) > 8:
                    cd["memb"].pop(0)
                _log(f"候选合并 {a:.2f}-{b:.2f} -> cand#{bi} n={cd['n']} sim={bs:.3f}")
                if cd["n"] >= confirm_count:
                    cands.pop(bi)
                    cid = f"陌生人{nxt}"
                    nxt += 1
                    memb = cd["memb"] or [((None, a, b), cd["emb"], cd["secs"])]
                    cen, secs, kept, expelled = _robust_centroid(memb)
                    pool[cid] = {"emb": cen, "secs": secs}
                    for key, _, _ in kept:
                        if key and key[0] is not None:
                            labels[key] = cid
                    for key, e0, d0 in expelled:
                        if key and key[0] is not None:
                            cands.append({"emb": e0, "secs": d0, "n": 1,
                                          "members": [key], "memb": [(key, e0, d0)], "cid": ""})
                    _log(f"候选升级 {cid} n={cd['n']} secs={secs:.1f} 踢出{len(expelled)}")
                else:
                    labels[(si, a, b)] = None
                continue
            # 3) 成为新候选（不发号）
            cands.append({"emb": emb, "secs": dur, "n": 1,
                          "members": [(si, a, b)], "memb": [((si, a, b), emb, dur)], "cid": ""})
            _log(f"新候选 {a:.2f}-{b:.2f} dur={dur:.2f}")

        # 批内 gap 聚类（本 call 未匹配单元互比）：>=2 成员=互认直接正式；孤段转候选
        if unmatched:
            clusters = []
            for si, a, b, emb in unmatched:
                if emb is None:
                    continue  # 无 embedding 碎段最后兜底
                placed = False
                for c in clusters:
                    if float(np.dot(emb, c["centroid"])) >= anon_threshold:
                        c["members"].append((si, a, b, emb))
                        placed = True
                        break
                if not placed:
                    clusters.append({"members": [(si, a, b, emb)], "centroid": emb})
            final = []
            for c in clusters:
                members = list(c["members"])
                while len(members) > 1:
                    cen = np.mean([m[3] for m in members], axis=0)
                    cen = cen / (np.linalg.norm(cen) + 1e-9)
                    sims = sorted(((float(np.dot(m[3], cen)), i)
                                   for i, m in enumerate(members)), reverse=True)
                    cut = None
                    for t in range(len(sims) - 1):
                        if sims[t][0] - sims[t + 1][0] > split_gap:
                            cut = sims[t + 1][1]
                            break
                    if cut is None:
                        break
                    final.append([members.pop(cut)])
                final.append(members)
            for members in final:
                if len(members) >= 2:
                    cid = f"陌生人{nxt}"
                    nxt += 1
                    memb = [((si, a, b), e0, b - a) for si, a, b, e0 in members]
                    cen, secs, kept, expelled = _robust_centroid(memb)
                    for key, _, _ in kept:
                        labels[key] = cid
                    for key, e0, d0 in expelled:
                        cands.append({"emb": e0, "secs": d0, "n": 1,
                                      "members": [key], "memb": [(key, e0, d0)], "cid": ""})
                    pool[cid] = {"emb": cen, "secs": secs}
                    _log(f"批内互认成簇 {cid} n={len(kept)} secs={secs:.1f} 踢出{len(expelled)}")
                else:
                    si, a, b, e0 = members[0]
                    cands.append({"emb": e0, "secs": b - a, "n": 1,
                                  "members": [(si, a, b)], "memb": [((si, a, b), e0, b - a)], "cid": ""})
                    _log(f"批内孤段转候选 {a:.2f}-{b:.2f}")

        # 兜底升级（仅会末 final）：候选统一发正式编号；会中候选原样回传，段标"待定"
        if final:
            orphan_cids = []
            for cd in cands:
                cid = f"陌生人{nxt}"
                nxt += 1
                pool[cid] = {"emb": cd["emb"], "secs": cd["secs"]}
                for m in cd["members"]:
                    labels[m] = cid
                _log(f"兜底升级 {cid} n={cd['n']} secs={cd['secs']:.1f}")
                if cd["n"] == 1:
                    orphan_cids.append(cid)
        else:
            for k, cd in enumerate(cands):
                if not cd.get("cid"):
                    cd["cid"] = f"cand-{nxt}-{k}"
                for m in cd["members"]:
                    labels[m] = "待定"
        cands = []

        # 无 embedding 的匿名碎段：就近贴同句已标标签，实在没有给瞬态号
        transient = set()
        for si, subs in enumerate(out_subs):
            for a, b, name, emb in subs:
                if name is None and (si, a, b) not in labels:
                    near = [k for k in labels if k[0] == si and labels[k]]
                    if near:
                        k = min(near, key=lambda k: abs(k[1] - a))
                        labels[(si, a, b)] = labels[k]
                        _log(f"碎段贴同句 {a:.2f}-{b:.2f} -> {labels[k]}")
                    else:
                        cid = f"陌生人{nxt}"
                        nxt += 1
                        labels[(si, a, b)] = cid
                        transient.add(cid)
                        _log(f"碎段瞬态号 {a:.2f}-{b:.2f} -> {cid}")

        # ---------- 尾音归属（仅会末 final；孤段候选：先成熟簇吸收，再前后选边贴附） ----------
        if final and tail_attach and orphan_cids:
            timeline = []
            for si, subs in enumerate(out_subs):
                for a, b, name, emb in subs:
                    lab = name or labels.get((si, a, b))
                    if not lab or lab == "待定":
                        continue
                    e0 = emb if emb is not None else sent_emb.get(si)
                    timeline.append([float(a), float(b), lab, e0])
            timeline.sort(key=lambda t: t[0])
            for cid in orphan_cids:
                mine = [t for t in timeline if t[2] == cid]
                if not mine:
                    continue
                a0, b0, _, e0 = mine[0]
                if e0 is None:
                    continue
                dur = b0 - a0
                # a. 成熟簇吸收（>= 入簇地板）
                tgt, ts = None, 0.0
                for oid, c in pool.items():
                    if oid == cid:
                        continue
                    s = float(np.dot(e0, c["emb"]))
                    if s > ts:
                        ts, tgt = s, oid
                if tgt and ts >= anon_threshold:
                    cc = pool[tgt]
                    w_new = ts * dur
                    w = cc["secs"] + w_new
                    ne = (cc["emb"] * cc["secs"] + e0 * w_new) / w
                    cc["emb"] = ne / (np.linalg.norm(ne) + 1e-9)
                    cc["secs"] = w
                    for mkey, lab in list(labels.items()):
                        if lab == cid:
                            labels[mkey] = tgt
                    for t in timeline:
                        if t[2] == cid:
                            t[2] = tgt
                    pool.pop(cid)
                    _log(f"尾音吸收 {a0:.2f}-{b0:.2f} -> {tgt} sim={ts:.3f}")
                    continue
                # b. 前后选边贴附（像谁贴谁，两边都太不像则不贴）
                prev_t = next_t = None
                for t in timeline:
                    if t[2] == cid:
                        continue
                    if t[1] <= a0:
                        prev_t = t
                    elif t[0] >= b0 and next_t is None:
                        next_t = t
                sp = float(np.dot(e0, prev_t[3])) if prev_t and prev_t[3] is not None else -1.0
                sn = float(np.dot(e0, next_t[3])) if next_t and next_t[3] is not None else -1.0
                if sp < 0 and sn < 0:
                    _log(f"尾音不贴 {a0:.2f}-{b0:.2f} 前后无可用向量")
                    continue
                if sp >= sn:
                    best_side, best_s = (prev_t[2], sp) if prev_t else (None, sp)
                else:
                    best_side, best_s = (next_t[2], sn) if next_t else (None, sn)
                if best_side and best_s >= tail_min_sim:
                    for mkey, lab in list(labels.items()):
                        if lab == cid:
                            labels[mkey] = best_side
                    for t in timeline:
                        if t[2] == cid:
                            t[2] = best_side
                    pool.pop(cid)
                    _log(f"尾音贴附 {a0:.2f}-{b0:.2f} -> {best_side} (前={sp:.3f} 后={sn:.3f})")
                else:
                    _log(f"尾音不贴 {a0:.2f}-{b0:.2f} (前={sp:.3f} 后={sn:.3f} 均<{tail_min_sim})")

        # 显示层（仅会末 final）：总时长 < display_min_sec 的匿名簇不显示编号
        if final and display_min_sec > 0:
            in_call: dict[str, float] = {}
            for mkey, lab in labels.items():
                si, a, b = mkey
                in_call[lab] = in_call.get(lab, 0.0) + (b - a)
            for cid, c in pool.items():
                if not cid.startswith("陌生人"):
                    continue
                total = max(c["secs"], in_call.get(cid, 0.0))
                if total < display_min_sec:
                    _log(f"显示降级 {cid} total={total:.1f}s < {display_min_sec}s -> 未识别短语音")
                    for mkey, lab in list(labels.items()):
                        if lab == cid:
                            labels[mkey] = "未识别短语音"
            for cid in transient:
                total = in_call.get(cid, 0.0)
                if total < display_min_sec:
                    _log(f"显示降级(瞬态) {cid} total={total:.1f}s -> 未识别短语音")
                    for mkey, lab in list(labels.items()):
                        if lab == cid:
                            labels[mkey] = "未识别短语音"

        anon_out = [{"id": cid, "emb": c["emb"].tolist(), "secs": round(c["secs"], 1)}
                    for cid, c in pool.items()]
        if not final:
            for k, cd in enumerate(cands):
                anon_out.append({"id": cd["cid"], "cand": 1, "n": cd["n"],
                                 "emb": cd["emb"].tolist(), "secs": round(cd["secs"], 1)})

        segs = []
        for si, subs in enumerate(out_subs):
            for a, b, name, _ in subs:
                segs.append([float(a), float(b), si,
                             name or labels.get((si, a, b)) or "说话人1"])
        return {"segs": segs, "registered": sorted(refs.keys()), "anon": anon_out}
    finally:
        Path(tmp).unlink(missing_ok=True)


# ============ 声纹库管理（2026-09-15：唯一权威库 = speaker_enrolls/*.pcm） ============
# 背景：旧架构声纹有两套互不通气的存储——FunASR speaker_db.json（仅列表展示）
# 与 speaker_enrolls/*.pcm（ER 认人实际读取）。DELETE 只删前者导致"假删除"
# （删了照样被认出），两库成员互相不一致（幽灵名/隐形人）。
# 现收敛为单库：注册/列表/删除全部走本服务，db.json 废弃不再读写。
import time as _time
from fastapi import Request as _Request, Query as _Query


import re as _re_nm

# --- SeACo seaco_weight 透传补丁（2026-09-17）---
# FunASR 的 SeacoParaformer.inference -> _seaco_decode_with_ASF 不接收 kwargs.seaco_weight，
# 合并权重恒为默认 1.0（偏置=完全覆盖，过强）。此处 monkeypatch 把 generate 的
# seaco_weight 透传下去；λ 语义: logits=(1-λ)*dec+λ*dha，λ 越小偏置越温和。
import funasr.models.seaco_paraformer.model as _seaco_mod
_orig_seaco_decode = _seaco_mod.SeacoParaformer._seaco_decode_with_ASF
def _seaco_decode_patched(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens,
                          hw_list, nfilter=50, seaco_weight=1.0):
    w = getattr(self, "_hw_weight_override", None)
    if w is not None:
        seaco_weight = w
    return _orig_seaco_decode(self, encoder_out, encoder_out_lens, sematic_embeds,
                              ys_pad_lens, hw_list, nfilter, seaco_weight)
_seaco_mod.SeacoParaformer._seaco_decode_with_ASF = _seaco_decode_patched
_orig_seaco_infer = _seaco_mod.SeacoParaformer.inference
def _seaco_infer_patched(self, data_in, **kwargs):
    self._hw_weight_override = kwargs.get("seaco_weight")
    try:
        return _orig_seaco_infer(self, data_in, **kwargs)
    finally:
        self._hw_weight_override = None
_seaco_mod.SeacoParaformer.inference = _seaco_infer_patched

_PUNC = None
def _punc():
    global _PUNC
    if _PUNC is None:
        from funasr import AutoModel as _AM
        _PUNC = _AM(model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    device="cpu", disable_pbar=True, disable_log=True, disable_update=True)
    return _PUNC

_ITN = None
def _itn():
    global _ITN
    if _ITN is None:
        from itn.chinese.inverse_normalizer import InverseNormalizer
        _ITN = InverseNormalizer()
    return _ITN


def _name_of_stem(stem: str) -> str:
    """注册段文件名 -> 人名：剥掉 _<ts> 或 _<ts>_<idx> 时间戳后缀。
    智能注册落盘为 {name}_{ts}_{i}.pcm；旧的 rsplit("_",1) 会把这类文件
    错解析成人名"xx_1789..."（2026-09-15 测试实测命中）。"""
    m = _re_nm.match(r"^(.*)_\d+_\d+$", stem) or _re_nm.match(r"^(.*)_\d+$", stem)
    return m.group(1) if m else stem


def _enroll_names() -> list[str]:
    d = Path(_SPEAKER_ENROLL_DIR)
    if not d.is_dir():
        return []
    return sorted({_name_of_stem(p.stem) for p in d.glob("*.pcm")})


def _drop_enroll_files(nm: str) -> list[str]:
    """删除某人的全部注册段文件并失效缓存，返回删掉的文件名。"""
    d = Path(_SPEAKER_ENROLL_DIR)
    removed = []
    if d.is_dir():
        for p in d.glob("*.pcm"):
            stem = p.stem
            pname = _name_of_stem(stem)
            if pname == nm:
                p.unlink(missing_ok=True)
                removed.append(p.name)
    return removed


@app.get("/speaker_names")
async def speaker_names():
    """已注册声纹列表（pcm 目录前缀去重，唯一权威）。"""
    return {"speakers": _enroll_names()}


@app.post("/speaker_delete")
async def speaker_delete(name: str = Form(...)):
    """删除声纹：删掉此人全部 pcm 注册段并失效缓存（真删除，立即生效）。"""
    nm = (name or "").strip()
    if not nm:
        return {"ok": False, "error": "name required"}
    removed = _drop_enroll_files(nm)
    return {"ok": True, "removed": removed, "speakers": _enroll_names()}


@app.get("/speaker_segments")
async def speaker_segments(name: str = _Query(..., min_length=1, max_length=64)):
    """列出某人的全部注册段（试听质检用）。返回每段的序号/文件名/时长秒。"""
    import glob as _g
    nm = (name or "").strip()
    import re as _re
    pat = _re.escape(nm) + r"_([0-9]+)_([0-9]+)\.pcm$"
    segs = []
    for f in sorted(_g.glob(f"{_SPEAKER_ENROLL_DIR}/{nm}_*.pcm")):
        m = _re.search(pat, f)
        if not m:
            continue
        segs.append({"i": int(m.group(2)), "file": f.split("/")[-1],
                     "dur": round(len(open(f, "rb").read()) / 2 / 16000.0, 2)})
    return {"name": nm, "segments": segs}


@app.get("/speaker_segment_audio")
async def speaker_segment_audio(name: str = _Query(..., min_length=1, max_length=64),
                                i: int = _Query(...)):
    """返回某注册段的 wav 音频流（16k/16bit/mono），前端 <audio> 直接播。"""
    import glob as _g
    import wave as _wave
    import io as _io
    from fastapi.responses import Response as _Resp
    nm = (name or "").strip()
    hits = sorted(_g.glob(f"{_SPEAKER_ENROLL_DIR}/{nm}_*_{i}.pcm"))
    if not hits:
        return _Resp(status_code=404, content=b"segment not found")
    pcm = open(hits[0], "rb").read()
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(pcm)
    return _Resp(content=buf.getvalue(), media_type="audio/wav")



@app.post("/speaker_enroll")
async def speaker_enroll(request: _Request, name: str = _Query(..., min_length=1, max_length=32),
                         min_total_sec: float = Form(default=5.0),
                         min_seg_sec: float = Form(default=2.0),
                         min_segments: int = Form(default=3),
                         min_effective_sec: float = Form(default=10.0),
                         max_keep: int = Form(default=5),
                         consistency: float = Form(default=0.45),
                         consistency_gap: float = Form(default=0.12),
                         vad_end_silence_ms: int = Form(default=800)):
    """智能注册：raw PCM(int16/16k/mono) -> fsmn VAD 切段 -> 逐段质量过滤
    -> ER 一致性选优 -> 留 3~5 段写 {name}_{ts}_{i}.pcm（多段质心），旧段整批替换。

    录入是全场识别的参照物来源，标准与识别相反（宁缺勿滥）：太短/太轻/爆音/
    与其它段声纹不一致（咳嗽、旁人插话、突发环境声）的段一律丢弃。
    不合格不落库；返回 kept/rejected 明细报告。

    补录（append，2026-09-15 用户定义"联合重选"）：同名已注册时，旧段文件 +
    本次合格段进同一池子，复用一致性双规则（排序 gap>0.12 截断 + 地板）选优，
    选中段（新旧混合，最多 max_keep）整批重写回库、质心重建——同一个人的段
    互相聚成一团，离群段被票出（防"万磁王"质心）。无"新段须匹配旧质心"门槛，
    跨域新数据不会被旧域拒之门外。新名字仍是严格模式（>=3段且>=10s）。
    """
    import re as _re
    nm = name.strip()
    if not _re.fullmatch(r"[\w\u4e00-\u9fff\-]+", nm):
        return {"ok": False, "error": "姓名只允许中文/英文/数字/下划线/横线"}
    raw = await request.body()
    if len(raw) % 2:
        return {"ok": False, "error": "音频数据格式错误（奇数字节）"}
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total = len(audio) / 16000.0
    if total < min_total_sec:
        return {"ok": False, "error": f"录音太短（{total:.1f}s < {min_total_sec:.0f}s），请自然说 20~60 秒"}
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 10 ** (-50 / 20):
        return {"ok": False, "error": "整体音量过低，几乎全是环境噪音，请靠近麦克风"}
    # 归一化到 RMS 0.1：与会议管线 meeting_chunk 处理域一致
    audio = np.clip(audio * (0.1 / rms), -1.0, 1.0)
    r = _vad().generate(input=audio.copy(), fs=16000, max_end_silence_time=vad_end_silence_ms)
    vad = r[0]["value"] if isinstance(r, list) else r["value"]
    segs = [[a / 1000.0, b / 1000.0] for a, b in vad]
    if not segs:
        return {"ok": False, "error": "未检测到语音，请对着麦克风自然说话"}
    # 逐段质量过滤
    kept, rejected = [], []
    ere = _ere()
    for i, (a, b) in enumerate(segs):
        dur = b - a
        if dur < min_seg_sec:
            rejected.append({"i": i, "dur": round(dur, 2), "reason": f"太短(<{min_seg_sec}s)"})
            continue
        clip = audio[int(a * 16000):int(b * 16000)]
        sr_ = float(np.sqrt(np.mean(clip ** 2)))
        if sr_ < 10 ** (-42 / 20):
            rejected.append({"i": i, "dur": round(dur, 2), "reason": "音量过低(疑似环境音)"})
            continue
        if float(np.mean(np.abs(clip) > 0.999)) > 0.01:
            rejected.append({"i": i, "dur": round(dur, 2), "reason": "爆音(麦克风太近/增益过高)"})
            continue
        e = np.asarray(ere.generate(input=clip, fs=16000, extract_embedding=True)[0]["spk_embedding"],
                       dtype=np.float32).ravel()
        kept.append({"i": i, "start": round(a, 2), "end": round(b, 2), "dur": round(dur, 2),
                     "emb": e / (np.linalg.norm(e) + 1e-9)})
    if not kept:
        return {"ok": False, "error": "没有合格的语音段，请重录", "rejected": rejected}
    # 一致性选优（双规则，迭代至收敛）：
    #  a) 排序截断 gap：与质心 sim 降序后最大相邻落差 > consistency_gap -> 低侧整组
    #     视为异质（他人插话/咳嗽；同批真段抖动仅 ±0.05，相似嗓音跨人 0.52+ 也能切开）
    #  b) 绝对地板 sim < consistency -> 逐段剔除
    while len(kept) >= 2:
        c = np.mean([k["emb"] for k in kept], axis=0)
        c /= np.linalg.norm(c) + 1e-9
        order = sorted(range(len(kept)), key=lambda j: -(float(kept[j]["emb"] @ c)))
        sims = {j: float(kept[j]["emb"] @ c) for j in range(len(kept))}
        cut = None
        for t in range(len(order) - 1):
            if sims[order[t]] - sims[order[t + 1]] > consistency_gap:
                cut = t + 1  # order[cut:] 为低侧异质组
                break
        if cut is not None and cut < len(order):
            drop = set(order[cut:])
            for j in sorted(drop, reverse=True):
                k = kept.pop(j)
                rejected.append({"i": k["i"], "dur": k["dur"],
                                 "reason": f"声纹不一致({sims[j]:.2f}，与主簇落差>{consistency_gap:.2f}，疑似他人/咳嗽/环境声)"})
            continue
        worst = min(range(len(kept)), key=lambda j: sims[j])
        if sims[worst] >= consistency:
            break
        k = kept.pop(worst)
        rejected.append({"i": k["i"], "dur": k["dur"],
                         "reason": f"声纹不一致({sims[worst]:.2f}<{consistency:.2f}，疑似他人/咳嗽/环境声)"})
    # ---------- 补录（append）= 新旧联合重选 ----------
    # 旧实现：新段须与旧质心 >=consistency 才收——跨域时新数据被自己门槛拒之门外。
    # 现实现：旧段文件 + 新段进同一池子，复用一致性双规则选优，选中段（新旧混合）
    # 整批重写回库、质心重建；同一个人的段互相聚成一团，离群段被票出。
    append_mode = nm in _enroll_names()
    old_pool: list[dict] = []
    if append_mode:
        for p in sorted(Path(_SPEAKER_ENROLL_DIR).glob("*.pcm")):
            if _name_of_stem(p.stem) != nm:
                continue
            raw_b = p.read_bytes()
            pcm_old = np.frombuffer(raw_b, dtype=np.int16).astype(np.float32) / 32768.0
            if len(pcm_old) < 16000 * 1.5:
                continue
            r_ = float(np.sqrt(np.mean(pcm_old ** 2)))
            if r_ > 1e-6:
                pcm_old = np.clip(pcm_old * (0.1 / r_), -1.0, 1.0)
            try:
                e = np.asarray(ere.generate(input=pcm_old, fs=16000, extract_embedding=True)[0]["spk_embedding"],
                               dtype=np.float32).ravel()
            except Exception:
                continue
            old_pool.append({"i": f"old:{p.name}", "dur": len(pcm_old) / 16000.0,
                             "emb": e / (np.linalg.norm(e) + 1e-9), "bytes": raw_b})
    if append_mode and old_pool:
        if not kept:
            return {"ok": False, "appended": True,
                    "error": "本次录音没有合格新段（太短/太轻/爆音），旧声纹原样保留，"
                             "请靠近麦克风多说几句再补录",
                    "rejected": rejected}
        pool = old_pool + kept
        # 地板规则（2026-09-15 修订：append 只用地板，不用 gap 截断——
        # gap 是为"同批清洗异质段"设计的，跨环境同人的段天然分布宽
        # （实测同人跨会话 0.5~0.66 vs 同会话 0.7+，gap 会把合法的新环境段
        # 误票出，使补录永远无法跨环境累积）。append 是本人主动声明"这是我"，
        # >=consistency 即收，异质段（他人/咳嗽）多数也在地板以下被挡。
        while len(pool) >= 2:
            cen = np.mean([x["emb"] for x in pool], axis=0)
            cen = cen / (np.linalg.norm(cen) + 1e-9)
            sims = {j: float(pool[j]["emb"] @ cen) for j in range(len(pool))}
            worst = min(range(len(pool)), key=lambda j: sims[j])
            if sims[worst] >= consistency:
                break
            x = pool.pop(worst)
            rejected.append({"i": x["i"], "dur": round(x["dur"], 2),
                             "reason": f"联合重选被票出({sims[worst]:.2f}<{consistency:.2f})"})
        if not pool:
            return {"ok": False, "appended": True,
                    "error": "新旧段互相都不聚类（可能非同一人或跨域过大），旧声纹已保留，"
                             "建议删除后重新注册",
                    "rejected": rejected}
        pool.sort(key=lambda x: -x["dur"])
        pool = pool[:max_keep]
        kept_old = [x for x in pool if str(x["i"]).startswith("old:")]
        kept_new = [x for x in pool if not str(x["i"]).startswith("old:")]
        d = Path(_SPEAKER_ENROLL_DIR)
        d.mkdir(parents=True, exist_ok=True)
        ts = int(_time.time())
        for j, x in enumerate(pool):
            if "bytes" in x:
                (d / f"{nm}_{ts}_{j}.pcm").write_bytes(x["bytes"])
            else:
                clip = audio[int(x["start"] * 16000):int(x["end"] * 16000)]
                (d / f"{nm}_{ts}_{j}.pcm").write_bytes(
                    (np.clip(clip, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        # 池里落选的旧段文件一并清掉（已不参与质心）
        keep_names = {f"{nm}_{ts}_{j}.pcm" for j in range(len(pool))}
        for p in Path(_SPEAKER_ENROLL_DIR).glob("*.pcm"):
            if _name_of_stem(p.stem) == nm and p.name not in keep_names:
                p.unlink(missing_ok=True)
        effective = sum(x["dur"] for x in pool)
        total_segs = len(pool)
        pending = total_segs < 3 and effective < 8.0
        return {"ok": True, "name": nm, "appended": True, "total_sec": round(total, 1),
                "effective_sec": round(effective, 1),
                "kept_old": len(kept_old), "kept_new": len(kept_new),
                "total_segments": total_segs, "pending": pending,
                "msg": ("已保存，待凑齐 3 段或 8s 后自动激活，同名再录继续累积"
                        if pending else
                        f"联合重选完成：保留旧 {len(kept_old)} 段 + 新 {len(kept_new)} 段，"
                        f"共 {total_segs} 段/{effective:.1f}s，质心已重建"),
                "kept": [{"dur": round(x["dur"], 2), "src": "旧" if str(x["i"]).startswith("old:") else "新"}
                          for x in pool],
                "rejected": rejected}
    # 按时长优先取 top max_keep（长段 embedding 更稳）
    kept.sort(key=lambda k: -k["dur"])
    kept = kept[:max_keep]
    effective = sum(k["dur"] for k in kept)
    if not kept:
        # 失败留存：原始音频落盘（仅诊断用，绝不进认人库），防"录了却说没存"
        try:
            fd = Path(_SPEAKER_ENROLL_DIR).parent / "enroll_failed"
            fd.mkdir(parents=True, exist_ok=True)
            (fd / f"{nm}_{int(_time.time())}.pcm").write_bytes(raw)
        except Exception as _e:
            print(f"  enroll_failed save failed: {_e}", flush=True)
        return {"ok": False,
               "error": "没有合格语音段（太短/太轻/爆音）。原始音频已留存，请联系管理员排查",
               "rejected": rejected}
    # 新名字=整批落盘；append 且无旧段文件（理论边缘）同路径
    d = Path(_SPEAKER_ENROLL_DIR)
    d.mkdir(parents=True, exist_ok=True)
    if not append_mode:
        _drop_enroll_files(nm)
    ts = int(_time.time())
    for j, k in enumerate(kept):
        clip = audio[int(k["start"] * 16000):int(k["end"] * 16000)]
        (d / f"{nm}_{ts}_{j}.pcm").write_bytes(
            (np.clip(clip, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    total_segs = len(kept)
    pending = total_segs < 3 and effective < 8.0
    return {"ok": True, "name": nm, "appended": append_mode, "total_sec": round(total, 1),
            "effective_sec": round(effective, 1), "segments": len(kept),
            "total_segments": total_segs, "pending": pending,
            "msg": ("已保存，待凑齐 3 段或 8s 后自动激活参与认人，同名再录会累积"
                    if pending else "已注册/更新"),
            "kept": [{"dur": k["dur"], "start": k["start"], "end": k["end"]} for k in kept],
            "rejected": rejected}


@app.post("/speaker_identify")
async def speaker_identify(request: _Request,
                           min_seg_sec: float = Form(default=1.5),
                           match_threshold: float = Form(default=0.40),
                           weak_threshold: float = Form(default=0.30)):
    """录入检测：说几秒 -> 命中声纹库里的谁（返回全候选相似度降序）。

    兼作注册质量自查：注册完立刻测，能稳定认出自己=注册合格。
    verdict: matched(>=match_threshold) / weak(>=weak_threshold) / unknown。
    """
    raw = await request.body()
    if len(raw) % 2:
        return {"ok": False, "error": "音频数据格式错误（奇数字节）"}
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total = len(audio) / 16000.0
    if total < 1.5:
        return {"ok": False, "error": f"太短（{total:.1f}s），请说 3~10 秒"}
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 10 ** (-55 / 20):
        return {"ok": False, "error": "几乎没声音，请靠近麦克风再说"}
    audio = np.clip(audio * (0.1 / rms), -1.0, 1.0)
    r = _vad().generate(input=audio.copy(), fs=16000, max_end_silence_time=800)
    vad = r[0]["value"] if isinstance(r, list) else r["value"]
    segs = [[a / 1000.0, b / 1000.0] for a, b in vad if (b - a) / 1000.0 >= min_seg_sec]
    if not segs:
        # 回退：VAD 切不出长段（安静/平稳语音）时整段当一段（识别端点已对整段归一化）
        if total >= 2.0:
            segs = [[0.0, total]]
        else:
            return {"ok": False, "error": "未检测到足够长的语音，请说 3~10 秒"}
    ere = _ere()
    vecs = []
    for a, b in segs[:8]:  # 最多 8 段，平均抑制单段噪声
        clip = audio[int(a * 16000):int(b * 16000)]
        e = np.asarray(ere.generate(input=clip, fs=16000, extract_embedding=True)[0]["spk_embedding"],
                       dtype=np.float32).ravel()
        vecs.append(e / (np.linalg.norm(e) + 1e-9))
    q = np.mean(vecs, axis=0)
    q /= np.linalg.norm(q) + 1e-9
    refs = _load_db_multi()
    if not refs:
        return {"ok": False, "error": "声纹库为空，请先注册"}
    ranked = sorted(({"name": n, "sim": round(float(q @ c), 3)} for n, c in refs.items()),
                    key=lambda x: -x["sim"])
    best = ranked[0]
    margin = round(best["sim"] - ranked[1]["sim"], 3) if len(ranked) > 1 else None
    verdict = "matched" if best["sim"] >= match_threshold else               "weak" if best["sim"] >= weak_threshold else "unknown"
    return {"ok": True, "total_sec": round(total, 1), "segments": len(vecs),
            "best": best["name"], "best_sim": best["sim"], "margin": margin,
            "verdict": verdict, "ranked": ranked}


# ============ /identify_segments：外部切段（3D-Speaker）认人+转写（换芯一期 B，2026-09-20） ============
# 会末终版唯一生产分离路径的 GPU 侧搭档：本地已完成 GTCRN 降噪 + 3D-Speaker
# 匿名聚类，边界由调用方给定（segments=[[start_s,end_s,cluster_id],...]）。
# 本端点只做两件事，不做任何切分/聚类（职责单一，不碰旧端点行为）：
#   1) 逐簇 ERes2NetV2 embedding（簇内段时长加权质心）→ 声纹库
#      （speaker_enrolls 多段质心）匈牙利一对一认人，未命中/低于阈值 → "说话人N"
#   2) 逐段 SeACo ASR（热词 + CT-Transformer 标点 + ITN，与 meeting_chunk 同一套模型）
# 复用现有 _ere/_load_db_multi/_asr/_punc/_itn，不重造模型。

@app.post("/identify_segments")
async def identify_segments(file: UploadFile = File(...), segments: str = Form(default="[]"),
                            name_threshold: float = Form(default=0.40),
                            hotwords: str = Form(default=""),
                            hw_weight: float = Form(default=1.0),
                            punc: bool = Form(default=False), itn: bool = Form(default=False)):
    """wav + 外部切段 -> [{start,end,cluster_id,speaker,text}]。

    一次 HTTP 批量：整段 wav + 全部边界（无逐段请求）。speaker 认人 =
    段 embedding（逐段 RMS 归一化）→ 簇时长加权质心 → 逐簇 argmax（允许重名）
    + 段级复核（混簇少数派段独立改判）；text 逐段转写（文字与段边界一一对应）。
    """
    import tempfile
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        audio, sr = sf.read(tmp)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        seg_list = [(float(a), float(b), int(c)) for a, b, c in json.loads(segments or "[]")]
        if not seg_list:
            return {"segs": [], "error": "segments empty"}

        # RMS 归一化（与 meeting_chunk 同域：远场轻录音 VAD 漏检修复）
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-6 and abs(rms - 0.1) / 0.1 > 0.3:
            audio = np.clip(audio * (0.1 / rms), -1.0, 1.0)

        # ---- 1) 簇级认人：段 embedding（逐段 RMS 归一化，轻声段不出废向量）→
        #         时长加权质心 → 逐簇 argmax（允许重名：同一人常有多簇）----
        ere = _ere()
        refs = _load_db_multi()
        cluster_ids = sorted({c for _, _, c in seg_list})
        cluster_cent: dict[int, np.ndarray] = {}
        for cid in cluster_ids:
            vecs, wts = [], []
            for a, b, c in seg_list:
                if c != cid:
                    continue
                dur = b - a
                if dur < 1.0:
                    continue
                clip = audio[int(a * sr):int(b * sr)]
                e = np.asarray(ere.generate(input=clip, fs=sr, extract_embedding=True)[0]
                               ["spk_embedding"], dtype=np.float32).ravel()
                vecs.append(e / (np.linalg.norm(e) + 1e-9))
                wts.append(dur)
            if not vecs:
                # 全碎段簇：退整簇整段一次 embedding（用最长段兜底）
                a, b = max(((a, b) for a, b, c in seg_list if c == cid),
                           key=lambda x: x[1] - x[0])
                clip = audio[int(a * sr):int(b * sr)]
                e = np.asarray(ere.generate(input=clip, fs=sr, extract_embedding=True)[0]
                               ["spk_embedding"], dtype=np.float32).ravel()
                vecs = [e / (np.linalg.norm(e) + 1e-9)]
                wts = [1.0]
            m = np.average(vecs, axis=0, weights=wts)
            cluster_cent[cid] = m / (np.linalg.norm(m) + 1e-9)
        # （实验回滚 2026-09-24：逐段 RMS 归一化 + 段级复核 DER 6.56 未优于
        #  v1 6.52，证据见 eval/tuning_log.md，生产用 v1 簇级逻辑）

        cluster_name: dict[int, str] = {}
        cluster_score: dict[int, float] = {}
        cluster_margin: dict[int, float] = {}
        if refs and cluster_cent:
            # 逐簇独立认人：和谁最像就选谁（允许重名）。
            # 同一人常被 3D 批内拆成多簇，匈牙利一对一强制"名不重复"会把
            # 多出的簇挤给"下一个压线的人"——2026-09-24 实锤：纯 001-M 段
            # 因 001-M 名被强簇占走，被挂到 006-F（0.431 压 0.40 线）。
            # 重名场景（两个不同的人同 argmax 一名）发生频率低且嵌入模糊时
            # 匈牙利同样救不回，故移除一对一约束。
            for c in sorted(cluster_cent):
                ranked = sorted(((n, float(cluster_cent[c] @ refs[n])) for n in refs),
                                key=lambda x: -x[1])
                nm, s = ranked[0]
                margin = float(s - ranked[1][1]) if len(ranked) > 1 else 1.0
                cluster_score[c] = s
                cluster_margin[c] = margin
                if s >= name_threshold:
                    cluster_name[c] = nm
                    second = f" 第二名 {ranked[1][0]}:{ranked[1][1]:.3f}" if len(ranked) > 1 else ""
                    print(f"  [identify] cluster {c} -> {nm} ({s:.3f}; margin={margin:.3f};{second})", flush=True)
            # 未命中簇统一匿名编号（按首次出现顺序）
            anon = 0
            first_seen: dict[int, float] = {}
            for a, _, c in seg_list:
                first_seen.setdefault(c, a)
            for c in sorted(cluster_ids, key=lambda x: first_seen[x]):
                if c not in cluster_name:
                    anon += 1
                    cluster_name[c] = f"说话人{anon}"
                    print(f"  [identify] cluster {c} -> 说话人{anon} (below thr)", flush=True)
        else:
            # 空库（盲测态）：全部匿名
            anon = 0
            first_seen: dict[int, float] = {}
            for a, _, c in seg_list:
                first_seen.setdefault(c, a)
            for c in sorted(cluster_ids, key=lambda x: first_seen[x]):
                anon += 1
                cluster_name[c] = f"说话人{anon}"

        # ---- 2) 逐段 ASR（与 meeting_chunk v2 同一套：SeACo 热词 + punc + itn） ----
        asr = _asr()
        segs = []
        for a, b, c in seg_list:
            clip = audio[int(a * sr):int(b * sr)]
            if len(clip) < sr * 0.1:
                txt = ""
            elif hotwords:
                rr = asr.generate(input=clip, fs=sr, hotword=hotwords, seaco_weight=hw_weight)
                txt = (rr[0]["text"] if isinstance(rr, list) else rr["text"]).strip()
            else:
                rr = asr.generate(input=clip, fs=sr)
                txt = (rr[0]["text"] if isinstance(rr, list) else rr["text"]).strip()
            if txt and punc:
                try:
                    rp = _punc().generate(input=txt.replace(" ", ""))
                    txt = (rp[0]["text"] if isinstance(rp, list) else rp["text"]).strip()
                except Exception:
                    pass
            if txt and itn:
                try:
                    txt = _itn().normalize(txt)
                except Exception:
                    pass
            segs.append({"start": round(a, 3), "end": round(b, 3),
                         "cluster_id": c,
                         "speaker": cluster_name.get(c, "说话人1"),
                         "score": cluster_score.get(c),
                         "margin": cluster_margin.get(c),
                         "text": txt})
        return {"segs": segs, "n_clusters": len(cluster_ids),
                "registered": sorted(refs.keys())}
    finally:
        Path(tmp).unlink(missing_ok=True)


# ============ /denoise：GTCRN 降噪（2026-09-30 收编：本地 torch 退役，运算全部上服务器） ============
# 与本地 app/core/preprocess.denoise_pcm 同一实现、同一权重（gtcrn_dns3.tar），
# 保证会中/会末/注册/识别走完全一致的信号域。部署：gtcrn_net.py 与本文件同目录，
# 权重在 ~/jvkit/asr/gtcrn_dns3.tar。
_GTCRN = None
_GTCRN_WIN = None


def _gtcrn():
    global _GTCRN, _GTCRN_WIN
    if _GTCRN is None:
        import sys as _sys
        _d = str(Path(__file__).resolve().parent)
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        from gtcrn_net import GTCRN
        m = GTCRN().eval()
        ckpt = torch.load(os.environ.get("GTCRN_CKPT", f"{_ASR_ROOT}/gtcrn_dns3.tar"), map_location=torch.device("cpu"))
        m.load_state_dict(ckpt["model"])
        _GTCRN_WIN = torch.hann_window(512).pow(0.5)
        _GTCRN = m
        print("[denoise] GTCRN loaded", flush=True)
    return _GTCRN


@app.post("/denoise")
async def denoise_ep(file: UploadFile = File(...)):
    """裸 int16/16k/mono PCM → 降噪后同长度 PCM（application/octet-stream）。"""
    from fastapi.responses import Response
    pcm = await file.read()
    try:
        if len(pcm) % 2:
            pcm = pcm[: len(pcm) - 1]
        if len(pcm) < 512 * 2:
            return Response(content=pcm, media_type="application/octet-stream")
        model = _gtcrn()
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        inp = torch.stft(torch.from_numpy(wav), 512, 256, 512, _GTCRN_WIN, return_complex=False)
        with torch.no_grad():
            output = model(inp[None])[0]
        enh = torch.istft(torch.view_as_complex(output.contiguous()),
                          512, 256, 512, _GTCRN_WIN, return_complex=False)
        arr = enh.detach().cpu().numpy()
        if len(arr) < len(wav):
            arr = np.concatenate([arr, np.zeros(len(wav) - len(arr), dtype=np.float32)])
        elif len(arr) > len(wav):
            arr = arr[: len(wav)]
        out = np.clip(arr * 32768.0, -32768, 32767).astype(np.int16)
        return Response(content=out.tobytes(), media_type="application/octet-stream")
    except Exception as e:
        # 降噪失败回退原始音频（与本地版同一兜底语义）
        print(f"[denoise] failed, passthrough: {e}", flush=True)
        return Response(content=pcm, media_type="application/octet-stream")


# ============ /diarize_full：3D-Speaker 整场分离（2026-09-30 收编本地 diarize3ds） ============
# 本地 app/core/diarize3ds.py 的服务器版：同一 third_party/3D-Speaker 源码、同一
# 聚类参数（与本地 config diar3ds_* 对齐，AGENTS 约定 GPU 默认值必须与本地一致）。
# 部署：third_party_3dspeaker/ 在 ~/jvkit/asr/ 下（speakerlab 包在其根目录）。
_DIAR3DS = None


def _diar3ds():
    global _DIAR3DS
    if _DIAR3DS is None:
        import sys as _sys
        _tp = os.environ.get("THIRD_PARTY_3DSPEAKER", f"{_ASR_ROOT}/third_party_3dspeaker")
        if _tp not in _sys.path:
            _sys.path.insert(0, _tp)
        from speakerlab.bin.infer_diarization import Diarization3Dspeaker
        p = Diarization3Dspeaker(device="cpu")
        cl = p.cluster
        cl.cluster_type = "spectral"      # diar3ds_cluster_type
        cl.cluster_line = 40              # 子段数 < 此值走 AHC 短段分支
        cl.mer_cos = 0.8                  # 簇心 cos≥此值再合并
        cl.min_cluster_size = 4           # ≤此段数的簇重指派到最近大簇
        cl.cluster.min_num_spks = 1
        cl.cluster.max_num_spks = 15      # 谱聚类选 k 上限
        cl.cluster.pval = 0.012           # 相似度矩阵剪枝阈值
        cl.cluster.k = None               # 不指定人数，自动估 k
        p.speaker_num = None
        _DIAR3DS = p
        print("[diarize_full] 3D-Speaker pipeline loaded", flush=True)
    return _DIAR3DS


@app.post("/diarize_full")
async def diarize_full(file: UploadFile = File(...), emb: str = Form(default="cam")):
    """wav → [{start, end, cluster_id}]（秒）。整段输入整段聚类，无滑窗。

    emb: "cam"（CAM++ 3D 原生，默认）| "ere"（ERes2NetV2 认人同款，与声纹库
    同尺子——2026-10-10 7444 实测 CAM++ 分不开同场两人（簇内 cos 仅 0.64），
    换 ERes2NetV2 让分簇与认人用同一把尺子，零迁移成本）。
    """
    data = await file.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        return {"segs": [], "error": f"仅支持 16kHz，收到 {sr}"}
    wav = torch.from_numpy(np.ascontiguousarray(audio))[None, :]  # [1,T] int16
    if emb == "ere":
        labels = _diarize_ere(wav)
    else:
        labels = _diar3ds()(wav, wav_fs=16000)
    return {"segs": [{"start": float(st), "end": float(ed), "cluster_id": int(cid)}
                     for st, ed, cid in labels]}


def _diarize_ere(wav):
    """ERes2NetV2 嵌入版整场分离：VAD 切段（复用 3D 的）→ ERes2NetV2 逐段
    embedding（与声纹库/identify_segments 同模型同尺子）→ 3D 谱聚类分簇。"""
    p = _diar3ds()
    wav_data = wav.float() / 32768.0 if wav.dtype in (torch.int16, torch.int32) else wav.float()
    vad_time = p.do_vad(wav_data)
    chunks = [c for (st, ed) in vad_time for c in p.chunk(st, ed)]
    if not chunks:
        return []
    ere = _ere()
    embs = []
    for st, ed in chunks:
        clip = wav_data[0, int(st * p.fs):int(ed * p.fs)].numpy()
        e = np.asarray(ere.generate(input=clip, fs=16000, extract_embedding=True)[0]
                       ["spk_embedding"], dtype=np.float32).ravel()
        embs.append(e / (np.linalg.norm(e) + 1e-9))
    X = np.stack(embs)
    _, out = p.do_clustering(chunks, X)
    return out


@app.post("/speaker_rename")
async def speaker_rename(old_name: str = Form(...), new_name: str = Form(...)):
    """重命名声纹：把此人全部注册段文件改到新名下并重建质心缓存。
    前端"重命名"用（打错名字可改，不用删了重录）。"""
    import re as _re2
    old = (old_name or "").strip()
    new = (new_name or "").strip()
    if not old or not new or old == new:
        return {"ok": False, "error": "姓名无效或未变化"}
    if not _re2.fullmatch(r"[\w\u4e00-\u9fff\-]+", new):
        return {"ok": False, "error": "新姓名只允许中文/英文/数字/下划线/横线"}
    d = Path(_SPEAKER_ENROLL_DIR)
    renamed = []
    for p in sorted(d.glob("*.pcm")):
        if _name_of_stem(p.stem) == old:
            # 文件名 = {name}_{ts}_{i}.pcm，保留 ts/i，只换 name 部分
            m = _re2.match(r"^(.*)_(\d+_\d+)$", p.stem)
            suffix = m.group(2) if m else str(int(_time.time()))
            p.rename(d / f"{new}_{suffix}.pcm")
            renamed.append(p.name)
    return {"ok": True, "renamed": renamed, "speakers": _enroll_names()}


@app.post("/speaker_segment_delete")
async def speaker_segment_delete(name: str = Form(...), i: int = Form(...)):
    """删除某人的一个注册段（其余段保留，质心自动重建）。
    试听发现某段混入他人声音/太噪时单独删，不用整人重录。"""
    import glob as _g
    nm = (name or "").strip()
    hits = sorted(_g.glob(f"{_SPEAKER_ENROLL_DIR}/{nm}_*_{i}.pcm"))
    if not hits:
        return {"ok": False, "error": "段不存在"}
    removed = []
    for f in hits:
        Path(f).unlink(missing_ok=True)
        removed.append(Path(f).name)
    return {"ok": True, "removed": removed, "speakers": _enroll_names()}


@app.post("/segments_sim")
async def segments_sim(request: _Request):
    """一组音频段 vs 自身质心的逐段相似度（2026-10-09）。

    用途：会末注册弹窗的"默认勾选 40s 最像的段" + 低分段标红提示"疑似混入他人"；
    声纹库内"局部 vs 整体"段级质检（哪段拖质心后腿一眼可见）。

    输入：raw body = int16/16k/mono PCM，由 |XXXX|（4 字节大端长度）分隔的多段。
    输出：{"ok": true, "sims": [float...], "centroid_sim": float}
      - sims[i] = 第 i 段 embedding vs 全体段质心的余弦相似度
      - centroid_sim = 全体段质心的自紧密度（近似，仅参考）
    段数 <2 或全部 embedding 提取失败返回 ok=False。
    """
    import io as _io
    import re as _re3
    raw = await request.body()
    # 分隔符：任意非 PCM 语义的 4+ 字节序列都可能混入音频，用显式魔法头：
    # 客户端每段前加 b"SEGS" + 4字节大端长度，再音频数据
    parts = []
    i = 0
    magic = b"SEGS"
    while i + 8 <= len(raw):
        if raw[i:i + 4] != magic:
            return {"ok": False, "error": f"bad segment header at offset {i}"}
        n = int.from_bytes(raw[i + 4:i + 8], "big")
        i += 8
        if i + n > len(raw):
            return {"ok": False, "error": "segment length overflow"}
        parts.append(raw[i:i + n])
        i += n
    if not parts:
        return {"ok": False, "error": "no segments"}
    ere = _ere()
    embs = []
    for j, p in enumerate(parts):
        if len(p) < 16000 * 2 * 0.5:  # <0.5s 的段 embedding 不稳
            embs.append(None)
            continue
        if len(p) % 2:  # VAD 切点带小数 ms → 本地切片可能出奇数字节，截齐防崩（2026-10-09 事故）
            p = p[:-1]
        audio = np.frombuffer(p, dtype=np.int16).astype(np.float32) / 32768.0
        # 居中截 8s：说话人 embedding 8s 即饱和，长段整跑纯浪费 CPU（2026-10-09 实测提速）
        if len(audio) > 16000 * 8:
            c0 = (len(audio) - 16000 * 8) // 2
            audio = audio[c0:c0 + 16000 * 8]
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-4:
            embs.append(None)
            continue
        audio = np.clip(audio * (0.1 / rms), -1.0, 1.0)
        e = np.asarray(ere.generate(input=audio, fs=16000, extract_embedding=True)[0]["spk_embedding"],
                       dtype=np.float32).ravel()
        embs.append(e / (np.linalg.norm(e) + 1e-9))
    valid = [e for e in embs if e is not None]
    if len(valid) < 2:
        return {"ok": False, "error": "valid segments < 2", "sims": []}
    c = np.mean(valid, axis=0)
    c /= np.linalg.norm(c) + 1e-9
    sims = [round(float(e @ c), 4) if e is not None else None for e in embs]
    return {"ok": True, "sims": sims, "centroid_sim": round(float(np.mean(
        [float(e @ c) for e in valid])), 4)}




@app.post("/speaker_replace")
async def speaker_replace(request: _Request, name: str = _Query(..., min_length=1, max_length=32)):
    """用户人工精修后的"合成回库"（2026-10-09）：SEGS 打包的音频片整批替换某人的注册段。

    与 /speaker_enroll 的区别：enroll 走质量过滤+一致性选优（自动裁决）；
    replace 是"用户已逐片听过并勾选"的信任写入——不再票出任何一片，
    每片直接落为 {name}_{ts}_{i}.pcm 并重建质心缓存。
    片数 1~20、单片 ≥1s（前端已按 VAD 切分，这里是兜底校验）。
    """
    import re as _re3
    nm = name.strip()
    if not _re3.fullmatch(r"[\w\u4e00-\u9fff\-]+", nm):
        return {"ok": False, "error": "姓名只允许中文/英文/数字/下划线/横线"}
    raw = await request.body()
    parts = []
    i = 0
    while i + 8 <= len(raw):
        if raw[i:i + 4] != b"SEGS":
            return {"ok": False, "error": f"bad segment header at {i}"}
        n = int.from_bytes(raw[i + 4:i + 8], "big")
        i += 8
        if i + n > len(raw):
            return {"ok": False, "error": "length overflow"}
        parts.append(raw[i:i + n])
        i += n
    if not (1 <= len(parts) <= 100):
        return {"ok": False, "error": f"片数 {len(parts)} 超界（1~100）"}
    parts = [p[:len(p) // 2 * 2] for p in parts]  # 奇数字节截齐（同 segments_sim 事故防御）
    if any(len(p) < 16000 * 1 for p in parts):
        return {"ok": False, "error": "存在 <1s 的片，请重新切分"}
    d = Path(_SPEAKER_ENROLL_DIR)
    # 先删旧段
    for p in d.glob("*.pcm"):
        if _name_of_stem(p.stem) == nm:
            p.unlink(missing_ok=True)
    ts = int(_time.time())
    total = 0.0
    for j, pcm in enumerate(parts):
        (d / f"{nm}_{ts}_{j}.pcm").write_bytes(pcm)
        total += len(pcm) / 2 / 16000.0
    return {"ok": True, "name": nm, "segments": len(parts),
            "effective_sec": round(total, 1), "speakers": _enroll_names()}


# ============ /enroll_debug：单质心 vs 多向量代表集 对照（2026-10-12，只读） ============
def _load_db_vectors(enroll_dir: str | None = None) -> dict[str, np.ndarray]:
    """每人逐段 embedding 矩阵（N,192），L2 归一——multi-vector 模式用。"""
    import glob as _glob
    d = Path(enroll_dir or _SPEAKER_ENROLL_DIR)
    if not d.is_dir():
        return {}
    ere = _ere()
    out: dict[str, np.ndarray] = {}
    by_name: dict[str, list[Path]] = {}
    for pth in sorted(d.iterdir()):
        if pth.name.endswith(".pcm"):
            by_name.setdefault(_name_of_stem(pth.stem), []).append(pth)
    for name, paths in by_name.items():
        tot_sec = sum(p.stat().st_size for p in paths) / 32000.0
        if len(paths) < 3 and tot_sec < 8.0:
            continue
        vecs = []
        for p in paths:
            try:
                pcm = np.frombuffer(p.read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
                if len(pcm) < 16000 * 1.5:
                    continue
                _r = float(np.sqrt(np.mean(pcm ** 2)))
                if _r > 1e-6:
                    pcm = np.clip(pcm * (0.1 / _r), -1.0, 1.0)
                e = ere.generate(input=pcm, fs=16000, extract_embedding=True)[0]["spk_embedding"]
                e = np.asarray(e, dtype=np.float32).ravel()
                n = np.linalg.norm(e)
                if n < 1e-9:
                    continue
                vecs.append(e / n)
            except Exception:
                continue
        if vecs:
            out[name] = np.stack(vecs)
    return out


@app.post("/enroll_debug")
async def enroll_debug(name: str = Form(...), top: int = Form(default=10)):
    """对照：name 的注册段在 单质心(旧) vs 多向量max(新) 下的自相关 + 库内 Top 对比。

    目的：验证"线上/线下混合注册时，算术平均质心把双峰揉烂导致自相关下降"
    是否被 multi-vector（每段一个代表，匹配取max）缓解。只读，不改库。
    """
    vecs_db = _load_db_vectors()
    if name not in vecs_db:
        return {"error": f"{name} 不在声纹库或无有效段", "have": sorted(vecs_db.keys())}

    mat = vecs_db[name]                    # (N,192) 该人全部注册段
    centroid = mat.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9   # 旧模式单质心

    # 此人自相关：每段 vs 旧质心 / vs 其余段max(留一)
    old_self = mat @ centroid
    new_self = []
    for i in range(len(mat)):
        rest = np.delete(mat, i, axis=0)
        new_self.append(float(np.max(rest @ mat[i])))
    new_self = np.array(new_self)

    # 库内竞争 top：段embedding 对所有人（旧质心 / 新max-over-reps）
    old_centroids = {n: v.mean(axis=0) / (np.linalg.norm(v.mean(axis=0)) + 1e-9)
                     for n, v in vecs_db.items()}
    rows = []
    for i in range(len(mat)):
        r_old = {n: float(mat[i] @ c) for n, c in old_centroids.items()}
        r_new = {n: float(np.max(v @ mat[i])) for n, v in vecs_db.items()}
        t_old = sorted(r_old.items(), key=lambda x: -x[1])[:top]
        t_new = sorted(r_new.items(), key=lambda x: -x[1])[:top]
        rows.append({
            "seg": i,
            "old_self": round(float(old_self[i]), 4),
            "new_self": round(float(new_self[i]), 4),
            "old_top": [[n, round(s, 4)] for n, s in t_old],
            "new_top": [[n, round(s, 4)] for n, s in t_new],
        })

    return {
        "name": name,
        "n_segs": len(mat),
        "mean_old_self": round(float(old_self.mean()), 4),
        "mean_new_self": round(float(new_self.mean()), 4),
        "min_old_self": round(float(old_self.min()), 4),
        "min_new_self": round(float(new_self.min()), 4),
        "segs": rows,
    }
