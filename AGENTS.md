# AGENTS.md — ASR-service（GPU diarize_svc）

> GPU 端服务仓库。全系统视角的知识地图在 **ASR-fullstack 仓库的 `KNOWLEDGE.md`**，两边共用；本文只写 service 视角的补充。
> 部署/启动：`~/workspace/asr/DEPLOY.md`（gpu2）。改代码安全流程、红线、排障速查见 fullstack 的 AGENTS.md 第 2/4/5 节，同样适用。

## 本仓库结构

- `diarize_svc/service.py`：**全部服务逻辑在这个文件**（~1900 行，改前备份+ast 验语法）
- `diarize_svc/gtcrn_net.py` + `gtcrn_dns3.tar`：GTCRN 降噪
- `third_party_3dspeaker/`：3D-Speaker speakerlab 包（**import 它需要 pyannote 全家桶，缺了 /diarize_full 直接 500**）
- `speaker_enrolls/`：声纹库 pcm（生产数据，别乱动）

## 端点清单（对照 KNOWLEDGE.md 第 3 节读）

`/health` `/denoise` `/vad` `/diarize`（带oracle搜索） `/diarize_full`（会末整场聚类） `/identify_segments`（认人+ASR 一体） `/meeting_chunk`（旧，弃用） `/embed` `/segments_sim` `/speaker_identify` `/speaker_enroll` `/speaker_replace` `/speaker_rename` `/speaker_segment_delete` `/speaker_segments` `/speaker_segment_audio` `/speaker_names` `/enroll_debug`（只读对照：单质心 vs multi-vector）

## 关键实现位置

| 逻辑 | 位置 |
|---|---|
| 声纹库加载（单质心+增量缓存） | `_load_db_multi()` |
| 簇级认人 argmax（**待改段投票**） | `/identify_segments` 内"簇级认人"段 |
| 聚类参数 | `_diar3ds()`（spectral, mer_cos=0.8, min_cluster_size=4） |
| 嵌入模型 | `_ere()` ERes2NetV2（认人与会末 ere 路径同模型；**ere 聚类路径有 bug，生产 cam**） |
| ASR | `_asr()` SeACo Paraformer + `_punc()` + `_itn()` |

## 待实现（已拍板）

1. `/identify_segments` 簇级 argmax → **簇内段投票**（治混合簇错标，如 93% 李炳晨标成刘恒）
2. `_load_db_multi` 单质心 → **multi-vector**（每人逐段向量，匹配取 max；治跨设备自相关崩；对照端点 `/enroll_debug` 已就绪）
3. 实现后记 `eval/tuning_log.md` 并同步 fullstack 的 KNOWLEDGE.md
