# ASR-service 经验

## 架构
GPU 服务器 172.16.16.120，FastAPI :10096，承担所有重运算（降噪/聚类/声纹/ASR），本地零 torch。

## 端点
- `/health` 健康检查
- `/denoise` GTCRN 降噪
- `/diarize_full` 3D-Speaker 整场聚类（emb=cam|ere）
- `/identify_segments` 簇级声纹认人 + ASR
- `/segments_sim` 段级相似度
- `/speaker_enroll` 注册声纹
- `/speaker_replace` 信任写入
- `/speaker_rename` 重命名
- `/speaker_segment_delete` 删段
- `/enroll_debug` 只读对照（单质心 vs multi-vector）

## 踩坑
1. `_load_db_multi` 增量缓存按人签名，避免全量重建卡死
2. scp 到 GPU 后追加代码用 `cat >>`，容易重复追加，先 `grep -c` 确认
3. 改前备份，改后 `python -c "import ast"` 验语法
4. GPU 服务可能无故死掉，症状是 httpx.ConnectError，检查 10096 端口
5. 重启命令：`cd ~/jvkit/asr && setsid nohup venv/bin/python -m uvicorn diarize_svc.service:app --host 0.0.0.0 --port 10096 --app-dir ~/jvkit/asr > diarize_svc.log 2>&1 < /dev/null &`

## 关键参数
- `reannotate_name_threshold = 0.50`（生产用）
- `merge_thr = 0.55`（实时版 /meeting_chunk_3ds）
- `mer_cos = 0.8`（簇合并阈值）
- `min_cluster_size = 4`（小簇重指派）

## 已知问题
- 簇级 argmax 在混合簇上系统性出错（李炳晨→刘恒），段投票更鲁棒
- CAM++ 分不开近邻声纹对（冯忠俊↔蒋泽鹏 cos 0.85）
- ere 嵌入路径有稳定性 bug（37e3 分出 1 簇），已回退 cam
- `/diarize_full` 对长音频不稳定，可能服务崩掉
