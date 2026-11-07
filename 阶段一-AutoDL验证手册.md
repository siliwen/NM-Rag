# 阶段一：AutoDL 验证手册（LiveTalking 实时渲染）

> 目标：在租用的云 GPU 上跑通「形象 → 实时口型驱动 → 网页对话」全链路，并记录 RTP 指标。
> 预算：RTX 4090 约 ¥1.98/小时，预计 6~10 小时。
> 约定：每步跑完做"验证"，任何一步报错，**终端截图发我**，不要自己重试超过两次。

---

## 第 0 步：开机

1. 注册/登录 AutoDL（autodl.com），充值 50 元
2. 算力市场选 **RTX 4090（24GB）**，地区任选（西北区较便宜）
3. 镜像选择：框架 → **PyTorch 2.9.1 / Python 3.12 / CUDA 12.8**（若有此组合的基础镜像直接用）
4. 创建实例，等状态变为"运行中"

## 第 1 步：进入终端并克隆项目

JupyterLab 页面打开终端，逐条执行：

```bash
# AutoDL 学术加速（下载 GitHub / HuggingFace 提速）
source /etc/network_turbo

git clone https://github.com/lipku/LiveTalking.git
cd LiveTalking
```

**验证**：`ls` 能看到 `app.py`、`requirements.txt`、`Dockerfile`。

## 第 2 步：装依赖

```bash
# AutoDL 自带 conda，直接建环境
conda create -n lt python=3.12 -y
conda activate lt

# 国内源安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证**：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
预期输出类似 `2.9.1+cu128 True`。

## 第 3 步：下载模型权重

LiveTalking 官方模型走网盘分发（README 提供夸克云盘 / Google Drive 链接）：

1. 在**你本机**浏览器打开 README 里的夸克云盘链接，下载：
   - `wav2lip256.pth`（主模型）
   - `wav2lip256_avatar1.tar.gz`（示例形象）
2. 上传到 AutoDL 两种方式任选：
   - **JupyterLab 直接拖拽上传**（文件大时较慢）
   - 或本机命令行 scp（AutoDL 复制登录指令里能看到端口和主机）：

```bash
scp -P <端口> wav2lip256.pth root@<主机>:/root/LiveTalking/models/
scp -P <端口> wav2lip256_avatar1.tar.gz root@<主机>:/root/LiveTalking/data/avatars/
```

3. 在 AutoDL 终端里：

```bash
cd /root/LiveTalking
# 主模型改名
mv models/wav2lip256.pth models/wav2lip.pth
# 示例形象解压
cd data/avatars && tar -xzf wav2lip256_avatar1.tar.gz && cd ../..
```

**验证**：`ls models/wav2lip.pth`、`ls data/avatars/` 能看到 avatar1 文件夹。

## 第 4 步：启动服务

```bash
conda activate lt
cd /root/LiveTalking
python app.py --model wav2lip --transport webrtc
```

> AutoDL 需要在「自定义服务」里开放端口（默认 6006/8010 映射），按页面提示获取外网访问地址。

**验证**：浏览器打开映射出的地址 + `/index.html`（例如 `https://xxx.autodl.com/proxy/8010/index.html`），页面出现数字人画面。

## 第 5 步：跑通对话

1. 页面文本框输入任意中文（如"大家好，欢迎了解皮影戏"），点发送
2. 观察数字人是否实时开口、能否被打断（说话中再发一条）
3. **记录推理终端日志**里的 chunk 行，计算 RTP：

```
RTP = elapsed / (帧数 / fps)
RTP < 1 → 实时有余量 ✅
```

4. 截图：①网页画面 ②终端 RTP 日志，发我

## 第 6 步：上传我们自己的形象素材（A 路线预演）

1. 页面打开 `/avatar.html`，上传一段 1~3 分钟正面口播视频
2. 等待形象生成任务完成（页面有进度）
3. 回 `/index.html` 选择新形象，重复第 5 步

**验证**：新形象能实时说话。到此，A 路线的最小闭环已验证 ✅

## 第 7 步（可选对照）：CyberVerse 跑分

```bash
# 另开一台 5090 实例（¥2.88/时），AutoDL 应用市场搜 CyberVerse 官方镜像直接启动
# config/cyberverse.yaml 中 inference.avatar.enabled: true
# 上传同一张照片，录制 30 秒对话视频
```

产出：两条同文案的对比视频，你目视拍板。

---

## 常见故障速查

| 现象 | 处理 |
|---|---|
| `CUDA out of memory` | 换 `--model wav2lip`（比 musetalk 轻）；确认没同时跑别的进程 |
| 页面打不开 | 检查 AutoDL 端口映射；确认 app.py 启动日志无报错 |
| 画面卡顿、日志 RTP > 1 | 降分辨率 / 确认用的是 4090 而非共享卡 |
| 网盘链接失效 | 告诉我，换 Google Drive 链接或我去找镜像源 |
| pip 装包超时 | 重试前先 `source /etc/network_turbo` |
