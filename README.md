# ChineseTrashMessage

项目概述
---
ChineseTrashMessage 是一个用于识别与分类中文垃圾短信（spam）的轻量项目。仓库包含训练脚本（pro.py）用于训练模型并导出序列化文件（`.pkl`），以及用于在线交互演示的 Streamlit 应用（app.py）。训练结果会生成可视化图表（model_results.png）和模型/向量化器文件（spam_model.pkl、tfidf_word.pkl、tfidf_char.pkl、scaler.pkl），可直接用于部署或二次开发。

主要目标
---
- 对中文短信进行自动化分类：正常 / 垃圾（诈骗、广告等）。
- 提供训练、评估、可视化与在线检测的端到端流水线。
- 简单易用：训练脚本一键运行，Streamlit 提供在线交互界面。

仓库结构（重要文件）
---
- `pro.py` — 训练/评估/导出脚本  
  - 完整的预处理、特征工程（词/字符 TF-IDF + 统计特征）、集成模型（VotingClassifier：LogisticRegression + SVC + ComplementNB）、评估与可视化。运行后会导出：`spam_model.pkl`、`tfidf_word.pkl`、`tfidf_char.pkl`、`scaler.pkl`、`model_results.png`。
  - 注意：脚本中默认的训练/验证/测试 CSV 路径是 Windows 下的绝对路径（在 `file_urls` 中），在使用前请按实际情况修改或放置相应数据文件。
- `app.py` — Streamlit Web 应用（在线检测界面）  
  - 加载 pro.py 导出的模型与向量化器，提供文本输入区、置信度与训练结果展示（显示 `model_results.png`）。
  - 含与 pro.py 相同的预处理逻辑（确保训练与推理时处理方式一致）。
- `tfidf_word.pkl`, `tfidf_char.pkl`, `scaler.pkl`, `spam_model.pkl` — 序列化的模型与预处理器（用于部署/推理）
- `model_results.png` — 训练后自动生成的可视化图表
- `data/` — 数据目录（仓库示例数据或用户自放的数据）

依赖（建议）
---
主要依赖（根据代码 import 列表）：
- Python 3.8+
- numpy
- pandas
- scikit-learn
- scipy
- joblib
- jieba
- matplotlib
- seaborn
- wordcloud
- streamlit

安装示例：
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install numpy pandas scikit-learn scipy joblib jieba matplotlib seaborn wordcloud streamlit
```

快速开始 — 训练模型（本地）
---
1. 准备数据（CSV，必须包含列 `text` 与 `label`；label: 0=正常，1=垃圾）  
2. 修改 `pro.py` 中的 `file_urls` 字典，指向你的训练 / 验证 / 测试文件路径，或把数据放到脚本所指的默认路径下。示例字段：
```python
file_urls = {
    "train": r"C:\path\to\data\train.csv",
    "validation": r"C:\path\to\data\dev.csv",
    "test": r"C:\path\to\data\test.csv"
}
```
3. 运行训练脚本：
```bash
python pro.py
```
运行后将依次执行：分词与特征提取 → 训练 VotingClassifier → 在测试集上评估 → 保存模型与向量化器 → 生成并保存 `model_results.png`。

注意与常见问题（训练）
- pro.py 在脚本开头设置了 matplotlib 后端为 `Agg`，适用于无界面/服务器环境；同时设定了 `KMP_DUPLICATE_LIB_OK=TRUE` 来规避部分环境下的多次初始化错误。
- 词云使用的字体路径在 pro.py 中用的是 Windows 常见路径 `C:\Windows\Fonts\simhei.ttf`，若在 Linux/Mac 或无该字体环境下运行，请修改为可用的中文字体路径或安装相应字体，否则词云可能无法显示中文。
- 若需要在训练时使用 GPU 或深度学习模型，可在此脚本的基础上替换模型训练部分。

快速开始 — 启动在线界面（Streamlit）
---
1. 确保 `spam_model.pkl` / `tfidf_word.pkl` / `tfidf_char.pkl` / `scaler.pkl` 与 `model_results.png` 位于与 `app.py` 同一目录，或在 app 中调整路径。  
2. 启动 Streamlit：
```bash
streamlit run app.py
```
3. 在打开的页面中：
- 左侧输入短信文本，点击 “开始识别” 会显示预测（正常 / 垃圾）与置信度（进度条与数值）。
- 右侧会显示模型训练的性能图表（若 `model_results.png` 存在）。

程序内部预处理（与生产部署一致性）
---
要保证训练与推理的一致性，app.py 中实现了与 pro.py 完全一致的预处理函数 `preprocess_input`，处理步骤包括：
- 正则清洗（保留中文、字母、数字）
- jieba 分词并移除停用词
- 词级与字符级 TF-IDF 向量化
- 统计特征（文本长度、数字数量、敏感关键词命中次数）
- 特征拼接（hstack）

示例：在其他 Python 代码中加载模型并进行单条预测
---
若你想在脚本中直接调用模型做预测（非 Streamlit），示例代码：
```python
import joblib
import jieba, re
import numpy as np
from scipy.sparse import hstack, csr_matrix

# 加载模型与预处理器
model = joblib.load('spam_model.pkl')
vec_word = joblib.load('tfidf_word.pkl')
vec_char = joblib.load('tfidf_char.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_single(text):
    stopwords = {'的','了','是','在','我','有','和','就','不','人','都','一'}
    clean_text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z]', ' ', str(text))
    words = " ".join([w for w in jieba.cut(clean_text) if w not in stopwords and w.strip()])
    x_w = vec_word.transform([words])
    x_c = vec_char.transform([words])
    keywords = ['免费','红包','链接','加微','退订','中奖','积分','回T']
    stats = np.array([[len(text), len(re.findall(r'\d+', text)), sum(text.count(w) for w in keywords)]])
    x_s = scaler.transform(stats)
    return hstack([x_w, x_c, csr_matrix(x_s)])

# 使用示例
text = "恭喜您中奖，点击链接领取红包。"
features = preprocess_single(text)
pred = model.predict(features)[0]
proba = model.predict_proba(features)[0]
print("label:", pred, "spam_prob:", proba[1])
```

部署建议
---
- 本地小规模服务：Streamlit 足以满足演示与低并发使用。  
- 生产级别：将加载的模型与预处理逻辑包装成 REST API（FastAPI/Flask + gunicorn/uvicorn），容器化（Docker），并做模型版本化与 CI/CD。  
- 注意隐私：短信可能包含个人敏感数据，上线前请做日志脱敏与访问权限控制，并遵守相关法规。

改进建议
---
- 将 `file_urls` 改为相对路径或通过命令行参数 / 配置文件指定，更适合多人/跨平台使用。  
- 增加 `requirements.txt`，并提供 `Dockerfile` 与 `docker-compose.yml` 以便快速部署。  
- 增加测试用例（pytest），并将 CI 集成到 GitHub Actions。  
- 将 Streamlit UI 改为 REST API（FastAPI）以便程序化调用。

许可与贡献
---
- 欢迎 Fork、Issue 与 PR。贡献时请确保：提供复现步骤、测试说明、和训练/推理影响的说明。

常见问题（快速提示）
---
- “运行 pro.py 后没有生成模型文件？”  
  - 检查 `file_urls` 的路径是否存在且 CSV 有 `text` 和 `label` 列；查看控制台输出以排查异常。  
- “Streamlit 页面没有显示模型图？”  
  - 确认 `model_results.png` 在 `app.py` 同级目录，或运行 `python pro.py` 先生成该图片。  
- “词云中文字显示为方块？”  
  - 修改 `pro.py` 中 `font_path` 为系统上可用的中文字体路径，或安装相应字体。
