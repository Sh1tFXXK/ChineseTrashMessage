import os  # 导入操作系统接口模块，用于处理路径和环境变量
import matplotlib  # 导入绘图库基类

# 【关键修复：强制指定后端】
# 'Agg' 是一个非交互式后端，只负责将图表渲染到文件，而不尝试打开窗口。
# 这能彻底解决在 Windows/Conda 环境下运行绘图代码时，程序卡死或静默崩溃的问题。
matplotlib.use('Agg')  # 设置 matplotlib 使用 Agg 后端，确保在无界面环境下稳定保存图片
import matplotlib.pyplot as plt  # 导入绘图终端接口

# 解决 Intel MKL 库在特定环境下多次初始化的冲突问题，防止程序报错退出
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 强制允许 OpenMP 运行时库重复初始化

import pandas as pd  # 导入数据分析库，用于处理表格数据
import numpy as np  # 导入数值计算库，用于矩阵运算
import re  # 导入正则表达式模块，用于文本清洗
import jieba  # 导入中文分词库，用于切分中文句子
import joblib  # 导入模型保存库，用于序列化 Python 对象
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入 TF-IDF 向量化工具
from sklearn.naive_bayes import ComplementNB  # 导入专门处理类别不平衡的朴素贝叶斯模型
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.svm import SVC  # 使用支持概率输出的标准支持向量机 (SVC)
from sklearn.ensemble import VotingClassifier  # 导入投票分类器集成工具
from sklearn.preprocessing import MaxAbsScaler  # 导入最大绝对值缩放器，适合稀疏数据
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 导入评估指标
from scipy.sparse import hstack, csr_matrix  # 导入稀疏矩阵合并工具
import seaborn as sns  # 导入 Seaborn 统计绘图库
from wordcloud import WordCloud  # 导入词云生成库
import warnings  # 导入警告控制模块

# 忽略训练过程中的不重要警告（如迭代未收敛警告等）
warnings.filterwarnings('ignore')  # 设置警告过滤器为忽略

# --- 可视化全局配置 ---
# 设置中文字体为黑体 (SimHei)，防止生成的图表中的中文显示为方框乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴中负号 (-) 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 设置绘图背景风格为白格，并同步指定字体配置
sns.set_style('whitegrid', {'font.sans-serif': ['SimHei']})

# 定义数据集的本地绝对路径字典
file_urls = {
    "train": r"C:\Users\Administrator\PycharmProjects\JupyterProject\data\train.csv",  # 训练集路径
    "validation": r"C:\Users\Administrator\PycharmProjects\JupyterProject\data\dev.csv",  # 验证集路径
    "test": r"C:\Users\Administrator\PycharmProjects\JupyterProject\data\test.csv"  # 测试集路径
}


def tokenize(text, stopwords):
    """
    文本分词与清洗模块：
    1. 正则清洗：通过 [^\u4e00-\u9fff0-9a-zA-Z] 过滤掉所有标点符号、特殊表情和杂质。
    2. 深度分词：调用 jieba 库将连续的中文切分为符合语义的词组。
    3. 停用词过滤：剔除“的”、“了”、“是”等高频但无实际分类意义的词汇。
    """
    # 使用正则表达式将非中文字符、非数字、非英文字母替换为空格
    text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z]', ' ', str(text))
    # 使用 jieba 进行中文精确模式分词
    words = jieba.cut(text)
    # 过滤掉停用词表中的词汇以及纯空白字符，并用空格拼接
    return " ".join([w for w in words if w not in stopwords and w.strip()])


def get_stats(texts):
    """
    统计特征提取模块：
    除了文字语义，垃圾短信在形态上也有显著特征。
    1. 文本长度：垃圾短信通常为了包含更多诱导内容，长度偏长。
    2. 数字密度：垃圾短信常含有电话、QQ号、金额或日期。
    3. 关键词触发：统计‘红包’、‘链接’、‘加微’等强诱导性词汇的出现次数。
    """
    res = []  # 初始化统计结果列表
    # 定义垃圾短信常见的人工识别关键词库
    keywords = ['免费', '红包', '链接', '加微', '退订', '中奖', '积分', '回T']
    for t in texts:  # 遍历每一条短信文本
        t = str(t)  # 强制转为字符串防止空值报错
        # 计算：1.总长度；2.数字串出现的频次；3.关键词库命中的总次数
        res.append([len(t), len(re.findall(r'\d+', t)), sum(t.count(w) for w in keywords)])
    return np.array(res)  # 返回 NumPy 数组格式的统计特征矩阵


def main_process():
    # --- 1. 数据加载 ---
    # 分别从本地路径读取 CSV 格式的数据集文件
    df_train = pd.read_csv(file_urls["train"])  # 加载训练数据
    df_val = pd.read_csv(file_urls["validation"])  # 加载开发/验证数据
    df_test = pd.read_csv(file_urls["test"])  # 加载原始测试数据

    # 将训练集和验证集合并，利用更多数据训练模型以提升鲁棒性
    full_train_df = pd.concat([df_train, df_val], ignore_index=True)

    # 提取文本列，填充缺失值为字符并转为 Python 列表格式
    train_texts = full_train_df['text'].fillna("").astype(str).tolist()
    # 提取训练标签列
    train_labels = full_train_df['label'].tolist()
    # 提取测试文本列
    test_texts = df_test['text'].fillna("").astype(str).tolist()
    # 提取测试标签列
    test_labels = df_test['label'].tolist()

    # --- 2. 特征工程 (Feature Engineering) ---
    print("正在提取特征并分词...")  # 打印控制台进度提示
    # 定义基础停用词过滤集合
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一'}
    # 对训练集文本进行清洗和分词处理
    train_cleaned = [tokenize(t, stopwords) for t in train_texts]
    # 对测试集文本进行同步清洗和分词处理
    test_cleaned = [tokenize(t, stopwords) for t in test_texts]

    # 特征A：词袋特征 (TF-IDF Word) - 设置 1-2 词组组合，取前 8000 个最显著特征
    tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    X_train_word = tfidf_word.fit_transform(train_cleaned)  # 学习并转换训练文本
    X_test_word = tfidf_word.transform(test_cleaned)  # 转换测试文本

    # 特征B：字符特征 (TF-IDF Char) - 按字符切割 2-4 组合，前 4000 特征，适合发现干扰变体词
    tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=4000, sublinear_tf=True)
    X_train_char = tfidf_char.fit_transform(train_cleaned)  # 学习并转换训练字符特征
    X_test_char = tfidf_char.transform(test_cleaned)  # 转换测试字符特征

    # 特征C：形态特征 (Stats) - 对长度、数字等统计值进行最大绝对值缩放（保持稀疏性）
    X_train_stats = MaxAbsScaler().fit_transform(get_stats(train_texts))
    # 使用训练集的缩放标准来转换测试集的统计特征，保持一致性
    X_test_stats = MaxAbsScaler().fit(get_stats(train_texts)).transform(get_stats(test_texts))
    # 实例化并保存缩放器，以便在部署应用中对单条输入进行处理
    scaler = MaxAbsScaler().fit(get_stats(train_texts))

    # 混合特征组合：将词 TF-IDF、字 TF-IDF、统计特征横向拼接为超大特征向量
    X_train = hstack([X_train_word, X_train_char, csr_matrix(X_train_stats)])
    X_test = hstack([X_test_word, X_test_char, csr_matrix(X_test_stats)])

    # --- 3. 集成模型设计 (Ensemble Modeling) ---
    print("开始训练 (软投票模式，计算概率中)...")  # 打印进度
    # 软投票原理：将多个分类器的概率输出取均值/加权平均，作为最终判定依据。
    ensemble = VotingClassifier(
        estimators=[
            # 逻辑回归：擅长捕捉全局词频分布，C=3.0 增加对重要特征的关注度
            ('lr', LogisticRegression(C=3.0, max_iter=1000, class_weight='balanced')),
            # 支持向量机：寻找最佳超平面，probability=True 开启概率预测，耗时但精确
            ('svc', SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')),
            # 补集朴素贝叶斯：针对非平衡语料优化的朴素贝叶斯，alpha=0.1 减弱平滑
            ('cnb', ComplementNB(alpha=0.1))
        ],
        voting='soft'  # 开启软投票：模型将返回概率百分比而非硬类别标签
    )
    ensemble.fit(X_train, train_labels)  # 使用拼接后的综合特征矩阵训练集成模型

    # --- 4. 性能评估 ---
    y_pred = ensemble.predict(X_test)  # 在测试集上生成预测结果
    print(f"\n🏆 最终测试集准确率: {accuracy_score(test_labels, y_pred):.4f}")  # 打印准确率
    # 打印详细的精确率 (P)、召回率 (R) 和 F1 值评估报告
    print(classification_report(test_labels, y_pred, target_names=['正常', '垃圾']))

    # --- 5. 模型持久化 ---
    print("正在保存模型和参数...")  # 打印提示
    joblib.dump(ensemble, 'spam_model.pkl')  # 保存集成模型对象
    joblib.dump(tfidf_word, 'tfidf_word.pkl')  # 保存词特征向量化器
    joblib.dump(tfidf_char, 'tfidf_char.pkl')  # 保存字特征向量化器
    joblib.dump(scaler, 'scaler.pkl')  # 保存数值缩放器

    # --- 6. 自动化可视化分析 ---
    print("正在生成并写入可视化图表...")  # 打印提示
    plt.figure(figsize=(18, 12))  # 创建一个宽 18 英寸、高 12 英寸的大画布
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # 调整子图间的垂直和水平间距

    # 6.1 混淆矩阵：用于观察哪些正常短信被误判成了垃圾（误报率）
    ax1 = plt.subplot(2, 2, 1)  # 定位到 2x2 画布的第 1 个位置
    cm = confusion_matrix(test_labels, y_pred)  # 计算混淆矩阵数值
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # 绘制热力图并显示数值
                xticklabels=['正常', '垃圾'], yticklabels=['正常', '垃圾'], ax=ax1)
    ax1.set_title('模型混淆矩阵 (Confusion Matrix)')  # 设置标题

    # 6.2 词云图：展示垃圾短信中出现频率最高的诱导性敏感词
    font_path = r"C:\Windows\Fonts\simhei.ttf"  # 指定系统字体路径以支持词云中文
    # 筛选出被标记为垃圾短信的文本并拼接成一个巨型字符串
    spam_text = " ".join([test_cleaned[i] for i in range(len(test_labels)) if test_labels[i] == 1])
    ax2 = plt.subplot(2, 2, 2)  # 定位到画布第 2 个位置
    if spam_text.strip():  # 如果垃圾文本不为空
        # 生成词云对象，设置背景色、宽高及字体
        wc = WordCloud(font_path=font_path, background_color='white', width=400, height=300).generate(spam_text)
        ax2.imshow(wc, interpolation='bilinear')  # 渲染词云图片
    ax2.axis('off')  # 关闭词云图的坐标轴显示
    ax2.set_title('垃圾短信高频关键词云')  # 设置标题

    # 6.3 预测概率分布图：展示集成模型对分类判定的信心分布（越靠近 0 或 1 越自信）
    ax3 = plt.subplot(2, 2, 3)  # 定位到画布第 3 个位置
    y_proba = ensemble.predict_proba(X_test)[:, 1]  # 获取判定为“垃圾短信”的概率值
    sns.histplot(y_proba, bins=20, kde=True, color='red', ax=ax3)  # 绘制带核密度估计的直方图
    ax3.set_title('预测概率分布 (置信度分析)')  # 设置标题
    ax3.set_xlabel('判定为“垃圾”的置信得分')  # 设置横轴标签

    # 6.4 长度对比图：直观展示垃圾短信与正常短信在文本长度上的分布差异
    ax4 = plt.subplot(2, 2, 4)  # 定位到画布第 4 个位置
    # 创建临时数据帧用于绘图
    df_plot = pd.DataFrame({'类别': [('垃圾' if l == 1 else '正常') for l in test_labels],
                            '长度': [len(t) for t in test_texts]})
    # 绘制箱线图，过滤掉长度超过 300 的异常长文本以便观察核心区间
    sns.boxplot(x='类别', y='长度', data=df_plot[df_plot['长度'] < 300], palette="Set2", ax=ax4)
    ax4.set_title('短信长度分布对比箱线图')  # 设置标题

    # --- 7. 图片强制保存 ---
    save_path = os.path.join(os.getcwd(), 'model_results.png')  # 获取当前脚本所在目录并定义图片文件名
    # 将画布保存为 PNG 图片，设置分辨率、紧凑布局及白色背景
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')  # 关闭内存中所有的绘图对象，防止内存泄漏

    print(f"✅ 可视化结果已保存至: {save_path}")  # 控制台打印完成路径提示


if __name__ == '__main__':
    main_process()  # 如果是直接运行此脚本，则启动主流程函数