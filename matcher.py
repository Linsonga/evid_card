import numpy as np
import pymysql
from sklearn.metrics.pairwise import cosine_similarity
from config import DB_CONFIG
from utils import vector_4b
from logger import logger
from database import get_db_connection
from utils import get_cached_vector

class QwenEmbeddingMatcher:
    def __init__(self):
        logger.info("正在初始化向量匹配器...")
        self.topic_ids = []
        self.topic_titles = []
        self.topic_descs = []
        self.desc_embeddings = np.array([])
        self.topic_cards_cache = {}
        self.load_topic_data()

    def load_topic_data(self):
        # 🌟 修改点：使用连接池
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        try:
            cursor.execute("SELECT id, title, `describe` FROM evidence_topic where id != 1")
            topic_results = cursor.fetchall()
        finally:
            # 🌟 修改点：确保归还连接
            cursor.close()
            conn.close()

        self.topic_ids = [row["id"] for row in topic_results]
        self.topic_titles = [row["title"] for row in topic_results]
        self.topic_descs = [row["describe"] for row in topic_results]

        # logger.info(f"成功加载 {len(self.topic_ids)} 个专区记录。正在向量化专区描述...")
        desc_embeddings_list = []
        for i in range(len(self.topic_ids)):
            desc_text = self.topic_descs[i] if self.topic_descs[i] else "暂无描述"
            desc_embeddings_list.append(vector_4b(desc_text))

        self.desc_embeddings = np.array(desc_embeddings_list)
        cursor.close()
        conn.close()
        # logger.info("专区向量初始化完成！启动速度大幅提升")

    def cardDistribution(self, title_vec, card_title, threshold=0.35, allowed_topic_ids=None):
        if not self.topic_descs or len(self.desc_embeddings) == 0: return None
        if not title_vec: return None

        # ================= 新增逻辑：过滤指定专区 =================
        if allowed_topic_ids is not None:
            # 找到 allowed_topic_ids 在当前内存列表中的索引
            allowed_indices = [i for i, tid in enumerate(self.topic_ids) if tid in allowed_topic_ids]

            if not allowed_indices:
                print(f"⚠️ 匹配失败：指定的专区 IDs {allowed_topic_ids} 不在已加载的缓存库中。")
                return None

            # 动态切片，只取这几个专区的向量和信息
            sub_embeddings = self.desc_embeddings[allowed_indices]
            sub_ids = [self.topic_ids[i] for i in allowed_indices]
            sub_titles = [self.topic_titles[i] for i in allowed_indices]
        else:
            # 保持原有逻辑：全量匹配
            sub_embeddings = self.desc_embeddings
            sub_ids = self.topic_ids
            sub_titles = self.topic_titles
        # ========================================================

        title_embedding = np.array([title_vec])
        # 与过滤后的子集计算相似度
        similarities = cosine_similarity(title_embedding, sub_embeddings)[0]
        max_idx = np.argmax(similarities)
        max_score = float(similarities[max_idx])

        if max_score >= threshold:
            best_topic_id = sub_ids[max_idx]
            best_topic_name = sub_titles[max_idx]
            print(f"专区匹配成功: 【{card_title}】 -> 【{best_topic_name}】")
            # 🌟 改动：返回 tuple(ID, 名称)
            return best_topic_id, best_topic_name

        print(f"专区匹配失败: 【{card_title}】 最高匹配分仅 {max_score:.4f}。")
        # 🌟 改动：返回双空值
        return None, None


    def filterExistingTitleInTopic(self, new_title, new_vec, topic_id, threshold=0.85):
        if not new_vec: return False
        if topic_id not in self.topic_cards_cache:
            self._load_cards_for_topic(topic_id)

        cache = self.topic_cards_cache[topic_id]
        if len(cache["embeddings"]) == 0:
            print(f"embeddings:0 当前card中没有对应的专区")
            return False

        new_emb = np.array([new_vec])
        similarities = cosine_similarity(new_emb, cache["embeddings"])[0]
        max_idx = np.argmax(similarities)
        max_score = similarities[max_idx]

        if max_score >= threshold:
            exist_title = cache["titles"][max_idx]
            logger.warning(f"同专区去重拦截: 【{new_title}】与已有卡片【{exist_title}】高度相似 ({max_score:.4f})，跳过。")
            return True
        return False

    def _load_cards_for_topic(self, topic_id):
        # 🌟 修改点：使用连接池
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute("SELECT title FROM evidence_card WHERE topic_id = %s", (topic_id,))
            results = cursor.fetchall()
        finally:
            # 🌟 修改点：确保归还连接
            cursor.close()
            conn.close()

        titles = [row["title"] for row in results]

        valid_titles = []
        embeddings_list = []

        # 2. 🌟 替换掉原来的双重推导式，改用带缓存的函数
        for t in titles:
            vec = get_cached_vector(t)
            if vec:
                valid_titles.append(t)
                embeddings_list.append(vec)

        # 3. 更新内存属性
        self.topic_cards_cache[topic_id] = {
            "titles": valid_titles,
            "embeddings": np.array(embeddings_list) if embeddings_list else np.array([])
        }

        logger.info(f"Topic_ID={topic_id} 局部卡片加载完成 (共 {len(valid_titles)} 条)。")

    def add_card_to_cache(self, topic_id, card_title, card_vec):
        if topic_id not in self.topic_cards_cache:
            self.topic_cards_cache[topic_id] = {"titles": [], "embeddings": np.array([])}

        cache = self.topic_cards_cache[topic_id]
        cache["titles"].append(card_title)
        new_emb = np.array([card_vec])
        if len(cache["embeddings"]) == 0:
            cache["embeddings"] = new_emb
        else:
            cache["embeddings"] = np.vstack((cache["embeddings"], new_emb))
        logger.info(f"缓存更新: 卡片【{card_title}】已动态加入 Topic_ID={topic_id} 局部防重库。")