# -*- coding: utf-8 -*-
"""
AIUserSession － PostgreSQL + pgvector 版本
-------------------------------------------------
已完全移除 SQLite，相依資料全部寫入 `tool_postgreSQL.AnalysisSummary`。
FAISS 仍用本地檔案索引（若多機部署請改放 S3 或 pgvector ANN）。
注意：在 Thread 內需用 `asyncio.run()` 建立臨時事件迴圈來執行 async 儲存流程。
"""
from __future__ import annotations
import asyncio
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import List
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from tool_async import tool_postgreSQL as db
from tool_async import tool_ai_chatbot_async as ai_chatbot
from tool_async import tool_ai_pet_prompt as prompt


load_dotenv()
# ------------------------------------
# AIUserSession
# ------------------------------------

class AIUserSession:
    """對單一 user 的對話、分析與資料存取管理"""
    def __init__(self, user_id: int, main_loop: asyncio.AbstractEventLoop | None = None, db_session: AsyncSession | None = None):
        self.main_loop = main_loop or asyncio.get_running_loop()
        self.user_id = user_id
        self.clientGPT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.message_analysis_threshold = 40
        self.temp_messages: List[dict] = []
        self.all_messages: List[dict] = []
        self.db_session = db_session  # FastAPI 會傳進來；背景腳本為 None

        # 專屬聊天機器人（短訊息分析）
        self.chatbot_short = ai_chatbot.AIChatbot(
            bot_name=str(user_id),
            ai_model="claude-3-7-sonnet-20250219",
        )
        self.chatbot_short.print_prompt = True
        self.chatbot_short.message_folder_path = f"./database/{user_id}/"



    # -------------------- 互動入口 --------------------
    
    async def handle_interaction(self, role: str, text: str):
        """外部（FastAPI route）呼叫此方法丟訊息進來"""
        # print(f"⚠️ 收到訊息：{role} - {text}")
        # print(f"⚠️ temp_messages：{self.temp_messages}")
        if not text.strip():
            return

        # 查詢模式
        if role == "search_user_data":
            return await self.search_similar_statements(     
                query=text,
                top_k=8,
                session=self.db_session
            )

        # 停止並分析
        if role == "stop_and_analysis" and len(self.temp_messages) > 1:
            await self.analyze_all()
            return

        # 確保第一句一定是 user
        if not self.temp_messages and role != "user":
            print("⚠️ 第一條訊息必須是 user，已忽略")
            return

        # 合併同角色連續訊息
        if self.temp_messages and self.temp_messages[-1]["role"] == role:
            self.temp_messages[-1]["content"] += text
            return

        self.temp_messages.append({"role": role, "content": text})
        self.all_messages.append({"role": role, "content": text})

        if len(self.temp_messages) > self.message_analysis_threshold:
            await self.analyze_all()


    async def analyze_all(self):
        """四種心理分析並行執行"""
        print("⚠️ 開始分析訊息")
        queries = [
            ("beliefs", *prompt.prompt_message_analysis_beliefs()),
            ("persona", *prompt.prompt_message_analysis_persona()),
            ("emotions", *prompt.prompt_message_analysis_emotions()),
            ("life_story", *prompt.prompt_message_analysis_life_story()),
        ]
        tasks = [self._run_analysis(query, params) for _, query, params in queries]
        await asyncio.gather(*tasks)

    async def _run_analysis(self, query: str, assistant_params: dict):
        print(f"⚠️ 開始分析：{query, assistant_params}")
        # 1. 將除最後一條 user 以外的 temp_messages 交給 Claude 分析
        last_user = None
        print(f"⚠️ temp_messages：{self.temp_messages}")
        if self.temp_messages and self.temp_messages[-1]["role"] == "user":
            last_user = self.temp_messages[-1]
            self.temp_messages.pop()
            print(f"⚠️ temp_messages[-1]：{last_user}")

        self.chatbot_short.read_previous_messages(messages_history=self.temp_messages)
        self.temp_messages = [last_user] if last_user else []

        # 2. 呼叫 LLM
        response = await self.chatbot_short.assistant_with_search_tool(**assistant_params)
        xml_block = ai_chatbot.extract_tagged_content_from_str(response, f"answer_{query}")
        _, df = parse_psychological_analysis(xml_block)

        # 3. 儲存
        await self._append_analysis(df, session=self.db_session)
        await self._batch_embed(batch_size=50, session=self.db_session)

        # 4. 清理聊天機器人對話
        await self.chatbot_short.save_messages_to_postgresql()
        self.chatbot_short.clear_all_messages()

    
    # ------------ 小工具：確保 session ------------
    @asynccontextmanager
    async def _ensure_session(self, ext=None):
        if ext or self.db_session:
            yield ext or self.db_session
        else:
            async with db.AsyncSessionLocal() as temp:
                yield temp

    # ------------ 1. 寫入分析 ------------
    async def _append_analysis(self, df: pd.DataFrame, session=None):
        print(f"⚠️ 儲存分析結果：{df}")
        async with self._ensure_session(session) as s:
            await db.insert_analyses(s, self.user_id, df.to_dict("records"))
            await s.commit()

    # ------------ 2. 批量向量化 ------------
    async def _batch_embed(self, batch_size=50, session=None):
        async with self._ensure_session(session) as s:
            rows = await db.select_unembedded(s, self.user_id)
            print(f"⚠️ 需要向量化的筆數：{len(rows)}", rows)
            if not rows:
                return
            for chunk in [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]:
                texts = [f"{r.statement} || {r.evidence}" for r in chunk]
                resp  = await self.clientGPT.embeddings.create(model=self.embedding_model, input=texts)
                pair  = [(r.id, v.embedding) for r, v in zip(chunk, resp.data)]
                await db.update_vectors(s, pair)
                await s.commit()

    # ------------ 3. 相似度查詢（改走 pgvector）------------
    async def search_similar_statements(self, query: str, top_k=5, session=None):
        # 取查詢向量
        emb = await self.clientGPT.embeddings.create(model=self.embedding_model, input=query)
        q_vec = emb.data[0].embedding

        # 查 pgvector
        async with self._ensure_session(session) as s:
            rows = await db.pgvector_search(s, self.user_id, q_vec, top_k)

        # 組裝
        results = [
            dict(
                analysis_id=r.id,
                category=r.category,
                subcategory=r.subcategory,
                statement=r.statement,
                evidence=r.evidence,
                score=float(r.distance),
            )
            for r in rows
        ]
        ui = "\n".join(
            f"📖 {r['statement']}\n🧾 {r['evidence']}\n🔹 distance:{r['score']:.3f}\n📌 {r['category']}/{r['subcategory']}\n--------"
            for r in results
        )
        return results, ui


# ------------------------------------
# 工具：解析 XML → DataFrame
# ------------------------------------

def parse_psychological_analysis(xml_input):
    if isinstance(xml_input, list):
        xml_input = xml_input[0]
    xml_string = f"<root>{xml_input}</root>"
    tree = ET.fromstring(xml_string)
    evidence_map = defaultdict(list)
    timestamp = datetime.now().strftime("%Y/%m/%d-%H:%M")

    def _extract(section, cat, sub):
        if section.attrib.get("has_content") == "true":
            for item in section.findall("item"):
                stmt = item.findtext("statement", default="").strip()
                evs = [e.text.strip() for e in item.findall("evidence") if e.text]
                evidence_map[(cat, sub, stmt)].extend(evs)

    for cat_node in tree:
        cat = cat_node.tag
        for sub_node in cat_node:
            _extract(sub_node, cat, sub_node.tag)

    records = [
        {
            "Timestamp": timestamp,
            "Category": k[0],
            "Subcategory": k[1],
            "Statement": k[2],
            "Evidence": " || ".join(v),
        }
        for k, v in evidence_map.items()
    ]
    df = pd.DataFrame(records)
    return records, df
