from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, select, text, update, func, insert
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import os
from dotenv import load_dotenv
from datetime import datetime
from passlib.context import CryptContext
from pgvector.sqlalchemy import Vector 
import numpy as np

# ✅ 載入環境變數（例如 DATABASE_URL）
load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")

# ✅ 建立非同步資料庫引擎與 Session 工廠
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# ✅ SQLAlchemy 基底類別
Base = declarative_base()

# ✅ 密碼加密設定
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 👤 使用者資料表
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # 建立關聯：一對多
    conversations = relationship("Conversation", back_populates="user")
    analyses = relationship("AnalysisSummary", back_populates="user")

# 💬 對話紀錄資料表
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)  # 'user' or 'assistant'
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # 建立反向關聯
    user = relationship("User", back_populates="conversations")
    
# 分析摘要記錄表
class AnalysisSummary(Base):
    __tablename__ = "analysis_summaries"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # 外鍵關聯到 User.id

    timestamp = Column(String)  # 或 Column(DateTime)
    category = Column(String)
    subcategory = Column(String)
    statement = Column(String)
    evidence = Column(String)
    statement_vector = Column(Vector(1536))  # ✅ 使用 pgvector 的向量欄位
    vector_analyzed = Column(Boolean, default=False)  # ✅ 是否已納入 ANALYZE 的標記

    # 建立反向關聯：每一筆分析紀錄都屬於某位使用者
    user = relationship("User", back_populates="analyses")

async def initialize_database():
    await init_db()
    async with AsyncSessionLocal() as session:
        await init_pgvector(session)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("✅ 非同步資料庫初始化完成")

async def init_pgvector(session: AsyncSession):
    try:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await session.commit()
        print("✅ pgvector 擴充已啟用")
    except Exception as e:
        print(f"❌ 初始化 pgvector 失敗: {e}")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ✅ 自動檢查是否需要執行 ANALYZE（提升向量查詢效率）
async def maybe_analyze_vectors(session: AsyncSession, threshold: int = 100):
    try:
        result = await session.execute(
            select(func.count()).select_from(AnalysisSummary).where(
                AnalysisSummary.vector_analyzed == False
            )
        )
        count = result.scalar_one()
        if count >= threshold:
            await session.execute("ANALYZE analysis_summaries;")
            await session.execute(
                update(AnalysisSummary)
                .where(AnalysisSummary.vector_analyzed == False)
                .values(vector_analyzed=True)
            )
            await session.commit()
            print(f"⚡ 已執行 ANALYZE（向量筆數: {count}）")
    except Exception as e:
        await session.rollback()
        print(f"❌ 資料庫錯誤: {e}")        

# === 1. 批次新增心理分析 ===
async def insert_analyses(session: AsyncSession, user_id: int, rows: list[dict]) -> None:
    if not rows:
        print("⚠️ 無分析資料，跳過儲存")
        return
    try:
        print(f"===⚠️ user_id：{user_id}")
        print(f"===⚠️ 儲存分析結果：{rows}")
        
        await session.execute(
            insert(AnalysisSummary),
            [
                dict(
                    user_id      = user_id,
                    timestamp    = r["Timestamp"],
                    category     = r["Category"],
                    subcategory  = r["Subcategory"],
                    statement    = r["Statement"],
                    evidence     = r["Evidence"],
                    vector_analyzed = False,
                )
                for r in rows
            ],
        )
        # await session.commit()     必須在外層commit 否則會錯誤
    except Exception as e:
        await session.rollback()
        print(f"❌ 資料庫錯誤: {e}")    

# === 2. 撈還沒向量化的紀錄 ===
async def select_unembedded(session: AsyncSession, user_id: int):
    try:
        q = (
            select(AnalysisSummary)
            .where(
                AnalysisSummary.user_id == user_id,
                AnalysisSummary.statement_vector == None,
                AnalysisSummary.statement != None,
            )
            .order_by(AnalysisSummary.id)
        )
        return (await session.execute(q)).scalars().all()
    except Exception as e:
        await session.rollback()
        print(f"❌ 資料庫錯誤: {e}")   

# === 3. 回寫向量 ===
async def update_vectors(session: AsyncSession, pairs: list[tuple[int, list[float]]]) -> None:
    try:
        for rec_id, vec in pairs:
            await session.execute(
                update(AnalysisSummary)
                .where(AnalysisSummary.id == rec_id)
                .values(statement_vector=vec)
            )
        # await session.commit()     必須在外層commit 否則會錯誤
    except Exception as e:
        # await session.rollback() => 外層處理錯誤，否則會導致錯誤
        print(f"❌ 資料庫錯誤: {e}")           

# === 4. pgvector 相似度搜尋（取前 k 筆） ===
async def pgvector_search(
    session: AsyncSession, user_id: int, query_vec: list[float], top_k: int = 5
):
    try:
        if isinstance(query_vec, np.ndarray):
            query_vec = query_vec.tolist()
        query_str = ",".join(str(round(x, 6)) for x in query_vec)
        vector_sql = f"ARRAY[{query_str}]::vector"

        sql = text(f"""
            SELECT id,
                   category,
                   subcategory,
                   statement,
                   evidence,
                   statement_vector <-> {vector_sql} AS distance
            FROM   analysis_summaries
            WHERE  user_id = :uid
              AND  statement_vector IS NOT NULL
            ORDER BY distance
            LIMIT  :k
        """) 

        rows = await session.execute(sql.bindparams(uid=user_id, k=top_k))
        return rows.fetchall()

    except Exception as e:
        await session.rollback()
        print(f"❌ 資料庫錯誤: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())
    print("資料庫初始化完成")