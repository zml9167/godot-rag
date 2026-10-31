from chunkana import chunk_markdown, ChunkConfig

config = ChunkConfig(
    max_chunk_size=2048,
    min_chunk_size=100,       # 调小，减少小片段被合并/丢弃
    overlap_size=100
)

with open(r"D:\code\python\godot-rag\docs-source\introduction.md", encoding='utf-8') as fp:
    content = fp.read()
    chunks = chunk_markdown(content, config)
    print(chunks)