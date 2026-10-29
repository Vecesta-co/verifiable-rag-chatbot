# DEVLOG — verifiable-rag-chatbot

## Development Notes

- **Obstacle**: Conventional RAG systems lose source traceability once chunks are embedded and synthesized.
- **Solution**: Embedded `start_line`, `end_line`, and SHA256 content hashes directly into `DocumentChunk` and `Citation` schema.
- **Obstacle**: Balancing keyword search vs semantic term frequency without vector database bloat.
- **Solution**: Engineered a unified `HybridIndex` implementing BM25 Okapi and TF-IDF with IDF smoothing in pure Python standard library math.
- **Testing**: 22 unit tests passed in 0.13s.
