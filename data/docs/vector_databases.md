Vector database

A vector database, vector store or vector search engine is a [database](https://en.wikipedia.org/wiki/Database) that stores and retrieves [embeddings](https://en.wikipedia.org/wiki/Embedding_(machine_learning)) of data in [vector space](https://en.wikipedia.org/wiki/Vector_space_model). Vector databases typically implement [approximate nearest neighbor](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods) algorithms so users can search for records [semantically similar](https://en.wikipedia.org/wiki/Semantic_similarity) to a given input, unlike traditional databases which primarily look up records by exact match. Use-cases for vector databases include [similarity search](https://en.wikipedia.org/wiki/Similarity_search), [semantic search](https://en.wikipedia.org/wiki/Semantic_search), [multi-modal search](https://en.wikipedia.org/wiki/Multi-modal_search), [recommendations engines](https://en.wikipedia.org/wiki/Recommendations_engine), [object detection](https://en.wikipedia.org/wiki/Object_detection), and [retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)(RAG).

Vector embeddings are mathematical representations of data in a high-dimensional space. In this space, each dimension corresponds to a [feature](https://en.wikipedia.org/wiki/Feature_(machine_learning)) of the data, with the number of dimensions ranging from a few hundred to tens of thousands, depending on the complexity of the data being represented. Each data item is represented by one vector in this space. Words, phrases, or entire documents, as well as images, audio, and other types of data, can all be vectorized.

These feature vectors may be computed from the raw data using [machine learning](https://en.wikipedia.org/wiki/Machine_learning) methods such as [feature extraction](https://en.wikipedia.org/wiki/Feature_extraction) algorithms, [word embeddings](https://en.wikipedia.org/wiki/Word_embeddings) or [deep learning](https://en.wikipedia.org/wiki/Deep_learning) networks. The goal is that semantically similar data items receive feature vectors close to each other.

## Techniques

The most important techniques for similarity search on high-dimensional vectors include:

- [Hierarchical Navigable Small World (HNSW) graphs](https://en.wikipedia.org/wiki/Hierarchical_Navigable_Small_World_graphs)
- [Locality-sensitive Hashing (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) and Sketching
- [Product Quantization](https://en.wikipedia.org/wiki/Product_quantization?action=edit&amp;redlink=1)(PQ)
- [Inverted Files](https://en.wikipedia.org/wiki/Inverted_index)

and combinations of these techniques.

In recent benchmarks, HNSW-based implementations have been among the best performers. Conferences such as the International Conference on Similarity Search and Applications (SISAP) and the [Conference on Neural Information Processing Systems (NeurIPS)](https://en.wikipedia.org/wiki/Conference_on_Neural_Information_Processing_Systems) have hosted competitions on vector search in large databases.

## Applications

Vector databases are used in a wide range of machine learning applications including [similarity search](https://en.wikipedia.org/wiki/Similarity_search), [semantic search](https://en.wikipedia.org/wiki/Semantic_search), [multi-modal search](https://en.wikipedia.org/wiki/Multi-modal_search), [recommendations engines](https://en.wikipedia.org/wiki/Recommendations_engine), [object detection](https://en.wikipedia.org/wiki/Object_detection), and [retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation).

### Retrieval-augmented generation

An especially common use-case for vector databases is in [retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)(RAG), a method to improve domain-specific responses of [large language models](https://en.wikipedia.org/wiki/Large_language_model). The retrieval component of a RAG can be any search system, but is most often implemented as a vector database. Text documents describing the domain of interest are collected, and for each document or document section, a feature vector (known as an " [embedding](https://en.wikipedia.org/wiki/Sentence_embedding)") is computed, typically using a deep learning network, and stored in a vector database along with a link to the document. Given a user prompt, the feature vector of the prompt is computed, and the database is queried to retrieve the most relevant documents. These are then automatically added into the context window of the large language model, and the large language model proceeds to create a response to the prompt given this context.

## Implementations

| Name | License |
| --- | --- |
| [Aerospike](https://en.wikipedia.org/wiki/Aerospike_(database)) | Proprietary |
| [AllegroGraph](https://en.wikipedia.org/wiki/AllegroGraph) | Proprietary (Managed Service) |
| [AlloyDB AI](https://en.wikipedia.org/wiki/AlloyDB_AI?action=edit&amp;redlink=1) | Proprietary (Managed Service) |
| [Apache Cassandra](https://en.wikipedia.org/wiki/Apache_Cassandra) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [Azure Cosmos DB](https://en.wikipedia.org/wiki/Cosmos_DB) | Proprietary (Managed Service) |
| [Chroma](https://en.wikipedia.org/wiki/Chroma_(vector_database)) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [ClickHouse](https://en.wikipedia.org/wiki/ClickHouse) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [Couchbase](https://en.wikipedia.org/wiki/Couchbase) | [BSL 1.1](https://en.wikipedia.org/wiki/Business_Source_License) |
| [CrateDB](https://en.wikipedia.org/wiki/CrateDB) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [DataStax](https://en.wikipedia.org/wiki/DataStax) | Proprietary (Managed Service) |
| [Elasticsearch](https://en.wikipedia.org/wiki/Elasticsearch) | [Server Side Public License](https://en.wikipedia.org/wiki/Server_Side_Public_License), Elastic License |
| JaguarDB | Proprietary |
| LanceDB | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| LlamaIndex | [MIT License](https://en.wikipedia.org/wiki/MIT_License) |
| [MariaDB](https://en.wikipedia.org/wiki/MariaDB) | [GPL v2](https://en.wikipedia.org/wiki/GNU_General_Public_License) |
| Marqo | Apache License 2.0 |
| Meilisearch | [MIT License](https://en.wikipedia.org/wiki/MIT_License) |
| [Milvus](https://en.wikipedia.org/wiki/Milvus_(vector_database)) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [MongoDB](https://en.wikipedia.org/wiki/MongoDB) Atlas | [Server Side Public License](https://en.wikipedia.org/wiki/Server_Side_Public_License)(Managed service) |
| MyScaleDB | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [Neo4j](https://en.wikipedia.org/wiki/Neo4j) | [GPL v3](https://en.wikipedia.org/wiki/GPL_v3)(Community Edition) |
| [ObjectBox](https://en.wikipedia.org/wiki/ObjectBox?action=edit&amp;redlink=1) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [OpenSearch](https://en.wikipedia.org/wiki/OpenSearch_(software)) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [Oracle Database](https://en.wikipedia.org/wiki/Oracle_Database) | Proprietary (Managed Service or License) |
| Pinecone | Proprietary (Managed Service) |
| [Pixeltable](https://en.wikipedia.org/wiki/Pixeltable?action=edit&amp;redlink=1)(Incremental Embedding) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [Postgres](https://en.wikipedia.org/wiki/Postgres) with pgvector | PostgreSQL License |
| Qdrant | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| [Redis](https://en.wikipedia.org/wiki/Redis) Stack | [Redis Source Available License](https://redis.io/docs/about/license/) [Archived](https://web.archive.org/web/20240131205042/https://redis.io/docs/about/license/) 2024-01-31 at the [Wayback Machine](https://en.wikipedia.org/wiki/Wayback_Machine) |
| [ScyllaDB](https://en.wikipedia.org/wiki/ScyllaDB) | Proprietary Source Available |
| [Snowflake](https://en.wikipedia.org/wiki/Snowflake_Inc.) | Proprietary (Managed Service) |
| SurrealDB | [BSL 1.1](https://en.wikipedia.org/wiki/Business_Source_License) |
| [TiDB](https://en.wikipedia.org/wiki/TiDB) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| Typesense | [GPL v3](https://en.wikipedia.org/wiki/GPL_v3)(Community Edition) |
| Vespa | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |
| Weaviate | [BSD 3-Clause](https://en.wikipedia.org/wiki/BSD_3-Clause) |
| [YDB](https://en.wikipedia.org/wiki/YDB_(database)) | [Apache License 2.0](https://en.wikipedia.org/wiki/Apache_License_2.0) |

## See also

- [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)– Difficulties arising when analyzing data with many aspects ("dimensions")
- [Graph database](https://en.wikipedia.org/wiki/Graph_database)– Database using graph structures for queries
- [Machine learning](https://en.wikipedia.org/wiki/Machine_learning)– Study of algorithms that improve automatically through experience
- [Nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)– Optimization problem in computer science
- [Recommender system](https://en.wikipedia.org/wiki/Recommender_system)– System to predict users' preferences

## References

1. ["Vector database"](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database). learn.microsoft.com. 2023-12-26. Retrieved 2024-01-11.
2. Roie Schwaber-Cohen. ["What is a Vector Database & How Does it Work"](https://www.pinecone.io/learn/vector-database/). Pinecone. Retrieved 18 November 2023.
3. ["What is a vector database"](https://www.elastic.co/what-is/vector-database). [Elastic](https://en.wikipedia.org/wiki/Elastic_NV). Retrieved 18 November 2023.
4. Evan Chaki (2023-07-31). ["What is a vector database?"](https://learn.microsoft.com/en-us/semantic-kernel/memories/vector-db). Microsoft. A vector database is a type of database that stores data as high-dimensional vectors, which are mathematical representations of features or attributes.
5. Aumüller, Martin; Bernhardsson, Erik; Faithfull, Alexander (2017), Beecks, Christian; Borutta, Felix; Kröger, Peer; Seidl, Thomas (eds.), ["ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms"](http://link.springer.com/10.1007/978-3-319-68474-1_3), Similarity Search and Applications, vol. 10609, Cham: Springer International Publishing, pp. 34–49, [arXiv](https://en.wikipedia.org/wiki/ArXiv_(identifier)): [1807.05614](https://arxiv.org/abs/1807.05614), [doi](https://en.wikipedia.org/wiki/Doi_(identifier)): [10.1007/978-3-319-68474-1_3](https://doi.org/10.1007%2F978-3-319-68474-1_3), [ISBN](https://en.wikipedia.org/wiki/ISBN_(identifier)) [978-3-319-68473-4](https://en.wikipedia.org/wiki/Special:BookSources/978-3-319-68473-4), retrieved 2024-03-19 [citation](https://en.wikipedia.org/wiki/Template:Citation)`{{}}`: CS1 maint: work parameter with ISBN ([link](https://en.wikipedia.org/wiki/Category:CS1_maint:_work_parameter_with_ISBN))
6. Aumüller, Martin; Bernhardsson, Erik; Faithfull, Alexander (2017). ["ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms"](https://link.springer.com/chapter/10.1007/978-3-319-68474-1_3). In Beecks, Christian; Borutta, Felix; Kröger, Peer; Seidl, Thomas (eds.). Similarity Search and Applications. Lecture Notes in Computer Science. Vol. 10609. Cham: Springer International Publishing. pp. 34–49. [arXiv](https://en.wikipedia.org/wiki/ArXiv_(identifier)): [1807.05614](https://arxiv.org/abs/1807.05614). [doi](https://en.wikipedia.org/wiki/Doi_(identifier)): [10.1007/978-3-319-68474-1_3](https://doi.org/10.1007%2F978-3-319-68474-1_3). [ISBN](https://en.wikipedia.org/wiki/ISBN_(identifier)) [978-3-319-68474-1](https://en.wikipedia.org/wiki/Special:BookSources/978-3-319-68474-1).
7. ["Task description and call for participation SISAP 2025 Indexing Challenge"](https://sisap-challenges.github.io/2025/index.html). sisap-challenges.github.io. Retrieved 2026-01-01.
8. ["NeurIPS Competition Practical Vector Search (Big ANN) Challenge 2023"](https://neurips.cc/virtual/2023/competition/66587). neurips.cc. Retrieved 2026-01-01.
9. Lewis, Patrick; Perez, Ethan; Piktus, Aleksandra; Petroni, Fabio; Karpukhin, Vladimir; Goyal, Naman; Küttler, Heinrich (2020). "Retrieval-augmented generation for knowledge-intensive NLP tasks". Advances in Neural Information Processing Systems 33: 9459–9474. [arXiv](https://en.wikipedia.org/wiki/ArXiv_(identifier)): [2005.11401](https://arxiv.org/abs/2005.11401).
10. ["Aerospike Recognized by Independent Research Firm Among Notable Vendors in Vector Databases Report"](https://www.morningstar.com/news/globe-newswire/9111790/aerospike-recognized-by-independent-research-firm-among-notable-vendors-in-vector-databases-report). Morningstar. 2024-05-07. Retrieved 2024-08-01.
11. ["Aerospike raises $109M for its real-time database platform to capitalize on the AI boom"](https://techcrunch.com/2024/04/04/aerospike-raises-100m-for-its-real-time-database-platform-to-capitalize-on-the-ai-boom/). TechCrunch. 2024-04-04. Retrieved 2024-08-01.
12. ["AllegroGraph 8.0 Incorporates Neuro-Symbolic AI, a Pathway to AGI"](https://thenewstack.io/allegrograph-8-0-incorporates-neuro-symbolic-ai-a-pathway-to-agi/). TheNewStack. 2023-12-29. Retrieved 2024-06-06.
13. ["Franz Inc. Introduces AllegroGraph Cloud: A Managed Service for Neuro-Symbolic AI Knowledge Graphs"](https://www.datanami.com/this-just-in/franz-inc-introduces-allegrograph-cloud-a-managed-service-for-neuro-symbolic-ai-knowledge-graphs/). Datanami. 2024-01-18. Retrieved 2024-06-06.
14. Wiggers, Kyle (2023-08-29). ["Google's AlloyDB AI transforms databases to power generative AI apps"](https://techcrunch.com/2023/08/29/googles-alloydb-ai-transforms-databases-to-power-generative-ai-apps/). TechCrunch. Retrieved 2026-01-01.
15. ["5 Hard Problems in Vector Search, and How Cassandra Solves Them"](https://thenewstack.io/5-hard-problems-in-vector-search-and-how-cassandra-solves-them/). TheNewStack. 2023-09-22. Retrieved 2023-09-22.
16. ["Vector Search quickstart"](https://cassandra.apache.org/doc/latest/cassandra/vector-search/overview.html). Retrieved 2023-11-21.
17. ["Vector database"](https://learn.microsoft.com/azure/cosmos-db/vector-database). learn.microsoft.com. 26 December 2023. Retrieved 2024-01-10.
18. Palazzolo, Stephanie. ["Vector database Chroma scored $18 million in seed funding at a $75 million valuation. Here's why its technology is key to helping generative AI startups"](https://www.businessinsider.com/vector-database-startup-chroma-raises-seed-funding-generative-artificial-intelligence-2023-4). Business Insider. Retrieved 2023-11-16.
19. MSV, Janakiram (2023-07-28). ["Exploring Chroma: The Open Source Vector Database for LLMs"](https://thenewstack.io/exploring-chroma-the-open-source-vector-database-for-llms/). The New Stack. Retrieved 2023-11-16.
20. ["chroma/LICENSE at main · chroma-core/chroma"](https://github.com/chroma-core/chroma/blob/main/LICENSE). GitHub.
21. ["Can you use ClickHouse for vector search? | ClickHouse Docs"](https://clickhouse.com/docs/knowledgebase/vector-search). 2023-10-26. [Archived](https://web.archive.org/web/20250622222726/https://clickhouse.com/docs/knowledgebase/vector-search) from the original on 2025-06-22. Retrieved 2025-07-02.
22. ["Couchbase aims to boost developer database productivity with Capella IQ AI tool"](https://venturebeat.com/ai/couchbase-aims-to-boost-developer-database-productivity-with-capella-iq-ai-tool/#h-next-on-the-roadmap-for-couchbase-is-vector-support). VentureBeat. 2023-08-30.
23. ["Investor Presentation Third Quarter Fiscal 2024"](https://investors.couchbase.com/static-files/551e5b96-5307-4119-b225-19cfd8540242). Couchbase Investor Relations. 2023-12-06.
24. Anderson, Scott (2021-03-26). ["Couchbase Adopts BSL License"](https://www.couchbase.com/blog/couchbase-adopts-bsl-license/). The Couchbase Blog. Retrieved 2024-02-14.
25. ["Open Source Vector Database"](https://cratedb.com/blog/open-source-vector-database). CrateDB Blog. 16 November 2023. Retrieved 2024-11-06.
26. Sean Michael Kerner (18 July 2023). ["DataStax brings vector database search to multicloud with Astra DB"](https://venturebeat.com/data-infrastructure/datastax-brings-vector-database-search-to-multicloud-with-astra-db/). Venture Beat.
27. Kerner, Sean (23 May 2023). ["Elasticsearch Relevance Engine brings new vectors to generative AI"](https://venturebeat.com/ai/elasticsearch-relevance-engine-brings-new-vectors-to-generative-ai/). [VentureBeat](https://en.wikipedia.org/wiki/VentureBeat). Retrieved 18 November 2023.
28. ["elasticsearch/LICENSE.txt at main · elastic/elasticsearch"](https://github.com/elastic/elasticsearch/blob/main/LICENSE.txt). GitHub.
29. ["JaguarDB Homepage"](http://jaguardb.com/). JaguarDB. Retrieved 2025-04-12.
30. ["Vector DBMS"](https://db-engines.com/de/ranking/vector+dbms). db-engines.com. 2023-07-03. Retrieved 2025-04-12.
31. ["LanceDB Homepage"](https://lancedb.com/). LanceDB. 2024-12-17. Retrieved 2024-12-17.
32. ["A scalable, elastic database and search solution for 1B+ vectors built on LanceDB and Amazon S3 | AWS Architecture Blog"](https://aws.amazon.com/blogs/architecture/a-scalable-elastic-database-and-search-solution-for-1b-vectors-built-on-lancedb-and-amazon-s3/). aws.amazon.com. 2025-09-22. Retrieved 2026-01-01.
33. ["lancedb/LICENSE at main · lancedb/lancedb"](https://github.com/lancedb/lancedb?tab=Apache-2.0-1-ov-file). GitHub. Retrieved 2024-12-17.
34. Wiggers, Kyle (2023-06-06). ["LlamaIndex adds private data to large language models"](https://techcrunch.com/2023/06/06/llamaindex-adds-private-data-to-large-language-models/). TechCrunch. Retrieved 2023-10-29.
35. ["llama_index/LICENSE at main · run-llama/llama_index"](https://github.com/run-llama/llama_index/blob/main/LICENSE). GitHub. Retrieved 2023-10-29.
36. ["MariaDB Vector"](https://mariadb.org/projects/mariadb-vector/). MariaDB.org. Retrieved 2024-07-30.
37. ["Vector search in old and modern databases"](https://manticoresearch.com/blog/vector-search-in-databases/). manticoresearch.com. Retrieved 2024-07-30.
38. ["Licensing FAQ"](https://mariadb.com/kb/en/licensing-faq/). MariaDB KnowledgeBase. Retrieved 2024-07-30.
39. Sawers, Paul (2023-08-16). ["Meet Marqo, an open source vector search engine for AI applications"](https://techcrunch.com/2023/08/16/meet-marqo-an-open-source-vector-search-engine-for-ai-applications/). TechCrunch. Retrieved 2024-08-20.
40. [marqo-ai/marqo](https://github.com/marqo-ai/marqo?tab=Apache-2.0-1-ov-file#readme), Marqo, 2024-08-20, retrieved 2024-08-20
41. ["Meilisearch Homepage"](https://meilisearch.com/). Meilisearch. 2024-10-08. Retrieved 2023-10-29.
42. ["Compare Algolia vs ElasticSearch vs Meilisearch vs Typesense"](https://typesense.org/typesense-vs-algolia-vs-elasticsearch-vs-meilisearch/). typesense.org. Retrieved 2026-01-01.
43. ["meilisearch/LICENSE at main · meilisearch/meilisearch"](https://github.com/meilisearch/meilisearch/blob/main/LICENSE). GitHub. Retrieved 2024-10-08.
44. Liao, Ingrid Lunden and Rita (2022-08-24). ["Zilliz raises $60M, relocates to SF"](https://techcrunch.com/2022/08/24/zilliz-the-startup-behind-the-milvus-open-source-vector-database-for-ai-applications-raises-60m-and-relocates-to-sf/). TechCrunch. Retrieved 2023-10-29.
45. ["Milvus license"](https://github.com/milvus-io/milvus/blob/master/LICENSE). [GitHub](https://en.wikipedia.org/wiki/GitHub).
46. ["Introducing Atlas Vector Search: Build Intelligent Applications with Semantic Search and AI Over Any Type of Data"](https://www.mongodb.com/blog/post/introducing-atlas-vector-search-build-intelligent-applications-semantic-search-ai). MongoDB. 2023-06-22.
47. Jamil, Usama (2024-05-20). ["Build an Advanced RAG Application Using MyScaleDB and LlamaIndex"](https://thenewstack.io/build-an-advanced-rag-application-using-myscaledb-and-llamaindex/). The New Stack. Retrieved 2026-01-01.
48. ["Neo4j enhances its graph database with vector search"](https://itbrief.com.au/story/neo4j-enhances-its-graph-database-with-vector-search). itbrief. 2023-08-22.
49. ["Vector search indexes"](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes). neo4j.
50. ["Neo4j Licensing"](https://neo4j.com/licensing/).
51. ["Top Fifteen Vector Databases"](https://db-engines.com/de/ranking/vektor+dbms). db-engines.com. 2024-07-03. Retrieved 2024-07-03.
52. ["ObjectBox Java license"](https://github.com/objectbox/objectbox-java/blob/main/LICENSE.txt). github.
53. ["Using OpenSearch as a Vector Database"](https://opensearch.org/platform/search/vector-database.html). OpenSearch.org. 2023-08-02. Retrieved 2024-02-07.
54. Pan, James Jie; Wang, Jianguo; Li, Guoliang (2023-10-21), Survey of Vector Database Management Systems, [arXiv](https://en.wikipedia.org/wiki/ArXiv_(identifier)): [2310.14021](https://arxiv.org/abs/2310.14021)
55. ["OpenSearch license"](https://github.com/opensearch-project/OpenSearch/blob/main/LICENSE.txt). github.
56. Hook, Doug; Priyadarshi, Ranjan (May 2, 2024). ["Oracle Announces General Availability of AI Vector Search in Oracle Database 23ai"](https://blogs.oracle.com/database/post/oracle-announces-general-availability-of-ai-vector-search-in-oracle-database-23ai). oracle. Retrieved July 9, 2024.
57. ["Pinecone leads 'explosion' in vector databases for generative AI"](https://venturebeat.com/ai/pinecone-leads-explosion-in-vector-databases-for-generative-ai/). VentureBeat. 2023-07-14. Retrieved 2023-10-29.
58. ["Automatic incremental embedding index"](https://www.pixeltable.com/blog/pixeltable-incremental-embedding-indexes). Pixeltable. 24 April 2025. Retrieved 2025-07-04.
59. ["pgvector"](https://github.com/pgvector/pgvector). GitHub. Retrieved 2023-11-27.
60. ["pgvector/License"](https://github.com/pgvector/pgvector/blob/master/LICENSE). GitHub. Retrieved 2023-11-27.
61. Sawers, Paul (2023-04-19). ["Qdrant, an open-source vector database startup, wants to help AI developers leverage unstructured data"](https://techcrunch.com/2023/04/19/qdrant-an-open-source-vector-database-startup-wants-to-help-ai-developers-leverage-unstructured-data/). TechCrunch. Retrieved 2023-10-29.
62. ["qdrant/LICENSE at master · qdrant/qdrant"](https://github.com/qdrant/qdrant/blob/master/LICENSE). GitHub. Retrieved 2023-10-29.
63. ["Using Redis as a Vector Database with OpenAI | OpenAI Cookbook"](https://cookbook.openai.com/examples/vector_databases/redis/getting-started-with-redis-and-openai). cookbook.openai.com. Retrieved 2024-02-10.
64. ["Redis as a vector database quick start guide"](https://web.archive.org/web/20240131205622/https://redis.io/docs/get-started/vector-database/). Redis. Archived from [the original](https://redis.io/docs/get-started/vector-database/) on 2024-01-31. Retrieved 2024-01-31.
65. ["Search and query"](https://redis.io/docs/interact/search-and-query/). Redis. Retrieved 2024-02-10.
66. ["Open source USearch library jumpstarts ScyllaDB vector search"](https://thenewstack.io/open-source-usearch-library-jumpstarts-scylladb-vector-search/). TheNewStack. 2026-02-05. Retrieved 2026-02-23.
67. ["ScyllaDB adds vector search to managed database platform"](https://www.techtarget.com/searchdatamanagement/news/366637573/ScyllaDB-adds-vector-search-to-managed-database-platform). TechTarget. 2026-01-20. Retrieved 2026-02-23.
68. ["Vector data type and vector similarity functions — General Availability"](https://docs.snowflake.com/en/release-notes/2024/other/2024-05-16-vector-data-type-ga). Snowflake. 2024-05-17. Retrieved 2024-05-17.
69. Wiggers, Kyle (2023-01-04). ["SurrealDB raises $6M for its database-as-a-service offering"](https://techcrunch.com/2023/01/04/surrealdb-raises-6m-startup-funding-database-as-a-service/). TechCrunch. Retrieved 2024-01-19.
70. ["SurrealDB | License FAQs | The ultimate multi-model database"](https://surrealdb.com/license). SurrealDB. Retrieved 2024-02-14.
71. ["TiDB Vector Search"](https://docs.pingcap.com/tidb/stable/vector-search-overview/). docs.pingcap.com.
72. [tidb/LICENSE at master · pingcap/tidb](https://github.com/pingcap/tidb/blob/master/LICENSE)
73. Martinez, Miguel (2024-06-20). ["Typesense Homepage"](https://typesense.org/). Typesense. Retrieved 2024-06-20.
74. ["Typesense licensing"](https://github.com/typesense/typesense/blob/main/LICENSE.txt). [GitHub](https://en.wikipedia.org/wiki/GitHub).
75. ["Creating a Vespa Vector Database"](https://www.capitalone.com/tech/ai/vector-database-intro/). Capital One. Retrieved 2026-01-01.
76. ["vespa/LICENSE at master · vespa-engine/vespa"](https://github.com/vespa-engine/vespa/blob/master/LICENSE). GitHub.
77. ["weaviate/LICENSE at master · weaviate/weaviate"](https://github.com/weaviate/weaviate/blob/master/LICENSE). GitHub. Retrieved 2023-10-29.
78. ["Langchain YDB"](https://python.langchain.com/docs/integrations/vectorstores/ydb/). Langchain. Retrieved 2025-07-26.
79. ["YDB - Vector Search"](https://ydb.tech/docs/en/concepts/vector_search). ydb.tech. Retrieved 2025-07-26.
80. ["ydb/LICENSE at master · ydb-platform/ydb"](https://github.com/ydb-platform/ydb/blob/main/LICENSE). GitHub. Retrieved 2025-07-26.

## External links

- Sawers, Paul (2024-04-20). ["Why vector databases are having a moment as the AI hype cycle peaks"](https://techcrunch.com/2024/04/20/why-vector-databases-are-having-a-moment-as-the-ai-hype-cycle-peaks/). TechCrunch. Retrieved 2024-04-23.