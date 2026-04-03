Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks 

Title:

Description:

# Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks

Yunfan Gao, Yun Xiong, Meng Wang, Haofen Wang Yunfan Gao is with Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University, Shanghai, 201210, China. Yun Xiong is with Shanghai Key Laboratory of Data Science, School of Computer Science, Fudan University, Shanghai, 200438, China. Meng Wang and Haofen Wang are with College of Design and Innovation, Tongji University, Shanghai, 20092, China. (Corresponding author: Haofen Wang. E-mail: carter.whfcarter@gmail.com)

###### Abstract

Retrieval-augmented Generation (RAG) has markedly enhanced the capabilities of Large Language Models (LLMs) in tackling knowledge-intensive tasks. The increasing demands of application scenarios have driven the evolution of RAG, leading to the integration of advanced retrievers, LLMs and other complementary technologies, which in turn has amplified the intricacy of RAG systems. However, the rapid advancements are outpacing the foundational RAG paradigm, with many methods struggling to be unified under the process of “retrieve-then-generate”. In this context, this paper examines the limitations of the existing RAG paradigm and introduces the modular RAG framework. By decomposing complex RAG systems into independent modules and specialized operators, it facilitates a highly reconfigurable framework. Modular RAG transcends the traditional linear architecture, embracing a more advanced design that integrates routing, scheduling, and fusion mechanisms. Drawing on extensive research, this paper further identifies prevalent RAG patterns—linear, conditional, branching, and looping—and offers a comprehensive analysis of their respective implementation nuances. Modular RAG presents innovative opportunities for the conceptualization and deployment of RAG systems. Finally, the paper explores the potential emergence of new operators and paradigms, establishing a solid theoretical foundation and a practical roadmap for the continued evolution and practical deployment of RAG technologies.

## I Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities, yet they still face numerous challenges, such as hallucination and the lag in information updates [1]. Retrieval-augmented Generation (RAG), by accessing external knowledge bases, provides LLMs with important contextual information, significantly enhancing their performance on knowledge-intensive tasks [2]. Currently, RAG, as an enhancement method, has been widely applied in various practical application scenarios, including knowledge question answering, recommendation systems, customer service, and personal assistants. [3, 4, 5, 6]

During the nascent stages of RAG , its core framework is constituted by indexing, retrieval, and generation, a paradigm referred to as Naive RAG [7]. However, as the complexity of tasks and the demands of applications have escalated, the limitations of Naive RAG have become increasingly apparent. As depicted in Figure 1, it predominantly hinges on the straightforward similarity of chunks, result in poor performance when confronted with complex queries and chunks with substantial variability. The primary challenges of Naive RAG include: 1) Shallow Understanding of Queries. The semantic similarity between a query and document chunk is not always highly consistent. Relying solely on similarity calculations for retrieval lacks an in-depth exploration of the relationship between the query and the document [8]. 2) Retrieval Redundancy and Noise. Feeding all retrieved chunks directly into LLMs is not always beneficial. Research indicates that an excess of redundant and noisy information may interfere with the LLM’s identification of key information, thereby increasing the risk of generating erroneous and hallucinated responses. [9]

To overcome the aforementioned limitations, Advanced RAG paradigm focuses on optimizing the retrieval phase, aiming to enhance retrieval efficiency and strengthen the utilization of retrieved chunks. As shown in Figure 1 ,typical strategies involve pre-retrieval processing and post-retrieval processing. For instance, query rewriting is used to make the queries more clear and specific, thereby increasing the accuracy of retrieval [10], and the reranking of retrieval results is employed to enhance the LLM’s ability to identify and utilize key information [11].

Despite the improvements in the practicality of Advanced RAG, there remains a gap between its capabilities and real-world application requirements. On one hand, as RAG technology advances, user expectations rise, demands continue to evolve, and application settings become more complex. For instance, the integration of heterogeneous data and the new demands for system transparency, control, and maintainability. On the other hand, the growth in application demands has further propelled the evolution of RAG technology.

Figure 1: Cases of Naive RAG and Advanced RAG.When faced with complex questions, both encounter limitations and struggle to provide satisfactory answers. Despite the fact that Advanced RAG improves retrieval accuracy through hierarchical indexing, pre-retrieval, and post-retrieval processes, these relevant documents have not been used correctly. Figure 2: Case of current Modular RAG.The system integrates diverse data and more functional components. The process is no longer confined to linear but is controlled by multiple control components for retrieval and generation, making the entire system more flexible and complex.

As shown in Figure 2, to achieve more accurate and efficient task execution, modern RAG systems are progressively integrating more sophisticated function, such as organizing more refined index base in the form of knowledge graphs, integrating structured data through query construction methods, and employing fine-tuning techniques to enable encoders to better adapt to domain-specific documents.

In terms of process design, the current RAG system has surpassed the traditional linear retrieval-generation paradigm. Researchers use iterative retrieval [12] to obtain richer context, recursive retrieval [13] to handle complex queries, and adaptive retrieval [14] to provide overall autonomy and flexibility. This flexibility in the process significantly enhances the expressive power and adaptability of RAG systems, enabling them to better adapt to various application scenarios. However, this also makes the orchestration and scheduling of workflows more complex, posing greater challenges to system design. Specifically, RAG currently faces the following new challenges:

Complex data sources integration. RAG are no longer confined to a single type of unstructured text data source but have expanded to include various data types, such as semi-structured data like tables and structured data like knowledge graphs [15]. Access to heterogeneous data from multiple sources can provide the system with a richer knowledge background, and more reliable knowledge verification capabilities [16].

New demands for system interpretability, controllability, and maintainability. With the increasing complexity of systems, system maintenance and debugging have become more challenging. Additionally, when issues arise, it is essential to quickly pinpoint the specific components that require optimization.

Component selection and optimization. More neural networks are involved in the RAG system, necessitating the selection of appropriate components to meet the needs of specific tasks and resource configurations. Moreover, additional components enhance the effectiveness of RAG but also bring new collaborative work requirements [17]. Ensuring that these models perform as intended and work efficiently together to enhance the overall system performance is crucial.

Workflow orchestration and scheduling. Components may need to be executed in a specific order, processed in parallel under certain conditions, or even judged by the LLM based on different outputs. Reasonable planning of the workflow is essential for improving system efficiency and achieving the desired outcomes [18].

To address the design, management, and maintenance challenges posed by the increasing complexity of RAG systems, and to meet the ever-growing and diverse demands and expectations, this paper proposes Modular RAG architecture. In modern computing systems, modularization is becoming a trend. It can enhance the system’s scalability and maintainability and achieve efficient task execution through process control.

The Modular RAG system consists of multiple independent yet tightly coordinated modules, each responsible for handling specific functions or tasks. This architecture is divided into three levels: the top level focuses on the critical stages of RAG, where each stage is treated as an independent module. This level not only inherits the main processes from the Advanced RAG paradigm but also introduces an orchestration module to control the coordination of RAG processes. The middle level is composed of sub-modules within each module, further refining and optimizing the functions. The bottom level consists of basic units of operation—operators. Within the Modular RAG framework, RAG systems can be represented in the form of computational graphs, where nodes represent specific operators. The comparison of the three paradigms is shown in the Figure 3. Modular RAG evolves based on the previous development of RAG. The relationships among these three paradigms are ones of inheritance and development. Advanced RAG is a special case of Modular RAG, while Naive RAG is a special case of Advanced RAG.

Figure 3: Comparison between three RAG paradigms. Modular RAG has evolved from previous paradigms and aligns with the current practical needs of RAG systems.

The advantages of Modular RAG are significant, as it enhances the flexibility and scalability of RAG systems. Users can flexibly combine different modules and operators according to the requirements of data sources and task scenarios. In summary, the contributions of this paper are as follows:

•

This paper proposes a new paradigm called modular RAG, which employs a three-tier architectural design comprising modules, sub-modules, and operators to define the RAG system in a unified and structured manner. This design not only enhances the system’s flexibility and scalability but also, through the independent design of operators, strengthens the system’s maintainability and comprehensibility.

•

Under the framework of Modular RAG, the orchestration of modules and operators forms the RAG Flow, which can flexibly express current RAG methods. This paper has further summarized six typical flow patterns and specific methods have been analyzed to reveal the universality of modular RAG in practical scenarios.

•

The Modular RAG framework offers exceptional flexibility and extensibility. This paper delves into the new opportunities brought by Modular RAG and provides a thorough discussion on the adaptation and expansion of new methods in different application scenarios, offering guidance for future research directions and practical exploration.

- References

## II Related Work

The development of RAG technology can be summarized in three stages. Initially, retrieval-augmented techniques were introduced to improve the performance of pre-trained language models on knowledge-intensive tasks [19, 20]. In specific implementations, Retro [21] optimized pre-trained autoregressive models through retrieval augmentation, while Atlas [22] utilized a retrieval-augmented few-shot fine-tuning method, enabling language models to adapt to diverse tasks. IRCOT [23] further enriched the reasoning process during the inference phase by combining chain-of-thought and multi-step retrieval processes. Entering the second stage, as the language processing capabilities of LLMs significantly improved, retrieval-augmented techniques began to serve as a means of supplementing additional knowledge and providing references, aiming to reduce the hallucination. For instance, RRR [24] improved the rewriting phase, and LLMlingua [25] removed redundant tokens in retrieved document chunks. With the continuous progress of RAG technology, research has become more refined and focused, while also achieving innovative integration with other technologies such as graph neural networks [26] and fine-tuning techniques [27]. The overall pipeline has also become more flexible, such as using LLMs to proactively determine the timing of retrieval and generation [14, 28].

The development of RAG technology has been accelerated by LLM technology and practical application needs. Researchers are examining and organizing the RAG framework and development pathways from different perspectives. Building upon the enhanced stages of RAG, Gao et al., [2] subdivided RAG into enhancement during pre-training, inference, and fine-tuning stages. Based on the main processes of RAG, relevant works on RAG were organized from the perspectives of retrieval, generation, and augmentation methods. Huang et al., [29] categorize RAG methods into four main classes: pre-retrieval, retrieval, post-retrieval, generation, and provide a detailed discussion of the methods and techniques within each class. Hu et al., [30] discuss Retrieval-Augmented Language Models (RALMs) form three key components, including retrievers, language models, augmentations, and how their interactions lead to different model structures and applications. They emphasize the importance of considering robustness, accuracy, and relevance when evaluating RALMs and propose several evaluation methods. Ding et al., [31] provide a comprehensive review from the perspectives of architecture, training strategies, and applications. They specifically discuss four training methods of RALMs: training-free methods, independent training methods, sequence training methods, and joint training methods, and compare their advantages and disadvantages. Zhao et al., [32]analyze the applications of RAG technology in various fields such as text generation, code generation, image generation, and video generation from the perspective of augmented intelligence with generative capabilities.

The current collation of RAG systems primarily focuses on methods with a fixed process, mainly concerned with optimizing the retrieval and generation stages. However, it has not turned its attention to the new characteristics that RAG research is continuously evolving, namely the characteristics of process scheduling and functional componentization. There is currently a lack of comprehensive analysis of the overall RAG system, which has led to research on paradigms lagging behind the development of RAG technology.

## III Framework and notation

| Notation | Description |
| --- | --- |
| qq | The original query |
| yy | The output of LLM |
| DD | A document retrieval repository composed of chunks did_{i}. |
| R​(q,D)R(q,D) | Retriever,find similar chunks from DD based on qq. |
| ℱ\mathcal{F} | RAG Flow |
| 𝒫\mathcal{P} | RAG Flow pattern |
| fq​ef_{qe} | Query expansion function |
| fq​cf_{qc} | Query transform function |
| fc​o​m​pf_{comp} | Chunk compression function |
| fs​e​lf_{sel} | Chunk selection function |
| frf_{r} | Routing function |
| MM | Module in modular RAG |
| o​pop | The specific operators within the Module. |

TABLE I: Important Notation

For query Q={qi}Q=\{q_{i}\}, a typical RAG system mainly consists of three key components. 1) Indexing. Given documents 𝒟={d1,d2,…,dn}\mathcal{D}=\{d_{1},d_{2},\ldots,d_{n}\} , where did_{i} represents the document chunk. Indexing is the process of converting did_{i} into vectors through an embedding model fe​(⋅)f_{e}(\cdot) , and then store vectors in vector database.

| ℐ={e1,e2,…,en}a​n​dei=fe​(di)∈ℝd\mathcal{I}=\{e_{1},e_{2},\ldots,e_{n}\}\quad and\quad e_{i}=f_{e}(d_{i})\in\mathbb{R}^{d} | (1) |
| --- | --- |

2) Retrieval . Transform the query into a vector using the same encoding model, and then filter out the top kk document chunks that are most similar based on vector similarity.

| ℛ:topkdi∈DSim⁡(q,di)→Dq\mathcal{R}:\mathop{\mathrm{topk}}_{d_{i}\in D}\operatorname{Sim}(q,d_{i})\rightarrow D^{q} | (2) |
| --- | --- |

Dq={d1,d2,…,dk}D^{q}=\{d_{1},d_{2},\ldots,d_{k}\} represents the relevant documents for question qq. The similarity function S​i​m​(⋅)Sim(\cdot) commonly used are dot product or cosine similarity.

| S​i​m​(q,di)=eq⋅edio​req⋅edi‖eq‖⋅‖edi‖Sim(q,d_{i})=e_{q}\cdot e_{d_{i}}\quad or\quad\frac{e_{q}\cdot e_{d_{i}}}{\|e_{q}\|\cdot\|e_{d_{i}}\|} | (3) |
| --- | --- |

3) Generation. After getting the relevant documents. The query qq and the retrieved document DqD^{q} chunks are inputted together to the LLM to generate the final answer, where [⋅,⋅][\cdot,\cdot] stands for concatenation.

| y=ℒ​ℒ​ℳ​([Dq,q])y=\mathcal{LLM}([D^{q},q]) | (4) |
| --- | --- |

With the evolution of RAG technology, more and more functional components are being integrated into systems. Modular RAG paradigm includes three levels, ranging from large to small:

L1 Module (ℳ={ℳs}\mathcal{M}=\{\mathcal{M}_{s}\}). The core process in RAG system.

L2 Sub-module (ℳs={O​p}\mathcal{M}_{s}=\{Op\}).The functional modules in module.

L3 Operator (𝒪​p={fθi}\mathcal{O}p=\{f_{\theta_{i}}\}). The the specific functional implementation in a module or sub-module. As a result, a Modular RAG system can be represented as:

| 𝒢={q,D,ℳ,{ℳs},{O​p}}\mathcal{G}=\{q,D,\mathcal{M},\{\mathcal{M}_{s}\},\{Op\}\} | (5) |
| --- | --- |

The arrangement between modules and operators constitutes the RAG Flow ℱ=(Mϕ1,…,Mϕn)\mathcal{F}=(M_{\phi_{1}},\ldots,M_{\phi_{n}}) where ϕ\phi stands for the set of module parameters. A modular rag flow can be decomposed into a graph of sub-functions. In the simplest case,the graph is a linear chain.

| N​a​i​v​e​R​A​G:q→T​e​x​t−E​m​b​e​d​d​i​n​gR​(q,D)Dq→O​p​e​n​A​I/G​P​T−4L​L​M​([q,Dq])yNaiveRAG:q\xrightarrow[Text-Embedding]{R(q,D)}D^{q}\xrightarrow[OpenAI/GPT-4]{LLM([q,D^{q}])}y | (6) |
| --- | --- |

## IV Module and Operator

This chapter will specifically introduce modules and operators under the Modular RAG framework. Based on the current stage of RAG development, we have established six main modules: Indexing, Pre-retrieval, Retrieval, Post-retrieval, Generation, and Orchestration.

### IV-A Indexing

Indexing is the process of split document into manageable chunks and it is a key step in organizing a system. Indexing faces three main challenges. 1) Incomplete content representation.The semantic information of chunks is influenced by the segmentation method, resulting in the loss or submergence of important information within longer contexts. 2) Inaccurate chunk similarity search. As data volume increases, noise in retrieval grows, leading to frequent matching with erroneous data, making the retrieval system fragile and unreliable. 3) Unclear reference trajectory. The retrieved chunks may originate from any document, devoid of citation trails, potentially resulting in the presence of chunks from multiple different documents that, despite being semantically similar, contain content on entirely different topics.

#### IV-A1 Chunk Optimization

The size of the chunks and the overlap between the chunks play a crucial role in the overall effectiveness of the RAG system. Given a chunk did_{i}, its chunk size is denoted as Li=|di|L_{i}=|d_{i}|, and the overlap is denoted as Lio=|di∩di+1|L^{o}_{i}=|d_{i}\cap d_{i+1}|. Larger chunks can capture more context, but they also generate more noise, requiring longer processing time and higher costs. While smaller chunks may not fully convey the necessary context, they do have less noise [17].

Sliding Window using overlapping chunks in a sliding window enhances semantic transitions. However, it has limitations such as imprecise context size control, potential truncation of words or sentences, and lacking semantic considerations.

Metadata Attachment. Chunks can be enriched with metadata like page number, file name, author, timestamp, summary, or relevant questions. This metadata allows for filtered retrieval, narrowing the search scope.

Small-to-Big [33] separate the chunks used for retrieval from those used for synthesis. Smaller chunks enhance retrieval accuracy, while larger chunks provide more context. One approach is to retrieve smaller summarized chunks and reference their parent larger chunks. Alternatively, individual sentences could be retrieved along with their surrounding text.

#### IV-A2 Structure Organization

One effective method for enhancing information retrieval is to establish a hierarchical structure for the documents. By constructing chunks structure, RAG system can expedite the retrieval and processing of pertinent data.

Hierarchical Index. In the hierarchical structure of documents, nodes are arranged in parent-child relationships, with chunks linked to them. Data summaries are stored at each node, aiding in the swift traversal of data and assisting the RAG system in determining which chunks to extract. This approach can also mitigate the illusion caused by chunk extraction issues. The methods for constructing a structured index primarily include: 1) Structural awareness based on paragraph and sentence segmentation in docs. 2) Content awareness based on inherent structure in PDF, HTML, and Latex. 3) Semantic awareness based on semantic recognition and segmentation of text.

KG Index [34]. Using Knowledge Graphs (KGs) to structure documents helps maintain consistency by clarifying connections between concepts and entities, reducing the risk of mismatch errors. KGs also transform information retrieval into instructions intelligible to language models, improving retrieval accuracy and enabling contextually coherent responses. This enhances the overall efficiency of the RAG system. For example, organizing a corpus in the format of graph G={𝒱,ℰ,𝒳}G=\{\mathcal{V},\mathcal{E},\mathcal{X}\}, where node V={vi}i=1nV=\{v_{i}\}^{n}_{i=1} represent document structures (e.g.passage, pages, table) , edge ℰ⊂𝒱×𝒱\mathcal{E}\subset\mathcal{V}\times\mathcal{V} represent semantic or lexical similarity and belonging relations, and node features 𝒳={𝒳i}i=1n\mathcal{X}=\{\mathcal{X}_{i}\}^{n}_{i=1} represent text or markdown content for passage.

### IV-B Pre-retrieval

One of the primary challenges with Naive RAG is its direct reliance on the user’s original query as the basis for retrieval. Formulating a precise and clear question is difficult, and imprudent queries result in subpar retrieval effectiveness. The primary challenges in this module include: 1) Poorly worded queries. The question itself is complex, and the language is not well-organized. 2) Language complexity and ambiguity. Language models often struggle when dealing with specialized vocabulary or ambiguous abbreviations with multiple meanings. For instance, they may not discern whether LLM refers to Large Language Model or a Master of Laws in a legal context.

#### IV-B1 Query Expansion

Expanding a single query into multiple queries enriches the content of the query, providing further context to address any lack of specific nuances, thereby ensuring the optimal relevance of the generated answers.

| fq​e​(q)={q1,q2,…,qn}∀qi∈{q1,q2,…,qn},qi∉Qf_{qe}(q)=\{q_{1},q_{2},\ldots,q_{n}\}\quad\forall q_{i}\in\{q_{1},q_{2},\ldots,q_{n}\},q_{i}\notin Q | (7) |
| --- | --- |

Multi-Query uses prompt engineering to expand queries via LLMs, allowing for parallel execution. These expansions are meticulously designed to ensure diversity and coverage. However, this approach can dilute the user’s original intent. To mitigate this, the model can be instructed to assign greater weight to the original query.

Sub-Query. By decomposing and planning for complex problems, multiple sub-problems are generated. Specifically, least-to-most prompting [35] can be employed to decompose the complex problem into a series of simpler sub-problems. Depending on the structure of the original problem, the generated sub-problems can be executed in parallel or sequentially. Another approach involves the use of the Chain-of-Verification (CoVe) [36]. The expanded queries undergo validation by LLM to achieve the effect of reducing hallucinations.

#### IV-B2 Query Transformation

Retrieve and generate based on a transformed query instead of the user’s original query.

| fq​t​(q)=q′f_{qt}(q)=q^{\prime} | (8) |
| --- | --- |

Rewrite. Original queries often fall short for retrieval in real-world scenarios. To address this, LLMs can be prompted to rewrite. Specialized smaller models can also be employed for this purpose [24]. The implementation of the query rewrite method in Taobao has significantly improved recall effectiveness for long-tail queries, leading to an increase in GMV [10].

HyDE [37]. In order to bridge the semantic gap between questions and answers, it constructs hypothetical documents (assumed answers) when responding to queries instead of directly searching the query. It focuses on embedding similarity from answer to answer rather than seeking embedding similarity for the problem or query. In addition, it also includes reverse HyDE, which generate hypothetical query for each chunks and focuses on retrieval from query to query.

Step-back Prompting [38]. The original query is abstracted into a high-level concept question (step-back question). In the RAG system, both the step-back question and the original query are used for retrieval, and their results are combined to generate the language model’s answer.

#### IV-B3 Query Construction

In addition to text data, an increasing amount of structured data, such as tables and graph data, is being integrated into RAG systems. To accommodate various data types, it is necessary to restructure the user’s query. This involve converting the query into another query language to access alternative data sources, with common methods including Text-to-SQL or Text-to-Cypher . In many scenarios, structured query languages (e.g., SQL, Cypher) are often used in conjunction with semantic information and metadata to construct more complex queries.

| fq​c​(q)=q∗,q∗∈Q∗={S​Q​L,C​y​p​h​e​r,…}f_{qc}(q)=q^{*},q^{*}\in Q^{*}=\{SQL,Cypher,\dots\} | (9) |
| --- | --- |

### IV-C Retrieval

The retrieval process is pivotal in RAG systems. By leveraging powerful embedding models, queries and text can be efficiently represented in latent spaces, which facilitates the establishment of semantic similarity between questions and documents, thereby enhancing retrieval. Three main considerations that need to be addressed include retrieval efficiency, quality, and the alignment of tasks, data and models.

#### IV-C1 Retriever Selection

With the widespread adoption of RAG technology, the development of embedding models has been in full swing. In addition to traditional models based on statistics and pre-trained models based on the encoder structure, embedding models fine-tuned on LLMs have also demonstrated powerful capabilities [39]. However, they often come with more parameters, leading to weaker inference and retrieval efficiency. Therefore, it is crucial to select the appropriate retriever based on different task scenarios.

Sparse Retriever uses statistical methods to convert queries and documents into sparse vectors. Its advantage lies in its efficiency in handling large datasets, focusing only on non-zero elements. However, it may be less effective than dense vectors in capturing complex semantics. Common methods include TF-IDF and BM25.

Dense Retriever employs pre-trained language models (PLMs) to provide dense representations of queries and documents. Despite higher computational and storage costs, it offers more complex semantic representations. Typical models include BERT structure PLMs, like ColBERT, and multi-task fine-tuned models like BGE [40] and GTE [41].

Hybrid Retriever is to use both sparse and dense retrievers simultaneously. Two embedding techniques complement each other to enhance retrieval effectiveness. Sparse retriever can provide initial screening results. Additionally, sparse models enhance the zero-shot retrieval capabilities of dense models, particularly in handling queries with rare entities, thereby increasing system robustness.

#### IV-C2 Retriever Fine-tuning

In cases where the context may diverge from pre-trained corpus, particularly in highly specialized fields like healthcare, law, and other domains abundant in proprietary terminology. While this adjustment demands additional effort, it can substantially enhance retrieval efficiency and domain alignment.

Supervised Fine-Tuning (SFT). Fine-tuning a retrieval model based on labeled domain data is typically done using contrastive learning. This involves reducing the distance between positive samples while increasing the distance between negative samples. The commonly used loss calculation is shown in the following:

| ℒ​(𝒟R)=−1T​∑i=1Tlog⁡e(s​i​m​(qi,di+))e(s​i​m​(qi,di+))+∑j=1Ne(s​i​m​(qi,di−))\mathcal{L}(\mathcal{D}_{R})=-\frac{1}{T}\sum_{i=1}^{T}\log\frac{e^{(sim(q_{i},d_{i}^{+}))}}{e^{(sim(q_{i},d_{i}^{+}))}+\sum_{j=1}^{N}e^{(sim(q_{i},d_{i}^{-}))}} | (10) |
| --- | --- |

where di+d_{i}^{+} is the positive sample document corresponding to the i-th query, di−d_{i}^{-} is several negative sample, TT is the total number of queries, NN is the number of negative samples, and 𝒟R\mathcal{D}_{R} is the fine-tuning dataset.

LM-supervised Retriever (LSR). In contrast to directly constructing a fine-tuning dataset from the dataset, LSR utilizes the LM-generated results as supervisory signals to fine-tune the embedding model during the RAG process.

| PL​S​R​(d|q,y)=ePL​M​(y|d,q)/β∑d′∈DePL​M(y|d,q)/β)P_{LSR}(d|q,y)=\frac{e^{P_{LM}(y|d,q)/\beta}}{\sum_{d^{\prime}\in D}{e^{P_{LM}(y|d,q)/\beta)}}} | (11) |
| --- | --- |

PL​M​(y|d,q)P_{LM}(y|d,q) is LM probability of the ground truth output yy given the input context dd and query qq, and β\beta is a hyperparamter.

Adapter. At times, fine-tuning a large retriever can be costly, especially when dealing with retrievers based on LLMs like gte-Qwen. In such cases, it can mitigate this by incorporating an adapter module and conducting fine-tuning. Another benefit of adding an adapter is the ability to achieve better alignment with specific downstream tasks [42].

### IV-D Post-retrieval

Feeding all retrieved chunks directly into the LLM is not an optimal choice. Post-processing the chunks can aid in better leveraging the contextual information. The primary challenges include: 1) Lost in the middle. Like humans, LLM tends to remember only the beginning or the end of long texts, while forgetting the middle portion [43]. 2) Noise/anti-fact chunks. Retrieved noisy or factually contradictory documents can impact the final retrieval generation [44]. 3) Context Window. Despite retrieving a substantial amount of relevant content, the limitation on the length of contextual information in large models prevents the inclusion of all this content.

#### IV-D1 Rerank

Rerank the retrieved chunks without altering their content or length, to enhance the visibility of the more crucial document chunks. Given the retrieved set DqD^{q} and a re-ranking method fr​e​r​a​n​kf_{rerank} to obtain the re-ranked set:

| Drq=fr​e​r​a​n​k​(q,Dq)={d1′,d2′,…,dk′}where​f​(d1′)≥f​(d2′)≥…≥f​(dk′).D^{q}_{r}=f_{rerank}(q,D^{q})=\{d_{1}^{\prime},d_{2}^{\prime},\ldots,d_{k}^{\prime}\}\\ \text{where}f(d^{\prime}_{1})\geq f(d^{\prime}_{2})\geq\ldots\geq f(d^{\prime}_{k}). | (12) |
| --- | --- |

Rule-base rerank. Metrics are calculated to rerank chunks according to certain rules. Common metrics include: diversity, relevance and MRR (Maximal Marginal Relevance) [45]. The idea is to reduce redundancy and increase result diversity. MMR selects phrases for the final key phrase list based on a combined criterion of query relevance and information novelty.

Model-base rerank. Utilize a language model to reorder the document chunks, commonly based on the relevance between the chunks and the query. Rerank models have become an important component of RAG systems, and relevant model technologies are also being iteratively upgraded. The scope reordering has also been extended to multimodal data such as tables and images [46].

#### IV-D2 Compression

A common misconception in the RAG process is the belief that retrieving as many relevant documents as possible and concatenating them to form a lengthy retrieval prompt is beneficial. However, excessive context can introduce more noise, diminishing the LLM’s perception of key information. A common approach to address this is to compress and select the retrieved content.

| Dcq=fc​o​m​p​(q,Dq),where​|diqc|<|diq|∀diq∈DqD^{q}_{c}=f_{comp}(q,D^{q}),\quad\text{where}|d_{i}^{q_{c}}|<|d_{i}^{q}|\quad\forall d_{i}^{q}\in D^{q} | (13) |
| --- | --- |

(Long)LLMLingua [47]. By utilizing aligned and trained small language models, such as GPT-2 Small or LLaMA-7B, the detection and removal of unimportant tokens from the prompt is achieved, transforming it into a form that is challenging for humans to comprehend but well understood by LLMs. This approach presents a direct and practical method for prompt compression, eliminating the need for additional training of LLMs while balancing language integrity and compression ratio.

#### IV-D3 Selection

Unlike compressing the content of document chunks, Selection directly removes irrelevant chunks.

| Dsq=fs​e​l​(Dq)={di∈D∣¬P​(di)}D^{q}_{s}=f_{sel}(D^{q})=\{d_{i}\in D\mid\neg P(d_{i})\} | (14) |
| --- | --- |

Where fs​e​lf_{sel} is the function for deletion operation and P​(di)P(d_{i}) is a conditional predicate indicating that document (di)d_{i}) satisfies a certain condition. If document (did_{i}) satisfies (P(diP(d_{i})), it will be deleted. Conversely, documents for which (¬P​(di)\neg P(d_{i})) is true will be retained.

Selective Context. By identifying and removing redundant content in the input context, the input is refined, thus improving the language model’s reasoning efficiency. In practice, selective context assesses the information content of lexical units based on the self-information computed by the base language model. By retaining content with higher self-information, this method offers a more concise and efficient textual representation, without compromising their performance across diverse applications. However, it overlooks the interdependence between compressed content and the alignment between the targeted language model and the small language model utilized for prompting compression [48].

LLM-Critique. Another straightforward and effective approach involves having the LLM evaluate the retrieved content before generating the final answer. This allows the LLM to filter out documents with poor relevance through LLM critique. For instance, in Chatlaw [49], the LLM is prompted to self-suggestion on the referenced legal provisions to assess their relevance.

### IV-E Generation

Utilize the LLM to generate answers based on the user’s query and the retrieved contextual information. Select an appropriate model based on the task requirements, considering factors such as the need for fine-tuning, inference efficiency, and privacy protection.

#### IV-E1 Generator Fine-tuning

In addition to direct LLM usage, targeted fine-tuning based on the scenario and data characteristics can yield better results. This is also one of the greatest advantages of using an on-premise setup LLMs.

Instruct-Tuning. When LLMs lack data in a specific domain, additional knowledge can be provided to the LLM through fine-tuning. General fine-tuning dataset can also be used as an initial step. Another benefit of fine-tuning is the ability to adjust the model’s input and output. For example, it can enable LLM to adapt to specific data formats and generate responses in a particular style as instructed [50].

Reinforcement learning. Aligning LLM outputs with human or retriever preferences through reinforcement learning is a potential approach [51]. For instance, manually annotating the final generated answers and then providing feedback through reinforcement learning. In addition to aligning with human preferences, it is also possible to align with the preferences of fine-tuned models and retrievers.

Dual Fine-tuing Fine-tuning both generator and retriever simultaneously to align their preferences. A typical approach, such as RA-DIT [27], aligns the scoring functions between retriever and generator using KL divergence. Retrieval likelihood of each retrieved document d is calculated as :

| PR​(d|q)=e(s​i​m​(d,q))/γ∑d∈𝒟​qe(sim(d,q)/γP_{R}(d|q)=\frac{e^{(sim(d,q))}/\gamma}{\sum_{d\in\mathcal{D}q}e^{(sim(d,q)/\gamma}} | (15) |
| --- | --- |

PL​M​(y|d,q)P_{LM}(y|d,q) is the LM probability of the ground truth output y given the input context dd, question qq, and γ\gamma is a hyperparamter. The overall loss is calculated as:

| ℒ=1|T|∑i=1TKL(PR(d|q)||PL​S​R(d|q,y|))\mathcal{L}=\frac{1}{|T|}\sum_{i=1}^{T}KL(P_{R}(d|q)||P_{LSR}(d|q,y|)) | (16) |
| --- | --- |

#### IV-E2 Verification

Although RAG enhances the reliability of LLM-generated answers, in many scenarios, it requires to minimize the probability of hallucinations. Therefore, it can filter out responses that do not meet the required standards through additional verification module. Common verification methods include knowledge-base and model-base .

| yk=fv​e​r​i​f​y​(q,Dq,y)y^{k}=f_{verify}(q,D^{q},y) | (17) |
| --- | --- |

Knowledge-base verification refers to directly validating the responses generated by LLMs through external knowledge. Generally, it extracts specific statements or triplets from response first. Then, relevant evidence is retrieved from verified knowledge base such as Wikipedia or specific knowledge graphs. Finally, each statement is incrementally compared with the evidence to determine whether the statement is supported, refuted, or if there is insufficient information [52].

Model-based verification refers to using a small language model to verify the responses generated by LLMs [53]. Given the input question, the retrieved knowledge, and the generated answer, a small language model is trained to determine whether the generated answer correctly reflects the retrieved knowledge. This process is framed as a multiple-choice question, where the verifier needs to judge whether the answer reflects correct answer . If the generated answer does not correctly reflect the retrieved knowledge, the answer can be iteratively regenerated until the verifier confirms that the answer is correct.

### IV-F Orchestration

Orchestration pertains to the control modules that govern the RAG process. Unlike the traditional, rigid approach of a fixed process, RAG now incorporates decision-making at pivotal junctures and dynamically selects subsequent steps contingent upon the previous outcomes. This adaptive and modular capability is a hallmark of modular RAG, distinguishing it from the more simplistic Naive and Advance RAG paradigm.

#### IV-F1 Routing

In response to diverse queries, the RAG system routes to specific pipelines tailored for different scenario, a feature essential for a versatile RAG architecture designed to handle a wide array of situations. A decision-making mechanism is necessary to ascertain which modules will be engaged, based on the input from the model or supplementary metadata. Different routes are employed for distinct prompts or components. This routing mechanism is executed through a function, denoted as fr​(⋅)f_{r}(\cdot), which assigns a score αi\alpha_{i} to each module. These scores dictate the selection of the active subset of modules. Mathematically, the routing function is represented as:

| fr:Q→ℱf_{r}:Q\rightarrow\mathcal{F} | (18) |
| --- | --- |

where fr​(⋅)f_{r}(\cdot) maps the identified query to its corresponding RAG flow.

Metadata routing involves extracting key terms, or entities, from the query, applying a filtration process that uses these keywords and associated metadata within the chunks to refine the routing parameters. For a specific RAG flow, denoted as FiF_{i}, the pre-defined routing keywords are represented as the set 𝒦i={ki​1,ki​2,…,ki​n}\mathcal{K}_{i}=\{k_{i1},k_{i2},\ldots,k_{in}\}. The keyword identified within the query qiq_{i} is designated as Ki′K^{\prime}_{i}. The matching process for the query qq is quantified by the key score equation:

| scorekey​(qi,Fj)=1|Kj′|​|𝒦i∩Kj′|\text{score}_{\text{key}}(q_{i},F_{j})=\frac{1}{|K^{\prime}_{j}|}|\mathcal{K}_{i}\cap K^{\prime}_{j}| | (19) |
| --- | --- |

This equation calculates the overlap between the pre-defined keywords and those identified in the query, normalized by the count of keywords in Kj′K^{\prime}_{j}. The final step is to determine the most relevant flow for the query qq:

| Fi​(q)=argmaxFj∈ℱ​score​(q,Fj)F_{i}(q)=\text{argmax}_{F_{j}\in\mathcal{F}}\text{score}(q,F_{j}) | (20) |
| --- | --- |

Semantic routing routes to different modules based on the semantic information of the query. Given a pre-defined intent Θ={θ1,θ2,…,θn}\Theta=\{\theta_{1},\theta_{2},\ldots,\theta_{n}\}, the possibility of intent for query q is PΘ​(θ|q)=ePL​M​(θ|q)∑θ∈ΘePL​M(θ|q))P_{\Theta}(\theta|q)=\frac{e^{P_{LM}(\theta|q)}}{\sum_{\theta\in\Theta}{e^{P_{LM}(\theta|q))}}}. Routing to specific RAG flow is determined by the semantic score:

| s​o​c​r​es​e​m​a​n​t​i​c​(q,Fj)=a​r​g​m​a​xθj∈Θ​P​(Θ)socre_{semantic}(q,F_{j})=argmax_{\theta_{j}\in\Theta}P(\Theta) | (21) |
| --- | --- |

The function δ​(⋅)\delta(\cdot) serves as a mapping function that assigns an intent to a distinct RAG flow ℱi=δ​(θi)\mathcal{F}_{i}=\delta(\theta_{i})

Hybrid Routing can be implemented to improve query routing by integrating both semantic analysis and metadata-based approaches, which can be defined as follows:

| αi=a⋅s​c​o​r​ek​e​y​(q,Fj)+(1−α)⋅m​a​xθj∈Θ​s​o​c​r​es​e​m​a​n​t​i​c​(q,Fj)\alpha_{i}=a\cdot score_{key}(q,F_{j})+(1-\alpha)\cdot max_{\theta_{j}\in\Theta}socre_{semantic}(q,F_{j}) | (22) |
| --- | --- |

aa is a weighting factor that balances the contribution of the key-based score and the semantic score.

#### IV-F2 Scheduling

The RAG system evolves in complexity and adaptability, with the ability to manage processes through a sophisticated scheduling module. The scheduling module plays a crucial role in the modular RAG , identifying critical junctures that require external data retrieval, assessing the adequacy of the responses, and deciding on the necessity for further investigation. It is commonly utilized in scenarios that involve recursive, iterative, and adaptive retrieval, ensuring that the system makes informed decisions on when to cease generation or initiate a new retrieval loop.

Rule judge. The subsequent steps are dictated by a set of established rules. Typically, the system evaluates the quality of generated answers through scoring mechanisms. The decision to proceed or halt the process is contingent upon whether these scores surpass certain predetermined thresholds, often related to the confidence levels of individual tokens, which can be defined as follow:

| yt={s^t if all tokens of ​s^t​ have probs≥τst=L​M​([Dqt,x,y ](javascript:toggleReadingMode();)