/**
 * Default corpus and queries. Direct port of `src/defaults.rs`.
 * Embeddings are 8-dimensional f64 literals (design doc §14 Q4).
 */
import { Corpus } from "./corpus.js";
import { Tokenizer } from "./tokenizer.js";

interface DocumentDef {
  id: string;
  text: string;
  embedding: number[];
}

export interface DefaultQuery {
  text: string;
  terms: string[];
  embedding: number[] | null;
  relevant: string[];
}

const DOCUMENTS: DocumentDef[] = [
  {
    id: "d01",
    text: "Machine learning algorithms learn patterns from data using statistical methods",
    embedding: [0.9, 0.3, 0.1, 0.0, 0.1, 0.0, 0.4, 0.1],
  },
  {
    id: "d02",
    text: "Deep learning neural networks require large training datasets for supervised learning",
    embedding: [0.8, 0.9, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3],
  },
  {
    id: "d03",
    text: "Unsupervised learning discovers hidden structure in unlabeled data",
    embedding: [0.9, 0.4, 0.0, 0.0, 0.1, 0.0, 0.3, 0.2],
  },
  {
    id: "d04",
    text: "Reinforcement learning agents maximize cumulative reward through exploration",
    embedding: [0.8, 0.5, 0.0, 0.0, 0.0, 0.1, 0.3, 0.0],
  },
  {
    id: "d05",
    text: "Transfer learning adapts pre-trained models to new domains with limited data",
    embedding: [0.9, 0.7, 0.1, 0.0, 0.0, 0.0, 0.2, 0.3],
  },
  {
    id: "d06",
    text: "Information retrieval systems search and rank documents by relevance to queries",
    embedding: [0.1, 0.0, 0.9, 0.8, 0.0, 0.0, 0.2, 0.1],
  },
  {
    id: "d07",
    text: "BM25 is a bag of words retrieval function that ranks documents based on term frequency",
    embedding: [0.1, 0.0, 0.8, 0.9, 0.0, 0.0, 0.3, 0.0],
  },
  {
    id: "d08",
    text: "TF-IDF weighting reflects how important a word is to a document in a collection",
    embedding: [0.1, 0.0, 0.8, 0.7, 0.0, 0.0, 0.2, 0.0],
  },
  {
    id: "d09",
    text: "Query expansion improves search recall by adding related terms to the original query",
    embedding: [0.2, 0.0, 0.9, 0.6, 0.0, 0.0, 0.1, 0.1],
  },
  {
    id: "d10",
    text: "Relevance feedback uses explicit user judgments to improve retrieval performance",
    embedding: [0.2, 0.0, 0.8, 0.7, 0.0, 0.0, 0.2, 0.0],
  },
  {
    id: "d11",
    text: "Relational databases store data in tables with SQL as the query language",
    embedding: [0.0, 0.0, 0.1, 0.0, 0.9, 0.2, 0.0, 0.0],
  },
  {
    id: "d12",
    text: "NoSQL databases provide flexible schema design for unstructured data storage",
    embedding: [0.0, 0.0, 0.1, 0.0, 0.9, 0.3, 0.0, 0.0],
  },
  {
    id: "d13",
    text: "Database indexing structures like B-trees accelerate data retrieval operations",
    embedding: [0.0, 0.0, 0.3, 0.1, 0.9, 0.1, 0.0, 0.0],
  },
  {
    id: "d14",
    text: "Transaction processing ensures ACID properties for reliable data operations",
    embedding: [0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0],
  },
  {
    id: "d15",
    text: "Distributed databases partition data across multiple nodes for scalability",
    embedding: [0.0, 0.0, 0.1, 0.0, 0.8, 0.9, 0.0, 0.0],
  },
  {
    id: "d16",
    text: "Vector search uses embedding similarity to find semantically related documents",
    embedding: [0.3, 0.3, 0.7, 0.5, 0.1, 0.0, 0.2, 0.9],
  },
  {
    id: "d17",
    text: "Hybrid search combines lexical matching with vector similarity for better retrieval",
    embedding: [0.2, 0.2, 0.8, 0.6, 0.0, 0.0, 0.3, 0.8],
  },
  {
    id: "d18",
    text: "Bayesian probability provides a framework for updating beliefs with new evidence",
    embedding: [0.3, 0.1, 0.2, 0.2, 0.0, 0.0, 0.9, 0.1],
  },
  {
    id: "d19",
    text: "Probabilistic models estimate relevance scores using statistical inference methods",
    embedding: [0.4, 0.1, 0.5, 0.4, 0.0, 0.0, 0.8, 0.2],
  },
  {
    id: "d20",
    text: "Cosine similarity measures the angle between two vectors in high-dimensional space",
    embedding: [0.2, 0.1, 0.3, 0.2, 0.0, 0.0, 0.3, 0.9],
  },
];

export function buildDefaultQueries(): DefaultQuery[] {
  return [
    {
      text: "machine learning",
      terms: ["machine", "learning"],
      embedding: [0.9, 0.5, 0.1, 0.0, 0.0, 0.0, 0.3, 0.2],
      relevant: ["d01", "d02", "d03", "d04", "d05"],
    },
    {
      text: "Bayesian probability",
      terms: ["bayesian", "probability"],
      embedding: [0.3, 0.1, 0.2, 0.2, 0.0, 0.0, 0.9, 0.1],
      relevant: ["d18", "d19"],
    },
    {
      text: "search",
      terms: ["search"],
      embedding: [0.1, 0.0, 0.9, 0.6, 0.0, 0.0, 0.1, 0.3],
      relevant: ["d06", "d09", "d16", "d17"],
    },
    {
      text: "transaction processing",
      terms: ["transaction", "processing"],
      embedding: [0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0],
      relevant: ["d14"],
    },
    {
      text: "data",
      terms: ["data"],
      embedding: [0.4, 0.2, 0.3, 0.1, 0.4, 0.2, 0.2, 0.2],
      relevant: ["d01", "d03", "d05", "d11", "d12", "d13", "d14", "d15"],
    },
    {
      text: "vector search embeddings",
      terms: ["vector", "search", "embeddings"],
      embedding: [0.2, 0.2, 0.7, 0.4, 0.0, 0.0, 0.2, 0.9],
      relevant: ["d16", "d17", "d20"],
    },
    {
      text: "retrieval augmented generation",
      terms: ["retrieval", "augmented", "generation"],
      embedding: [0.4, 0.4, 0.7, 0.5, 0.0, 0.0, 0.2, 0.4],
      relevant: ["d06", "d07", "d10", "d17"],
    },
  ];
}

export function buildDefaultCorpus(): Corpus {
  const corpus = new Corpus(new Tokenizer());
  for (const doc of DOCUMENTS) {
    corpus.addDocument(doc.id, doc.text, doc.embedding);
  }
  corpus.buildIndex();
  return corpus;
}
