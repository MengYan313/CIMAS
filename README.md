# CIMAS: A Modular Framework for C++ Code Idiom Mining

## 🔍 Overview

**Code idioms**—common, reusable patterns or templates—play a crucial role in improving programming efficiency and code quality. However, the complex and semantically rich nature of the C++ language makes systematic idiom mining a challenging task.

**CIMAS** (Code Idiom Mining via Abstraction and Semantics) is a modular framework designed to address this gap by performing structured and progressive idiom mining from large-scale C++ repositories.

---

## 🧩 Key Components

CIMAS is composed of four sequential modules:

1. **Idiom Representation Identification**  
   Performs static analysis across the repository to detect candidate idiomatic code segments based on structural features.

2. **Idiomatic Code Mining**  
   Clusters code segments using vectorized embeddings to discover recurring idiomatic patterns across large repositories.

3. **Differentiating Element Abstraction**  
   Abstracts variable components such as constants, variable names, and function names while preserving invariant idiomatic structure.

4. **Idiom Judgment and Synthesis**  
   Utilizes Large Language Models (LLMs) to validate the semantic coherence of idioms and support idiom-level synthesis.

---

## 📊 Evaluation

To validate the effectiveness of CIMAS, we apply the mined idiom repository to two downstream tasks:

- ✅ **Unit Test Generation**  
- ✅ **Programming Specification Recognition**

### ✨ Results

- AST node count increased by an average of **38.8%** and **28.4%** across the two tasks, indicating enhanced semantic richness in generated content.
