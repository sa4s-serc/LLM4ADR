# DRAFT-ing Architectural Design Decisions using LLMs

This repository contains codes and data for our paper [DRAFT-ing Architectural Design Decisions using LLMs](https://arxiv.org/abs/2504.08207)

In this paper we cam up with a novel approach DRAFT (Domain Specific Retreival Augumented Few Shot Fine Tuninng), to generate Architectural Design Decisions in the context of ADRs (Architecture Decision Records).


## Abstract

Architectural Knowledge Management (AKM) is crucial for software development but remains challenging due to the lack of standardization and high manual effort. Architecture Decision Records (ADRs) provide a structured approach to capture Architecture Design Decisions (ADDs), but their adoption is limited due to the manual effort involved and insufficient tool support. Our previous work has shown that Large Language Models (LLMs) can assist in generating ADDs. However, simply prompting the LLM does not produce quality ADDs. Moreover, using third-party LLMs raises privacy concerns, while self-hosting them poses resource challenges.

To this end, we experimented with different approaches like few-shot, retrieval-augmented generation (RAG) and fine-tuning to enhance LLM's ability to generate ADDs. Our results show that both techniques improve effectiveness. Building on this, we propose Domain Specific Retreival Augumented Few Shot Fine Tuninng, DRAFT, which combines the strengths of all these three approaches for more effective ADD generation. DRAFT operates in two phases: an offline phase that fine-tunes an LLM on generating ADDs augmented with retrieved examples and an online phase that generates ADDs by leveraging retrieved ADRs and the fine-tuned model.

We evaluated DRAFT against existing approaches on a dataset of 4,911 ADRs and various LLMs and analyzed them using automated metrics and human evaluations. Results show DRAFT outperforms all other approaches in effectiveness while maintaining efficiency. Our findings indicate that DRAFT can aid architects in drafting ADDs while addressing privacy and resource constraints.

Here is our ![Graphical Abstract](Diagram/graphical_abstract.pdf)

## Repository Description

### Diagrams
The various images used in the paper are kept in the 'Diagrams' Directory.

### Data
The data used in this study consists of 4,911 ADRs, sourced from a previous study [Using Architecture Decision Records in Open Source Projects—An MSR Study on GitHub](https://ieeexplore.ieee.org/document/10155430).
The extracted data along with the code is given in the 'Data' directory.

### LLMs
The LLMs used the study were picked from the rankings in [Chatbot Arena (formerly LMSYS)](https://lmarena.ai/) with some filtration criteria.
The details are given in the 'LLMs' directory.

