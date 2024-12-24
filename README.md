
<div align="center" style="padding-bottom: 20px">
    <img src="./artifacts/Logo.png" alt="logo" height="400"/>
</div>

# Community Detection and Recommendation System for a Question - Solver matching problem 
## GNN-based approach to explore stack-overflow and other communities.

## Overview

The system performs the following steps:

1. **Graph Building**:
   - **Nodes**: There two components: one represent individual users (`solvers`), other questions.
   - **User-User Edges**: There are two approaches to form those edges:
     - **Tags similarity** of their profile badges.
     - **Question interests**: if two users have non-negative rated answers on same question.
   - **Question-Question Edges**: this edges created based on:
     - **Tags similarity** of their topic.
     - **Embedding similarity** of the question content.
     - **Complexity metrics** such as emergening of technical words, text len, appearance of negative answers (`for complexity communities searching`).
   - **Question-User Edges** based on non-negative answers.

2. **Community Formation**: 
   - GNNs are applied to detect communities of users and questions separately. Then stereotypes are creating based on tags, complexity coefficient, etc.

3. **Solver Searching**: 
   - New question, ranks users closest to average of found stereotype, to find most appropriate solver.

## Key results

|                 Approach                  |     Score    |
|-------------------------------------------|--------------|
|            Community stereotype           |      0.60    |
|                 CatBoost                  |       -      |
|                 LightGCN                  |       -      |
| LightGCN + Question complexity stochastic |       -      |

## Installation & Data load

To install and run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/setday/hse_sna_p2024
cd hse_sna_p2024
pip install -r requirements.txt
```

Then download dataset using:

```bash
utils\load_data.sh
```

## Dataset

We used open archived datasets of `archive.org` platform from two stack-exchange platforms - `softwareengineering.stackexchange.com` and `networkengineering.stackexchange.com`.
