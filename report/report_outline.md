# Report outline (fill with your real numbers from results/)

1. **Abstract** - problem, topic-aware recommender idea, headline result (routing vs best single model).
2. **Introduction & motivation** - why sarcasm detection is hard; why topic might matter.
3. **Related work** - sarcasm detection surveys, Twitter sarcasm, zero-shot topic classification.
4. **Data** - iSarcasmEval En train (3,468 rows originally; explain rows dropped by load_dataset.py
   - empty/duplicate text; 867 sarcastic). Class imbalance (~25% sarcastic).
5. **Topic tagging** - zero-shot BART-MNLI, 3 topics, descriptive labels; Part A confidence;
   Part B hand-check accuracy (n=60, with CI); limitation: short tweets have no topic.
6. **Preprocessing** - emoji->words, URL/user tokens, hashtag splitting, stop-words kept (justify).
7. **Experimental setup** - stratified 80/10/10 (label x topic), 5-fold CV tuning, F1(sarcastic) metric, no balancing.
8. **Models** - NB, LR, SVM, RF, KNN (TF-IDF 1-2 grams); BiGRU + frozen GloVe (+ trainable ablation).
9. **Results** - overall table (results/overall.csv with bootstrap CIs); per-topic table & figure;
   learning curves; error analysis.
10. **Recommender** - topic x model matrix, tie-break rule, routing vs best-single-model on test.
11. **Limitations** - small per-topic test sets (CIs wide); zero-shot tagging noise; single dataset;
    no contextual (transformer) sarcasm model; intended-sarcasm labels; Twitter-only; no human
    topic gold labels beyond 60 tweets.
12. **Conclusion & future work.**
13. **References.**

## Candidate references (verify every detail against the original before citing)
1. Abu Farha et al. (2022). SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic. SemEval-2022.
2. Joshi, Bhattacharyya & Carman (2017). Automatic Sarcasm Detection: A Survey. ACM Computing Surveys 50(5).
3. Davidov, Tsur & Rappoport (2010). Semi-Supervised Recognition of Sarcastic Sentences in Twitter and Amazon. CoNLL.
4. Gonzalez-Ibanez, Muresan & Wacholder (2011). Identifying Sarcasm in Twitter: A Closer Look. ACL.
5. Bamman & Smith (2015). Contextualized Sarcasm Detection on Twitter. ICWSM.
6. Pennington, Socher & Manning (2014). GloVe: Global Vectors for Word Representation. EMNLP.
7. Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP.
8. Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.
9. Lewis et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training... ACL.
10. Williams, Nangia & Bowman (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI). NAACL.
11. Yin, Hay & Roth (2019). Benchmarking Zero-shot Text Classification. EMNLP-IJCNLP.
12. Cortes & Vapnik (1995). Support-Vector Networks. Machine Learning.
13. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.

## Novelty check (your team's job)
Search Google Scholar for: "topic-aware sarcasm detection", "sarcasm detection per topic classifier selection",
"zero-shot topic sarcasm". Record query, date, and the closest 3 papers and how yours differs.
