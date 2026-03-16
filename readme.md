# Data and code for the paper: "Equitable Alignment of LLMs Across Languages with Regularized Direct Preference Optimization"

The ***data*** folder contains all the data we used in our experiments and our splits:
- Multilingual Jigsaw dataset ([Original data](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data))
- MoralStories dataset [1] ([Original data](https://huggingface.co/datasets/LabHC/moral_stories))
- HistoiresMorales [2] ([Original data](https://huggingface.co/datasets/LabHC/histoires_morales))

The ***src*** folder contains:
- jigsaw_experiments.py to launch the experiments on the Jigsaw dataset.  
- moral_experiments.py to launch the experiments on the Moral corpus.
- PPLTrainer.py that contains our regularized DPO implementation based on [HuggingFace DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOTrainer).
- utils.py that contains the training and evaluation pipelines adapted from the code of [2] ([Original code](https://github.com/upunaprosk/histoires-morales)).

---
[1] Denis Emelin, Ronan Le Bras, Jena D. Hwang, Maxwell Forbes, and Yejin Choi. 2021. Moral Stories: Situated Reasoning about Norms, Intents, Actions, and their Consequences. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 698–718, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

[2] Thibaud Leteno, Irina Proskurina, Antoine Gourru, Julien Velcin, Charlotte Laclau, Guillaume Metzler, and Christophe Gravier. 2025. HISTOIRESMORALES: A French Dataset for Assessing Moral Alignment. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2590–2612, Albuquerque, New Mexico. Association for Computational Linguistics.
