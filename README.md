
The dissertation can be found [here](https://dspace.ut.ee/items/e18a76f7-5905-49ac-b398-1e216df6323f).

`data` -- folder containing texts from the etTenTen corpus.

`annotation_phd_project_vaik.csv` -- a CSV table with the judgements obtained from the annotation study. The judgements were made based on a four-point Likert scale (0 - not present, 1 - weakly present, 2 - moderately present, 3 - strongly present [moderate and strong were combined into a single label due to the uneven size]). The table includes the average rating for each text on all 12 dimensions. See Chapter 4 for more information.

`text_tagger.py` - performs morphological and syntactic analysis for the texts in the data folder and outputs the result in a single JSON file.

`extract_features.py` -- extracts the relative frequencies from the JSON file and saves the result in a single CSV file. See Chapter 5 for more information. 

`lexicons` -- specialized lexicons used for the feature extraction code.

`annotation_groups_per_dimension` -- separate folders for all 12 dimensions. Each folder includes six files -  `DIM_not_present.csv`, `DIM_not_present_ls`; `DIM_weak.csv`, `DIM_weak_ls`; and `DIM_strong.csv`, `DIM_strong_ls` - where CSV files include text URLs and the relative frequencies of the extracted features and files with the `_ls` suffix is a list of the text URLs. These CSV files are used as an input for get_statistics.py 

`get_statistics.ipynb` -- (messy notebook) code for ANOVA + post hoc tests, correlation analysis

