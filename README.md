# MIC Predictor

Viruses remain an area of concern despite constant development of antiviral drugs and therapies. One of the contributors among others is the flaviviridae family of viruses. Like other spaces, antiviral peptides (AVP) are gaining importance for studying flaviviridae family. Along with antiviral properties of peptides, information about bioactivity takes it even closer to accurate predictions of peptide capabilities. Experimental identification of bioactivity of each potential peptide is an expensive and time consuming task. Computational methods like Proteochemometric modelling (PCM) are promising for prediction of bioactivity based on peptide and target sequence. The additional edge PCM methods bring in is the aspect of considering both peptide and target properties instead of only looking at peptide properties.  In this study, we propose prediction of pIC50 for AVP against flaviviridae family target proteins. The target proteins were manually curated from literature. Here we utilize the PCM descriptors as peptide descriptors, target descriptors and cross term descriptors.

**Usage**
```
python <script_name.py>
```

All scripts run hyper-parameter tuning, feature selection using adjusted R-squared and calculate Leave-one-out CV score for selected features with best hyper-parameters. Features used in each script is as detailed in below table


| scripts                                           | Features                                       |
|---------------------------------------------------|------------------------------------------------|
| peptide_prop_feature_selection_adj_r2             | Peptide properties                             |
| peptide_pep_tar_prop_feature_selection_adj_r2     | Peptide properties, target properties          |
| pcm_zscales_pep_tar_prop_feature_selection_adj_r2 | Zscales, peptide properties, target properties |
| pcm_zscales_pep_prop_feature_selection_adj_r2     | Zscales, peptide properties                    |
| pcm_zscales_feature_selection_adj_r2              | Zscales                                        |
| pcm_zscales_no_crossterm_feature_selection_adj_r2 | Zscales (without cross-terms)                  |