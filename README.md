# Fusion-Permutation-Importance

A work done by Anita Narovlyansky and Daniel Goman

A project in an AI seminar

As data scientists, sometimes as important as accuracy in prediction is interpretability of machine learning models. 
In some cases, linear models are favored over more complex models because of their interpretability, even though they don’t always yield the best results. 
For this reason, we would like to derive effective estimators of feature relevance in more complex models, such as Random Forest (RF). 
The solution is the algorithm of Permutation Importance (PIMP). 

Despite the fact that feature importance approaches help in interpretation, there is a lack of consensus on how features are quantified, which makes the explanations unreliable. Combination of the results from multiple feature importance quantifiers reduces the variance of estimates and addressing the issue of the lack of agreement. Our assumption was that this will lead to more robust interpretations of the importance of each feature to the prediction. 

Our work is based on two feature importance algorithms: 

    Classic Permutation Importance
    
    Permutation Importance using P-value measure
    
We implement the two permutation importance algorithms, as well as a fusion model which uses both algorithms to output the final PI scores.

We proceed to display our results and findings, and discuss the implications of the results and our work, as well as potential improvements and ideas future work.
