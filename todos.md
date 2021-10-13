# Project 1 todos

Vi må fikse SEED_VALUE

## Exercise 1
Add R2 metrics 


## Exercise 2
feil konvergerer som 1/sqrt(n) (kanskje). (MonteCarlo)
sett i lys av computational heavy

What separates Machine Learning from optimization
- low generalization error as well (Goodfellow p.107).
- Motivation for bootstrap (and kfold CV)

note about iid assumption

underfitting and overfitting

relate the bias-variance discussion to figure 2.11 Hastie

## Exercise 3
Kommenter overlapp i standardavvik i CV og bootstrap.. ref degree4
Complexity degree 6, low variance between kfold splits

Compare bootstrap MSE for one degree with all kfold mean MSE for one degree.

Støyete vs. lite støyete datasett, ("Det har ikke så mye å si")
Usikkerheten går opp med flere og flere folds, for antall datapunkter går ned. 

Test bootstrap for n > 100. MSE for bootstrap blir mye bedre fordi sjansen for å sample samme punkt 2 
ganger blir mye mindre med større dataset! 

Compare and plot with sk_learn!!!


## Exercise 4-5
3D grid search,
lambdaplot for optimal degree

Section 5.2.2 in Goodfellow

vår fit opp mot sklearn

obs obs: 
Felles for oppgave 4 og 5: BOOTSTRAP
Kun for oppgavbe 4: CROSS-VAL

plotte vertikal linje for optimal lambda i beta-plot

## Exercise 6
Tenk på funksjonen man fitter som en kompresjon av terrenget. 

Curse of dimensionality, "The more dimensions the training data set has, the greater the risk of overfitting is"[Hands on scikit p215]

### EX 6.3
k-fold 5 analysis of model degree 2 - 7. Check robustness of MSE