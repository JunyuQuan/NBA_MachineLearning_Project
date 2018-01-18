# NBA_MachineLearning_Project

For ECE5424(Adv. Machine Learning), use Scikit-learn, perform analyses on per player NBA data, including:

1. Reduce dimensionality of player statistics utilizing Principle Component Analysis
(PCA). PCA can be applied to the data in order to compress the n-dimensional statistics
pertaining to a player's per game offensive, defensive, or overall contribution to their team's
success to a k-dimensional representation, where k < n.

2. Build a Support Vector Machine(SVM) that takes the reduced dimensional player
statistics to predict the winner of head to head matchups. As a maximum margin,
nonlinear classifier an SVM is an excellent option to use as a predictive tool.

3. Sort player contribution to team success using reduced dimension statistics against
team wins, in order to identity contributions of players with "intangible" skills.
Implementing a k-means clustering algorithm to group players by performance and team success
will identify players who help their team in ways that do not show in team box scores.
