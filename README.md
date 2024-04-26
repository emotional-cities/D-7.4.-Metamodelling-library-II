# D7.2-Metamodelling

This is the second version of the metamodelling library for eMotional Cities project.

•	LHS.py: performs LHS sampling in an input space of three variables.

•	Model.py: compute the number of leisure trips given an x value with the changes in some beta values (input space of uncertainty). This could be seen as the simulation for this simplified example.

•	ModelSD.py: this file combines the two previous ones. The user can set the number of simulations runs, and it directly saves the LHS input values and the corresponding output in two different files.

•	GaussianProcess.ipynb: explains and constructs a Gaussian Process metamodel for the simulation model with 400 labeled points. After, it computes the prediction in another 400 unlabeled input points.

•	GP_AL.ipynb: explains and constructs a Gaussian Process metamodel for the simulation model using Active learning. It computes the prediction in 400 unlabeled input points using for the estimation only 200 labeled points and 100 query points.

•	Visualizations.ipynb: Notebook employed to visualize the results of the metamodel by highlighting the regions where the output is higher.

•	RFMetamodel.py: Random Forest metamodelling of the Day Patter Model using cross validation.